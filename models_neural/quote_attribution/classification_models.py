import torch
from torch import nn
from models_neural.src.layers_sentence_embedding import PretrainedModelLoader, SentenceEmbeddingsLayer
from models_neural.src.utils_general import get_config
from models_neural.src.layers_head import HeadLayerBinaryFF, HeadLayerBinaryLSTM, HeadLayerBinaryTransformer
from models_neural.src.utils_lightning import LightningMixin, LightningLMSteps, LightningOptimizer, LightningQASteps
from torch.nn import CrossEntropyLoss

class SourceSentenceEmbeddingLayer(SentenceEmbeddingsLayer):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.person_embedding = nn.Embedding(2, self.config.embedding_dim)
        self.target_sentence_embedding = nn.Embedding(2, self.config.embedding_dim)
        if self.config.sentence_embedding_method == 'multiheaded-attention':
            from models_neural.src.layers_attention import TGMultiHeadedSelfAttention
            self.attention = TGMultiHeadedSelfAttention(
                self.config.hidden_dim,
                self.config.embedding_dim,
                8
            )

    def get_sentence_embs(self, word_embs, attention_mask):
        # aggregate
        if self.config.sentence_embedding_method == 'average':
            return self._avg_representation(word_embs, attention_mask)
        elif self.config.sentence_embedding_method == 'cls':
            return self._cls_token(word_embs, attention_mask)
        elif self.config.sentence_embedding_method == 'attention':
            return self._attention_representation(word_embs, attention_mask)
        elif self.config.sentence_embedding_method == 'multiheaded-attention':
            return self._multiheaded_attention(word_embs, attention_mask)
        else:
            raise NotImplementedError(
                'SENTENCE EMBEDDING METHOD %s not in {average, cls, attention, multiheaded-attention}' % self.config.sentence_embedding_method
            )

    def _multiheaded_attention(self, hidden, attention_mask):
        return self.attention(hidden, attention_mask)

    def forward(
            self,
            input_ids,
            target_sentence_ids,
            target_person_ids,
            attention_mask,
            *args, **kwargs
    ):
        """
        Helper method to calculate sentence embeddings for text.

        Parameters:
            * input_ids: normally, this is a matrix of size (len doc X max len sents) (unless sequence_lens is passed).
            * attention_mask: matrix of size (len doc X max len sents) with zeros to represent input_id padding.
            * target_sentence_ids
            * target_person_ids
            * attention_mask
        """
        word_embs = self._get_word_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        source_type_embs = self.person_embedding(target_person_ids)
        word_embs = word_embs + source_type_embs
        sent_embs = self.get_sentence_embs(word_embs, attention_mask)
        sentence_type_embs = self.target_sentence_embedding(target_sentence_ids)
        return sent_embs + sentence_type_embs


class SourceClassifier(LightningMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = get_config(kwargs=kwargs)
        self.transformer = SourceSentenceEmbeddingLayer(*args, **kwargs)
        self.head = self.get_head_layer()(*args, **kwargs)

    def get_head_layer(self):
        if self.config.num_contextual_layers == 0:
            return HeadLayerBinaryFF
        if self.config.context_layer == 'lstm':
            return HeadLayerBinaryLSTM
        elif self.config.context_layer == 'gpt2-sentence':
            return HeadLayerBinaryTransformer

    def forward(
            self,
            input_ids,
            target_sentence_ids,
            target_person_ids,
            labels=None,
            attention_mask=None,
            input_lens=None,
            *args,
            **kwargs
    ):
        """
        Step that's shared between training loop and validation loop. Contains sequence-specific processing,
        so we're keeping it in the child class.

        Parameters:
            * `input_ids`: list of docs, len(input_ids) = # batches (default = 1).
                Each item is a flat list of token-ids of length `num_toks_in_doc`.
            * `labels`: [optional] list of sentence-level labels of length batch_size.
                Each item contains tensor of labels length `num_sents_in_doc`.
            * `attention_mask`: [optional] list of attention matrices of length batch_size.
                Each item is a matrix of size `num_sents_in_doc` x `max_i[num tokens in sent i]`
            * `input_lens` [optional]: list of sentence-lengths of length `batch_size`.
                Each item is a tensor of length `num_sentences_in_doc`.


        Returns tuple of (loss, y_preds, y_trues)
         if labels is not None, else
         returns tuple of (None, y_preds, None)
        """
        sent_embeddings = self.transformer(input_ids, target_sentence_ids, target_person_ids, attention_mask)
        loss, prediction, labels = self.head.classification(sent_embeddings, labels)
        return loss


class SourceQA(LightningOptimizer, LightningQASteps):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = get_config(kwargs=kwargs)
        self.num_labels = 2
        pretrained_model_loader = PretrainedModelLoader(*args, **kwargs)
        self.encoder_model = pretrained_model_loader.encoder_model
        self.qa_outputs = nn.Linear(self.config.hidden_size, self.num_labels)
        self.target_sentence_embedding = nn.Embedding(2, self.config.embedding_dim)

    def forward(
            self,
            input_ids,
            sentence_ids,
            start_positions=None,
            end_positions=None,
            input_lens=None,
            attention_mask=None,
            *args,
            **kwargs
    ):

        outputs = self.encoder_model(input_ids, attention_mask=attention_mask)
        sentence_type_embs = self.target_sentence_embedding(sentence_ids)
        word_embs = outputs[0]
        word_embs = word_embs + sentence_type_embs

        logits = self.qa_outputs(word_embs)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        output = (start_logits, end_logits) + outputs[2:]
        return ((total_loss,) + output) if total_loss is not None else output