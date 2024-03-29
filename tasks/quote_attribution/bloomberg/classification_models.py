import torch
from torch import nn
from models_neural.src.layers_sentence_embedding import PretrainedModelLoader, SentenceEmbeddingsLayer
from models_neural.src.utils_general import get_config
from models_neural.src.layers_head import HeadLayerBinaryFF, HeadLayerBinaryLSTM, HeadLayerBinaryTransformer
from models_neural.src.utils_lightning import LightningOptimizer
from models_neural.quote_attribution.utils_lightning import LightningClassificationSteps, LightningQASteps
from torch.nn import CrossEntropyLoss


class SourceSentenceEmbeddingLayer(SentenceEmbeddingsLayer):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.person_embedding = nn.Embedding(2, self.config.embedding_dim)
        self.target_sentence_embedding = nn.Embedding(2, self.config.embedding_dim)

    def forward(
            self,
            input_ids,
            target_sentence_ids,
            target_person_ids,
            attention_mask,
            sent_lens=None,
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
        word_embs = self._get_word_embeddings(input_ids=input_ids, attention_mask=attention_mask)
        source_type_embs = self.person_embedding(target_person_ids)
        sentence_type_embs = self.target_sentence_embedding(target_sentence_ids)
        word_embs = word_embs + source_type_embs + sentence_type_embs
        return self.get_sentence_embed_helper(word_embs, attention_mask)


class BinaryClassificationHead(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config
        # for FFNN
        self.embedding_to_hidden = nn.Linear(self.config.embedding_dim, self.config.hidden_dim, bias=False)

        self.pred = nn.Linear(self.config.hidden_dim, self.config.num_output_tags)
        self.drop = nn.Dropout(self.config.dropout)
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding_to_hidden.state_dict()['weight'])
        nn.init.xavier_uniform_(self.pred.state_dict()['weight'])
        self.pred.bias.data.fill_(0)

    def get_contextualized_embeddings(self, cls_embeddings, *args, **kwargs):
        """Determines whether we contextualize the sentence-level embeddings with an LSTM or Transformer layer, or
        whether we just pass through a FFNN."""
        hidden_output = self.embedding_to_hidden(cls_embeddings)
        return hidden_output

    def calculate_loss(self, preds, labels):
        if labels.shape != preds.shape:
            labels = labels.reshape_as(preds)
        loss = self.criterion(preds, labels)
        return loss

    def classification(self, hidden_embs, labels=None):
        prediction = self.pred(self.drop(torch.tanh(hidden_embs)))  # pred = ( batch_size x num_labels)
        if labels is None:
            return None, prediction

        loss = self.calculate_loss(prediction, labels)
        loss = torch.mean(loss)
        return loss, prediction, labels

    def forward(self, sent_embs, labels):
        sent_embs = self.get_contextualized_embeddings(sent_embs)
        return self.classification(sent_embs, labels)


class SourceClassifier(LightningOptimizer, LightningClassificationSteps):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = get_config(kwargs=kwargs)
        self.transformer = SourceSentenceEmbeddingLayer(*args, **kwargs)
        self.head = self.get_head_layer() # (*args, **kwargs)

    def get_head_layer(self):
        # if self.config.num_contextual_layers == 0:
        #     return HeadLayerBinaryFF
        # if self.config.context_layer == 'lstm':
        #     return HeadLayerBinaryLSTM
        # elif self.config.context_layer == 'gpt2-sentence':
        #     return HeadLayerBinaryTransformer
        return BinaryClassificationHead(self.config)

    def forward(self, input_ids, target_sentence_ids, target_person_ids,
                labels=None, attention_mask=None, input_lens=None, *args, **kwargs):
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
        sent_embeddings = self.transformer(
            input_ids, target_sentence_ids, target_person_ids, attention_mask, sent_lens=input_lens)
        return self.head(sent_embeddings, labels)


class SanityCheckClassifier(LightningOptimizer, LightningClassificationSteps):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = get_config(kwargs=kwargs)
        self.transformer = SentenceEmbeddingsLayer(*args, **kwargs)
        self.head = BinaryClassificationHead(self.config)

    def forward(self, input_ids, labels=None, attention_mask=None, input_lens=None, *args, **kwargs):
        sent_embeddings = self.transformer.get_sentence_embedding(input_ids, attention_mask)
        return self.head(sent_embeddings, labels)


class SourceClassifierWithSourceSentVecs(SourceClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = self.get_head_layer()(hidden_dim=self.config.hidden_dim * 3, *args, **kwargs)

    def get_tokens_from_input_tensors(self, token_tensor, selector_tensor):
        idx = (selector_tensor == 1).nonzero()
        return token_tensor[:, idx[:, 1]]
        # return self.dim2_index_select(token_tensor, idx)

    # def dim2_index_select(self, vec, idx):
    #     idx_0, idx_1 = idx.T
    #     return vec[idx_0, idx_1]

    def _get_sentence_embeddings(self, tokens, attention_mask):
        word_embs = self.transformer._get_word_embeddings(tokens, attention_mask=attention_mask)
        return self.transformer.get_sentence_embed_helper(word_embs, attention_mask)

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
        # todo: this method will fail if batch_size > 1.
        orig_sent_embs = self.transformer(
            input_ids, target_sentence_ids, target_person_ids, attention_mask, sent_lens=input_lens)
        source_toks = self.get_tokens_from_input_tensors(input_ids, target_person_ids)
        if source_toks.shape[1] == 0:
            source_embs = torch.zeros_like(orig_sent_embs, device=orig_sent_embs.device)
        else:
            source_embs = self._get_sentence_embeddings(source_toks, attention_mask=None)
        sent_toks = self.get_tokens_from_input_tensors(input_ids, target_sentence_ids)
        sent_embs = self._get_sentence_embeddings(sent_toks, attention_mask=None)
        all_embs = torch.hstack((orig_sent_embs, source_embs, sent_embs))
        return self.head.classification(all_embs, labels)


class SourceQA(LightningOptimizer, LightningQASteps):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = get_config(kwargs=kwargs)
        self.num_labels = 2
        pretrained_model_loader = PretrainedModelLoader(*args, **kwargs)
        self.encoder_model = pretrained_model_loader.encoder_model
        self.qa_outputs = nn.Linear(self.config.embedding_dim, self.num_labels)
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