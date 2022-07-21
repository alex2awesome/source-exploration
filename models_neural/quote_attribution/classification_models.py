import torch
from torch import nn
from models_neural.src.layers_sentence_embedding import SentenceEmbeddingsLayer
from models_neural.src.utils_general import get_config
from models_neural.src.layers_head import HeadLayerBinaryFF, HeadLayerBinaryLSTM, HeadLayerBinaryTransformer
from models_neural.src.utils_lightning import LightningMixin


class SourceSentenceEmbeddingLayer(SentenceEmbeddingsLayer):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.person_embedding = nn.Embedding(2, self.config.hidden_dim)
        self.target_sentence_embedding = nn.Embedding(2, self.config.hidden_dim)
        if self.config.sentence_embedding_method == 'multiheaded-attention':
            from models_neural.src.layers_attention import MultiHeadedSelfAttention
            self.attention = MultiHeadedSelfAttention(self.config.max_length)

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