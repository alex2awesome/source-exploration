import torch
from torch import nn
# import sys, os
# here = os.path.dirname(__file__)
# sys.path.insert(0, here)

from .layers_attention import DocLevelSelfAttention
from .utils_general import get_config

class DocEmbMixin(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = get_config(config=config, kwargs=kwargs)
        # we find DocLevelSelfAttention performs the same/better than DocLevelAttention
        self.doc_attention = DocLevelSelfAttention(config=self.config, input_embedding_size=self.get_final_hidden_layer_size())

    def get_doc_embedding(self, cls):
        if self.config.use_doc_emb:
            return self.doc_attention(cls)
        else:
            return None


class PosEmbMixin(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = get_config(config=config, kwargs=kwargs)
        hidden_dim = self.config.hidden_dim * 2 if self.config.bidirectional else self.config.hidden_dim

        if not self.config.sinusoidal_embeddings:
            self.max_position_embs = nn.Parameter(torch.tensor(self.config.max_position_embeddings), requires_grad=False)
            self.default_max_pos = nn.Parameter(torch.tensor(self.max_position_embs - 1), requires_grad=False)
            self.position_embeddings = nn.Embedding(self.config.max_position_embeddings, hidden_dim)
        else:
            from fairseq.modules import SinusoidalPositionalEmbedding
            self.position_embeddings = SinusoidalPositionalEmbedding(hidden_dim, self.config.pad_id, self.config.max_position_embeddings)

    def get_position_embeddings(self, hidden_embs):
        if not self.config.use_positional:
            return None

        # get position embeddings
        if not self.config.sinusoidal_embeddings:
            position_ids = torch.arange(len(hidden_embs), dtype=torch.long, device=hidden_embs.device)
            position_ids = position_ids.where(position_ids < self.max_position_embs, self.default_max_pos) ## assign long sequences the same embedding
            position_embeddings = self.position_embeddings(position_ids)
        else:
            position_embeddings = self.position_embeddings(hidden_embs[:, [0]]).squeeze()

        return position_embeddings


class HeadlineMixin(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = get_config(config=config, kwargs=kwargs)

    def get_headline_embedding(self, cls):
        '''Given a sequence of [CLS] tokens, separate a headline embedding matrix and the rest of sentences.'''
        if self.config.use_headlines:
            headline_embedding = cls[0]
            cls = cls[1:]
            headline_embedding = headline_embedding.unsqueeze(0).expand(cls.size())
            return headline_embedding, cls
        else:
            return None, cls


class EmbeddingHandlerMixin(HeadlineMixin, PosEmbMixin, DocEmbMixin):
    def __init__(self, config=None, *args, **kwargs):
        self.config = get_config(config=config, kwargs=kwargs)
        super().__init__(config, *args, **kwargs)
        self.num_addt_vectors = int(((self.config.use_headline_embs or 0) and (self.config.concat_headline or 0))) + \
                                int(self.config.use_positional or 0) + \
                                int(self.config.use_doc_emb or 0) + \
                                int(self.config.doc_embed_arithmetic or 0) # We add 2 vectors. However, use_doc_embed provides 1.
        #
        self.concatenated_hidden_dim = self.get_total_hidden_dim()
        #
        if self.num_addt_vectors > 0:
            self.pre_pred = nn.Linear(self.concatenated_hidden_dim, self.config.hidden_dim)  # orig_code: config.embedding_dim
            self._init_pre_prediction_weights()

    def get_total_hidden_dim(self):
        hidden_dim = self.config.hidden_dim
        hidden_dim = hidden_dim * (1 + int(self.config.bidirectional))
        return hidden_dim * (1 + self.num_addt_vectors)

    def concat_vectors(self, sentence_embs, position_embs=None, headline_embs=None, doc_embs=None):
        to_concat = [sentence_embs]
        if self.config.use_headlines and self.config.concat_headline and headline_embs is not None:
            to_concat += [headline_embs]
        if self.config.use_positional:
            to_concat += [position_embs]
        if self.config.use_doc_emb:
            if self.config.doc_embed_arithmetic:
                to_concat += [doc_embs * sentence_embs, doc_embs - sentence_embs]
            else:
                to_concat += [doc_embs]
        return torch.cat(to_concat, 1)

    def combine_vectors(self, sentence_embs, position_embs, headline_embs, doc_embs):
        """Concantenate all vectors and"""
        concatted_embs = self.concat_vectors(sentence_embs, position_embs, headline_embs, doc_embs)
        if self.num_addt_vectors > 0:
            # pre_pred = (batch_size x hidden_dim * 2)
            concatted_embs = self.pre_pred(self.drop(torch.tanh(concatted_embs)))
        return concatted_embs

    def _init_pre_prediction_weights(self):
        if self.num_addt_vectors > 0:
            nn.init.xavier_uniform_(self.pre_pred.state_dict()['weight'])
            self.pre_pred.bias.data.fill_(0)

    def transform_sentence_embeddings(self, sent_embs):
        """Main method for this class."""
        headline_embs, sent_embs = self.get_headline_embedding(sent_embs)
        position_embs = self.get_position_embeddings(sent_embs)
        doc_embs = self.get_doc_embedding(sent_embs)
        augmented_embs = self.combine_vectors(sent_embs, position_embs, headline_embs, doc_embs)
        return augmented_embs


