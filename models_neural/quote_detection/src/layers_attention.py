import torch
import torch.nn as nn

class AdditiveSelfAttention(nn.Module):
    def __init__(self, input_dim, dropout):
        super().__init__()
        self.ws1 = nn.Linear(input_dim, input_dim)
        self.ws2 = nn.Linear(input_dim, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.ws1.state_dict()['weight'])
        self.ws1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.ws2.state_dict()['weight'])

    def forward(self, sent_embeds, context_mask=None):
        ## get sentence encoding using additive attention (appears to be based on Bahdanau 2015) where:
        ##     score(s_t, h_i) = v_a^T tanh(W_a * [s_t; h_i]),
        ## here, s_t, h_i = word embeddings
        ## align(emb) = softmax(score(Bi-LSTM(word_emb)))
        # word_embs: shape = (num sentences in curr batch * max_len * embedding_dim)     # for word-attention:
        #     where embedding_dim = hidden_dim * 2                                       # -------------------------------------
        # sent_embs: shape = if one doc:   (num sentences in curr batch * embedding_dim)
        #         #          if many docs: (num docs x num sentences in batch x max word len x hidden_dim)
        self_attention = torch.tanh(self.ws1(self.drop(sent_embeds)))         # self attention : if one doc: (num sentences in curr batch x max_len x hidden_dim
                                                                              #   if >1 doc: if many docs: (num docs x num sents x max word len x hidden_dim)
        self_attention = self.ws2(self.drop(self_attention)).squeeze(-1)      # self_attention : (num_sentences in curr batch x max_len)
        if context_mask is not None:
            context_mask = -10000 * (context_mask == 0).float()
            self_attention = self_attention + context_mask                    # self_attention : (num_sentences in curr batch x max_len)
        if len(self_attention.shape) == 1:
            self_attention = self_attention.unsqueeze(0)  # todo: does this cause problems?
        self_attention = self.softmax(self_attention).unsqueeze(1)            # self_attention : (num_sentences in curr batch x 1 x max_len)
        return self_attention


class WordLevelAttention(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        # input_dim is config.hidden_dim * 2 if we're using a bidirectional LSTM
        self.self_attention = AdditiveSelfAttention(input_dim=input_dim, dropout=config.dropout)
        self.inner_pred = nn.Linear(input_dim, config.embedding_dim)  # Prafulla 3
        self.drop = nn.Dropout(config.dropout)

        # init weights
        self._init_attention_weights()

    def _init_attention_weights(self):
        nn.init.xavier_uniform_(self.inner_pred.state_dict()['weight'])
        self.inner_pred.bias.data.fill_(0)

    def forward(self, word_embs, context_mask):
        self_attention = self.self_attention(word_embs, context_mask)      # sent_encoding: (# sents in batch x (hidden_dim * 2))
        sent_encoding = torch.matmul(self_attention, word_embs)
        if len(sent_encoding.shape) == 4:
            sent_encoding = sent_encoding.squeeze(dim=2)
        if len(sent_encoding.shape) == 3:
            sent_encoding = sent_encoding.squeeze(dim=1)  # sent_encoding: (# sents in batch x (hidden_dim * 2))
        return self.drop(sent_encoding)


class DocLevelAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.self_attention = AdditiveSelfAttention(input_dim=kwargs.get('input_embedding_size'), dropout=kwargs.get('config').dropout)

    def forward(self, inner_pred, sent_encoding):
        ## get document embedding?
        ## inner_pred: shape = 1 x batch_size x (hidden_dim * 2)
        self_attention = self.self_attention(inner_pred)                                   # self_attention = 1 x batch_size
        doc_encoding = torch.matmul(self_attention.squeeze(), sent_encoding).unsqueeze(0)  # doc_encoding   = 1 x (hidden_dim * 2)

        ## reshape
        inner_pred = inner_pred.squeeze()                                                  # inner_pred = batch_size x (hidden_dim * 2)
        doc_encoding = doc_encoding.expand(inner_pred.size())         #  doc_encoding = batch_size x (hidden_dim * 2)
        return inner_pred, doc_encoding


class DocLevelSelfAttention(DocLevelAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expand = kwargs.get('expand', True)

    def forward(self, sent_encoding):
        ## get document embedding?
        ## sent_encoding: shape= batch_size x hidden_dim
        self_attention = self.self_attention(sent_encoding)                   # self_attention = batch_size x 1 x batch_size
        doc_encoding = torch.matmul(self_attention.squeeze(1), sent_encoding)  # doc_encoding  = 1 x hidden_dim
        if self.expand:
            doc_encoding = doc_encoding.expand(sent_encoding.size())              # doc_encoding = batch_size x hidden_dim
        return doc_encoding


# class DocLevelSelfAttention(nn.Module):
#     def __init__(self, config, expand=True):
#         super().__init__()
#         self.self_attention = AdditiveSelfAttention(input_dim=config.hidden_dim, dropout=config.dropout)
#         self.expand = expand
#
#     def forward(self, sent_encoding):
#         ## get document embedding?
#         ## inner_pred: shape = 1 x batch_size x (hidden_dim * 2)
#         ## sent_encoding: shape= batch_size x (hidden_dim * 2)
#         sent_encoding = sent_encoding.unsqueeze(0)
#         self_attention = self.self_attention(sent_encoding)                   # self_attention = 1 x batch_size
#         doc_encoding = torch.matmul(self_attention.squeeze(), sent_encoding)  # doc_encoding   = 1 x (hidden_dim * 2)
#
#         ## reshape
#         sent_encoding = sent_encoding.squeeze()                               # inner_pred = batch_size x (hidden_dim * 2)
#         if self.expand:
#             doc_encoding = doc_encoding.expand(sent_encoding.size())              #  doc_encoding = batch_size x (hidden_dim * 2)
#         return doc_encoding
#


class DocEmbeddingForDocLabelClass(nn.Module):
    """Generate a single embedding vector for an entire document."""
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.doc_attention = DocLevelSelfAttention(
            config=config,
            input_embedding_size=config.hidden_dim,
            expand=False
        )

    def forward(self, sentence_embeddings):
        return self.doc_attention(sentence_embeddings)


class LabelEmbeddingWithContext(nn.Module):
    """Used to combine multiple embeddings with a context window"""
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config
        self.window_back = self.config.label_context_back
        self.window_forward = self.config.label_context_forward
        self.label_attention = DocLevelSelfAttention(
            config=config,
            input_embedding_size=config.hidden_dim,
            expand=False
        )

    def forward(self, label_embeddings, label_idx):
        """`label_embeddings` is of shape (num_labels + 2)
            --> start and end idxs are appended.
        """
        label_embs = []
        if self.window_back != 0:
            if self.window_back != -1:
                reach_back = max(label_idx - self.window_back, 0)
            else:
                reach_back = None
            label_embs_back = label_embeddings[reach_back: label_idx]
            label_embs.append(label_embs_back)

        if self.window_forward != 0:
            if self.window_forward != -1:
                reach_forward = label_idx + self.window_forward + 1
            else:
                reach_forward = None
            label_embs_forward = label_embeddings[label_idx + 1: reach_forward]
            label_embs.append(label_embs_forward)

        label_embs = torch.vstack(label_embs)
        return self.label_attention(label_embs)
