import torch
import torch.nn as nn
from torch import nn as nn


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


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, max_seq_len, emb_dim=512, hidden_size=768, num_heads=12):
        super().__init__()
        self.max_seq_len = max_seq_len    # S
        self.emb_dim = emb_dim            # E
        self.hidden_dim = hidden_size     # H
        self.output_seq_len = 1           # L
        self.num_heads = num_heads        # M

        self.query_layer = torch.nn.Linear(self.emb_dim, self.hidden_dim)
        self.compressor_layer = torch.nn.Linear(self.max_seq_len, self.output_seq_len)
        self.key_layer = torch.nn.Linear(self.emb_dim, self.hidden_dim)
        self.value_layer = torch.nn.Linear(self.emb_dim, self.hidden_dim)
        self.attention = torch.nn.MultiheadAttention(self.hidden_dim, self.num_heads, batch_first=True)

    def forward(self, x, attention_mask):
        batch_size, x_seq_len, emb_dim = x.shape                # shape: N x S x E
        if x_seq_len < self.max_seq_len:
            z = torch.zeros(batch_size, self.max_seq_len - x_seq_len, emb_dim)
            x = torch.hstack([x, z])
            attention_mask = torch.hstack([attention_mask, x])
            batch_size, x_seq_len, emb_dim = x.shape  # shape: N x S x E

        assert x_seq_len == self.max_seq_len

        Q = self.query_layer(x)
        K = self.key_layer(x)
        V = self.value_layer(x)

        attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, self.hidden_dim)
        masked_Q = Q + torch.where(attention_mask != 0, torch.zeros_like(Q), -Q)
        masked_K = K + torch.where(attention_mask != 0, torch.zeros_like(K), -K)
        masked_V = V + torch.where(attention_mask != 0, torch.zeros_like(V), -V)

        Q_s = self.compressor_layer(masked_Q.permute(0, 2, 1)).permute(0, 2, 1)
        o, _ = self.attention.forward(Q_s, masked_K, masked_V, attn_mask=attention_mask, need_weights=False)                        # N x L x E
        return o.squeeze()


class TGMultiHeadedSelfAttention(nn.Module):
    """
    Compresses token representation into a single vector
    """

    def __init__(self, hidden_dim, embed_dim, num_heads):
        super().__init__()
        self.cls_repr = nn.Parameter(torch.zeros(hidden_dim))
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.attn = torch.nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.key_layer = torch.nn.Linear(self.embed_dim, self.hidden_dim)
        self.value_layer = torch.nn.Linear(self.embed_dim, self.hidden_dim)


    def forward(self, x, attention_mask):
        B, T, D = x.size()  # [Batch, Time, Dim]
        assert D == self.embed_dim
        query = self.cls_repr.view(1, 1, self.hidden_dim).repeat(B, 1, 1)

        key = self.key_layer(x)
        value = self.value_layer(x)

        # Args: Query, Key, Value, Mask
        cls_repr = self.attn(query, key, value, attention_mask)
        cls_repr = cls_repr.view(B, self.hidden_dim)  # [B, D]
        return cls_repr


from torch.nn import Parameter
from torch.nn import init
from torch.autograd import Variable

class BatchSelfAttention(nn.Module):
    def __init__(self, attention_size, batch_first=False, non_linearity="tanh"):
        super().__init__()

        self.batch_first = batch_first
        self.attention_weights = Parameter(torch.FloatTensor(attention_size))
        self.softmax = nn.Softmax(dim=-1)

        if non_linearity == "relu":
            self.non_linearity = nn.ReLU()
        else:
            self.non_linearity = nn.Tanh()

        init.uniform(self.attention_weights.data, -0.005, 0.005)

    def get_mask(self, attentions, lengths):
        """
        Construct mask for padded itemsteps, based on lengths
        """
        max_len = max(lengths.data)
        mask = Variable(torch.ones(attentions.size())).detach()

        if attentions.data.is_cuda:
            mask = mask.cuda()

        for i, l in enumerate(lengths.data):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0
        return mask

    def forward(self, inputs, lengths):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.non_linearity(inputs.matmul(self.attention_weights))
        scores = self.softmax(scores)

        ##################################################################
        # Step 2 - Masking
        ##################################################################

        # construct a mask, based on the sentence lengths
        mask = self.get_mask(scores, lengths)

        # apply the mask - zero out masked timesteps
        masked_scores = scores * mask

        # re-normalize the masked scores
        _sums = masked_scores.sum(-1, keepdim=True)  # sums per row
        scores = masked_scores.div(_sums)  # divide by row sum

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states
        representations = weighted.sum(1).squeeze()

        return representations, scores