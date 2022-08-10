import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)
class RelativePositionEmbedding(nn.Module):
    # implements https://aclanthology.org/N18-2074.pdf
    # Shaw et al 2018, Self-Attention with Relative Position Representations

    def __init__(self, emb_dim: int, n_rel_pos: int, max_len=512):
        super(RelativePositionEmbedding, self).__init__()
        self.emb_dim = emb_dim
        self.n_rel_pos = n_rel_pos
        self.emb = nn.Embedding(num_embeddings=2 * self.n_rel_pos + 1, embedding_dim=self.emb_dim)
        self._rel_pos_chart = self.make_relative_positions(n=max_len, k=n_rel_pos, positive_index=True)

    @classmethod
    def make_relative_positions(cls, n: int, k: int, positive_index=True):
        """
        :param n: sequence length
        :param k: relative window size; k as in [-k, ... -2, -1, 0, 1, 2, ... k]
        :param positive_index: novert neagtive index to positive index.
          i.e. [-k, ... -2, -1, 0, 1, 2, ... k] --> [0, 1, 2, .. k, k+1, k+2, .. 2*k
        :return:  a matrix of [n x n] with relative positions
        """
        assert n > 0
        k = k or n
        seq = torch.arange(n, device=device)  # [n]
        matrix = seq.repeat(n, 1)  # [n x n]
        matrix = matrix - seq.view(-1, 1)
        matrix = matrix.masked_fill(matrix > k, k)
        matrix = matrix.masked_fill(matrix < -k, -k)
        if positive_index:  # convert negatives to positive index;
            matrix += k
        return matrix

    def forward(self, query, key):
        assert query.shape[2] == key.shape[2], 'This feature works only for self-attention'
        n = query.shape[2]

        # efficient implementation
        # query, key:  [BatchSize x Heads x SeqLen x d ]  ; d=model_dim/heads
        # emb.weieght: [2*k+1 x d]
        dots = torch.matmul(query, self.emb.weight.transpose(-1, -2)) # [b x h x n x 2*k+1]
        if n > len(self._rel_pos_chart):
            self._rel_pos_chart = self.make_relative_positions(n=n, k=self.n_rel_pos, positive_index=True)
        rel_idx = self._rel_pos_chart[:n, :n] # n x n
        rel_idx = rel_idx.view(1, 1, *rel_idx.shape)        # [1, 1, n, n] ; add batch and heads dim
        scores = dots.gather(-1, rel_idx)  # pick scores along the last dim
        return scores


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None, query_key_emb: 'RelativePositionEmbedding' = None):
    """
    Compute 'Scaled Dot Product Attention'
    :param query:
    :param key:
    :param value:
    :param mask:
    :param dropout:
    :param query_key_emb:
    :return:
    """

    d_k = query.size(-1)
    # Beware: this is a batch multiplier!
    # See https://pytorch.org/docs/stable/torch.html?highlight=matmul#torch.matmul
    scores = torch.matmul(query, key.transpose(-2, -1))
    if query_key_emb is not None:
        rel_scores = query_key_emb(query=query, key=key)
        scores = scores + rel_scores
    scores = scores / math.sqrt(d_k)
    # scores: [BatchSize x Heads x Time=SeqLen x SeqLen ]
    if mask is not None:
        # How masking works:
        # src_mask is [BatchSize x 1=Heads x 1=Time x SeqLen ]  --> used in enc self_attn
        # tgt_mask is [BatchSize x 1=Heads x SeqLen=Time x SeqLen ]
        #               --> used in dec self_attn and dec_to_enc_attn
        # 1=Heads gets broad casted for all the heads
        # 1=Time is not broad casting, since it is used with encoder, we can encode the
        #    whole encoder seqs at once (unlike decoder, which goes at one time step at a time)
        # SeqLen=Time is a magic for the Decoding sequences to only rely on the previous time steps
        #
        # Now, if you got this, take a moment to thank http://nlp.seas.harvard.edu/rush.html
        # for devising this concise code. I needed a lot of time to understand how this code works!
        #
        # scores = scores.masked_fill(mask == 0, -1e9)
        # low_val = -2 ** 14 if dtorch.fp16 else -1e9  # -2**15 causes nan on float16
        low_val = -1e9  # now we use bfloat16, which is awesome
        scores = scores.masked_fill(mask == 0, low_val)
    p_attn = F.softmax(scores, dim=-1)  # [BatchSize x Heads x Time=SeqLen x SeqLen ]
    if dropout is not None:
        p_attn = dropout(p_attn)

    # Beware: this is a batch multiplier!

    ctx_vals = torch.matmul(p_attn, value)
    # ctx_vals = ctx_vals.to(value.dtype)  # p_attn maybe float, value maybe half
    return ctx_vals, p_attn


class TGMultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, bias=True, cache_attn=False, n_rel_pos=0):
        "Take in model size and number of heads."
        super().__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model, bias=bias), 4)
        self.cache_attn = cache_attn
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.n_rel_pos = n_rel_pos
        self.rel_pos_emb = None
        if self.n_rel_pos > 0:
            self.rel_pos_emb = RelativePositionEmbedding(emb_dim=self.d_k, n_rel_pos=n_rel_pos)


    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # [BatchSize x 1 x Time x SeqLen]  1=Broadcast for all heads
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # Q,K,V  --> input, linear: [BatchSize x SeqLen x ModelDim]
        #        --> view: [BatchSize x SeqLen x Heads x ModelDim/Heads ]
        #        --> transpose: [BatchSize x Heads x SeqLen x ModelDim/Heads ]
        x, attn = attention(query, key, value, mask=mask, dropout=self.dropout, query_key_emb=self.rel_pos_emb)
        if self.cache_attn:  # dont cache this at training time
            self.attn = attn.detach()
        # attn: [BatchSize x Heads x SeqLen_query x SeqLen_value ]
        # x : [BatchSize x Heads x SeqLen x ModelDim/Heads ]

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        # x : transpose [BatchSize x SeqLen x Heads x ModelDim/Heads ]
        # x : view [BatchSize x SeqLen x ModelDim ]
        return self.linears[-1](x)


class SentenceCompressor(nn.Module):
    """
    Compresses token representation into a single vector
    """

    def __init__(self, hidden_dim, embed_dim, num_heads):
        super().__init__()
        self.cls_repr = nn.Parameter(torch.zeros(hidden_dim))
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.attn = TGMultiHeadedAttention(h=num_heads, d_model=self.hidden_dim)
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
