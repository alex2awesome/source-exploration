from transformers import BertModel
from transformers import AutoConfig
import torch
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence


test_torch = False
test_bert_multiheaded = False
test_bert_attention = False
test_bert_sentence_compression = False

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

input_ids = list(map(lambda x:
                     tokenizer.encode(x, return_tensors='pt').squeeze(),
                     [ 'Hello my name is Alex', 'Alex is my name' ])
)
seq_lens = list(map(len, input_ids))
max_seq_len = max(seq_lens)
attn_mask = list(map(lambda x: [1] * x + [0] * (max_seq_len - x), seq_lens))
attn_mask = torch.tensor(attn_mask)
input_ids = pad_sequence(input_ids, batch_first=True)


def get_extended_attention_mask(attention_mask, dtype=torch.float32):
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (:obj:`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        device: (:obj:`torch.device`):
            The device of the input to the model.

    Returns:
        :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask = attention_mask[:, None, None, :]
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


## torch multiheaded attention
if test_torch:
    import torch
    max_seq_len = 10        # S
    emb_dim = 512           # E
    hidden_dim = 768        # H
    output_seq_len = 1      # L
    num_heads = 12          # M

    query_layer = torch.nn.Linear(emb_dim, hidden_dim)
    compressor_layer = torch.nn.Linear(max_seq_len, output_seq_len)
    key_layer = torch.nn.Linear(emb_dim, hidden_dim)
    value_layer = torch.nn.Linear(emb_dim, hidden_dim)
    attention = torch.nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    batch_size = 15
    x_seq_len = 5
    x = torch.rand((batch_size, x_seq_len, emb_dim))

    z = torch.zeros(batch_size, max_seq_len - x_seq_len, emb_dim)
    x_inp = torch.hstack([x, z])

    Q = query_layer(x_inp)
    V = value_layer(x_inp)
    K = key_layer(x_inp)

    Q_s = compressor_layer(Q.permute(0, 2, 1)).permute(0, 2, 1)
    o, _ = attention.forward(Q_s, K, V, need_weights=False)

    o.squeeze()


if test_bert_multiheaded:
    ## BERT Multiheaded attention

    S = 5
    N = 10
    E = 768


    model = BertModel.from_pretrained('bert-base-uncased')

    hidden = torch.rand((N, S, E))
    model.encoder.layer[0].attention.self.forward(hidden)[0].shape


if test_bert_attention:
    model = BertModel.from_pretrained('bert-base-uncased')
    model.forward(input_ids=input_ids, attention_mask=attn_mask)


if test_bert_sentence_compression:
    model = BertModel.from_pretrained('bert-base-uncased')

