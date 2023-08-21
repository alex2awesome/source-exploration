import torch.nn as nn
from typing import Optional, List, Dict, Union, Any

from torch.utils.data import Dataset
from transformers import AutoModel, AutoConfig, RobertaConfig, RobertaModel, PreTrainedModel, BertPreTrainedModel

import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torch.special import expit

######################################
# data components
def _get_attention_mask(x: List[torch.Tensor], max_length_seq: Optional[int]=10000) -> torch.Tensor:
    max_len = max(map(lambda y: y.shape.numel(), x))
    max_len = min(max_len, max_length_seq)
    attention_masks = []
    for x_i in x:
        input_len = x_i.shape.numel()
        if input_len < max_length_seq:
            mask = torch.cat((torch.ones(input_len), torch.zeros(max_len - input_len)))
        else:
            mask = torch.ones(max_length_seq)
        attention_masks.append(mask)
    return torch.stack(attention_masks)


class TokenizedDataset(Dataset):
    def __init__(self, doc_list, config, tokenizer):
        """
        Processes a dataset, `doc_list` for either training/evaluation or scoring.
        * doc_list: We expect a list of dictionaries.
        * source_encoding_method: whether to encode sources as:
            * full-names: their representation will be tokens.
            * ordinals: their representation will be a number from [1...num_sources] with 0 being "no source"
            * boolean: their representation will be [0, 1] for "no source" or "source"
            * none: no representations will be passed
        """
        self.tokenizer = tokenizer
        self.input_ids = []
        self.input_id_attention = []
        self.source_tokens = []  # designation for the source identity per sentence (ordinal, boolean or tokens)
        self.source_token_attention = []

        self.labels = []
        self.max_length = config.max_sequence_len
        self.source_encoding_method = config.source_encoding_method
        self.process_data(doc_list)

    def process_data(self, doc_list):
        for doc in doc_list:
            input_ids, input_att_mask, source_ids, source_att_mask, doc_label = self.process_one_doc(doc)
            if input_ids is not None:
                self.input_ids.append(input_ids)
                self.input_id_attention.append(input_att_mask)

                self.source_tokens.append(source_ids)
                self.source_token_attention.append(source_att_mask)

                self.labels.append(doc_label)

    def process_source(self, source_str, source_lookup=None):
        if self.source_encoding_method == 'full-name':
            return self.tokenizer.encode(source_str)
        elif self.source_encoding_method == 'ordinal':
            if source_str != 'None':
                return [source_lookup[source_str]]
            else:
                return [0]
        elif self.source_encoding_method == 'boolean':
            return [int(source_str != 'None')]
        elif self.source_encoding_method is None:
            return [0]

    def process_one_doc(self, doc):
        sent_tokens = []
        source_tokens = []
        doc_label = int(doc['label'])

        # encode sources as ordinals if `self.source_encoding_method == 'ordinal'`
        source_lookup = None
        if self.source_encoding_method == 'ordinal':
            source_lookup = {v: k + 1 for k, v in enumerate(set(doc['attribution']))}

        # encode and chunk
        for source, sentence in zip(doc['attribution'], doc['sent']):
            sent_tokens.append(self.tokenizer.encode(sentence))
            source_tokens.append(self.process_source(source, source_lookup))

        # tensorize and pad
        num_doc_toks = map(len, sent_tokens)
        if any(num_doc_toks) <= self.max_length:
            sent_tokens = list(map(torch.tensor, sent_tokens))
            sent_token_att_mask = _get_attention_mask(sent_tokens)
            sent_tokens_finalized = pad_sequence(sent_tokens, batch_first=True)

            source_tokens = list(map(torch.tensor, source_tokens))
            source_token_att_mask = _get_attention_mask(source_tokens)
            source_tokens_finalized = pad_sequence(source_tokens, batch_first=True)

            # we need have the labels like this for this weird `nested_concat` function
            # in the `evaluation_loop` of the `Trainer.py`.
            doc_label = torch.tensor([doc_label])
            return sent_tokens_finalized, sent_token_att_mask, source_tokens_finalized, source_token_att_mask, doc_label
        else:
            return None, None, None, None, None

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        output_dict = {
            'input_ids': self.input_ids[idx],
            'input_id_att_mask': self.input_id_attention[idx],
            'source_ids': self.source_tokens[idx],
            'source_id_att_mask': self.source_token_attention[idx],
            'labels': self.labels[idx]
        }
        return output_dict


def collate_fn(dataset):
    """
    Takes in an instance of Torch Dataset.
    Returns:
     * input_ids: List len N,  elements are len num toks in doc
     * label_ids: List size N, elements are len num sents in doc
    """
    # transpose dict
    batch_by_columns = {}
    for key in dataset[0].keys():
        batch_by_columns[key] = list(map(lambda d: d[key], dataset))
    return batch_by_columns


###############################
# model components
class TransformerContext(nn.Module):
    def __init__(self, config, num_sent_attn_heads=2, num_contextual_layers=2, max_num_sentences=100):
        super().__init__()
        # load transformer
        config.num_attention_heads = num_sent_attn_heads
        config.num_hidden_layers = num_contextual_layers
        config.max_position_embeddings = max_num_sentences + 20

        self.base_model = AutoModel.from_config(config)
        TransformerContext.base_model_prefix = self.base_model.base_model_prefix
        TransformerContext.config_class = self.base_model.config_class

    def forward(self, cls_embeddings):
        # inputs_embeds: input of shape: (batch_size, sequence_length, hidden_size)
        contextualized_embeds = self.base_model(inputs_embeds=cls_embeddings.unsqueeze(0))[0]
        return contextualized_embeds.squeeze()


class BiLSTMContext(nn.Module):
    def __init__(self, config, num_contextual_layers=2, bidirectional=True):
        super().__init__()
        self.bidirectional = True
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=num_contextual_layers,
            bidirectional=bidirectional
        )
        lstm_output_size = config.hidden_size * 2 if self.bidirectional else config.hidden_size
        self.resize = nn.Linear(lstm_output_size, config.hidden_size)
        # init params
        for name, param in self.lstm.state_dict().items():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, cls_embeddings):
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(cls_embeddings)
        resized = self.resize(lstm_out)
        return resized


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

    def forward(self, hidden_embeds, context_mask=None):
        ## get sentence encoding using additive attention (appears to be based on Bahdanau 2015) where:
        ##     score(s_t, h_i) = v_a^T tanh(W_a * [s_t; h_i]),
        ## here, s_t, h_i = word embeddings
        ## align(emb) = softmax(score(Bi-LSTM(word_emb)))
        # word_embs: shape = (num sentences in curr batch * max_len * embedding_dim)     # for word-attention:
        #     where embedding_dim = hidden_dim * 2                                       # -------------------------------------
        # sent_embs: shape = if one doc:   (num sentences in curr batch * embedding_dim)
        #         #          if many docs: (num docs x num sentences in batch x max word len x hidden_dim)
        self_attention = torch.tanh(self.ws1(self.drop(hidden_embeds)))         # self attention : if one doc: (num sentences in curr batch x max_len x hidden_dim
                                                                              #   if >1 doc: if many docs: (num docs x num sents x max word len x hidden_dim)
        self_attention = self.ws2(self.drop(self_attention)).squeeze(-1)      # self_attention : (num_sentences in curr batch x max_len)
        if context_mask is not None:
            context_mask = -10000 * (context_mask == 0).float()
            self_attention = self_attention + context_mask                    # self_attention : (num_sentences in curr batch x max_len)
        if len(self_attention.shape) == 1:
            self_attention = self_attention.unsqueeze(0)  # todo: does this cause problems?
        self_attention = self.softmax(self_attention).unsqueeze(1)            # self_attention : (num_sentences in curr batch x 1 x max_len)
        return self_attention


class AttentionCompression(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.self_attention = AdditiveSelfAttention(input_dim=hidden_size, dropout=dropout)

    def forward(self, hidden_embs, attention_mask=None):
        ## `'hidden_emds'`: shape = N x hidden_dim
        self_attention = self.self_attention(hidden_embs, attention_mask)  # self_attention = N x 1 x N
        ## batched matrix x batched matrix:
        output_encoding = torch.matmul(self_attention, hidden_embs).squeeze(1)
        return output_encoding


def freeze_hf_model(model, freeze_layers, freeze_embeddings=False):
    def freeze_all_params(subgraph):
        for p in subgraph.parameters():
            p.requires_grad = False

    if isinstance(model, RobertaModel):
        layers = model.encoder.layer
        embeddings = model.embeddings
    else:
        layers = model.transformer.h

    if freeze_layers is not None:
        for layer in freeze_layers:
            freeze_all_params(layers[layer])

    if freeze_embeddings:
        freeze_all_params(embeddings)


class DocClassificationModelSentenceLevel(BertPreTrainedModel):
    def __init__(self, config, hf_model=None):
        super().__init__(config)

        base_model = AutoModel.from_config(config) if hf_model is None else hf_model
        DocClassificationModelSentenceLevel.base_model_prefix = base_model.base_model_prefix
        DocClassificationModelSentenceLevel.config_class = base_model.config_class
        setattr(self, self.base_model_prefix, base_model)  # setattr(x, 'y', v) is equivalent to ``x.y = v''

        self.use_input_ids = config.use_input_ids
        self.use_source_ids = config.use_source_ids

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if config.source_encoding_method == 'full-name':
            self.source_embeddings = base_model.embeddings.word_embeddings
        else:
            self.source_embeddings = nn.Embedding(config.max_num_sources, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.num_labels = config.num_labels
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.loss_fct = CrossEntropyLoss() if self.num_labels > 1 else BCEWithLogitsLoss()
        self.word_pooling_method = config.word_pooling_method
        self.sent_pooling_method = config.sent_pooling_method
        self.word_attention = AttentionCompression(hidden_size=config.hidden_size, dropout=config.hidden_dropout_prob)
        self.sent_attention = AttentionCompression(hidden_size=config.hidden_size, dropout=config.hidden_dropout_prob)
        if config.context_layer == 'transformer':
            self.context_layer = TransformerContext(config, num_contextual_layers=config.num_contextual_layers)
        else:
            self.context_layer = None
        self.post_init()

    def post_init(self):
        # during prediction, we don't have to pass this in
        if hasattr(self.config, 'freeze_layers'):
            base_model = getattr(self.base_model_prefix)
            freeze_hf_model(base_model, freeze_layers=self.config.freeze_layers,
                            freeze_embeddings=self.config.freeze_embeddings)

    def pool(self, hidden, attention_mask=None, ):
        # trick: attention mask is not none if we're pooling words...
        if attention_mask is not None:
            if self.word_pooling_method == 'average':
                return (hidden.T * attention_mask.T).T.mean(axis=1)
            elif self.word_pooling_method == 'cls':
                return hidden[:, 0, :]
            else:
                return self.word_attention(hidden, attention_mask)

        # trick: attention mask is none if we're pooling sentences...
        if attention_mask is None:
            if self.sent_pooling_method == 'average':
                return hidden.mean(axis=0).unsqueeze(dim=0)
            elif self.sent_pooling_method == 'cls':
                return hidden[:, 0, :]
            else:
                return self.sent_attention(hidden)

    def get_embeddings(self, input_ids, input_id_att_mask, source_ids, source_id_att_mask):
        # Get embeddings for everything before feeding into the Bert model
        all_embeds = []
        all_attention_mask = []
        if self.use_input_ids:
            inputs_embeds = self.base_model.embeddings(input_ids)
            all_embeds.append(inputs_embeds)
            all_attention_mask.append(input_id_att_mask)

        # todo: especially with layer-wise freezing, maybe these embeddings need to get moved out of the main
        #  transformer?
        if self.use_source_ids:
            source_embeds = self.source_embeddings(source_ids)
            source_token_type_ids = torch.ones(source_ids.shape, dtype=torch.long, device=source_ids.device)
            source_token_type_embeds = self.base_model.embeddings.token_type_embeddings(source_token_type_ids)
            source_embeds += source_token_type_embeds
            all_embeds.append(source_embeds)
            all_attention_mask.append(source_id_att_mask)

        embeddings = torch.hstack(all_embeds)
        attention_mask = torch.hstack(all_attention_mask)
        return embeddings, attention_mask

    def process_one_doc(
            self,
            input_ids: Optional[torch.Tensor] = None,
            input_id_att_mask: Optional[torch.Tensor] = None,
            source_ids: Optional[torch.Tensor] = None,
            source_id_att_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None
    ):

        embeddings, attention_mask = self.get_embeddings(input_ids, input_id_att_mask, source_ids, source_id_att_mask)
        outputs = self.base_model(inputs_embeds=embeddings, attention_mask=attention_mask)

        # pool word embeddings
        hidden = outputs[0]
        pooled_output = self.pool(hidden=hidden, attention_mask=attention_mask)
        pooled_output = self.dropout(pooled_output)
        if self.context_layer is not None:
            pooled_output = self.context_layer(pooled_output)
        pooled_output = self.pool(pooled_output)
        logits = self.classifier(pooled_output)

        # calculate loss
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)
        return loss, logits

    def forward(self, *args, **kwargs):
        item_kwargs = transpose_coldict_to_rowdict(kwargs)
        outputs = list(map(lambda x: self.process_one_doc(**x), item_kwargs))
        losses, logits = list(zip(*outputs))
        loss = None if losses[0] == None else sum(losses)
        logits = torch.vstack(logits)
        return (loss, logits) if loss is not None else logits

    def estimate_tokens(self, input_dict: Dict[str, Union[torch.Tensor, Any]]) -> int:
        """
        Helper function to estimate the total number of tokens from the model inputs. Needed for `Trainer.py` class.
        (Custom implementation is necessary because our dataset is in a list format.)
        """
        token_inputs = [tensor for key, tensor in input_dict.items() if "input" in key]
        if token_inputs:
            # torch.vstack is the only change
            return sum([torch.vstack(token_input).numel() for token_input in token_inputs])
        else:
            return 0

def transpose_coldict_to_rowdict(coldict):
    keys = list(coldict.keys())
    output = []
    for i in range(len(coldict[keys[0]])):
        output.append({k: coldict[k][i] for k in keys})
    return output

