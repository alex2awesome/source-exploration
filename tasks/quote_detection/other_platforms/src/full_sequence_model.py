import torch.nn as nn
from typing import Optional, List

from torch.utils.data import Dataset
from transformers import AutoModel, AutoConfig, RobertaConfig, RobertaModel
import functools

import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
import re
from unidecode import unidecode
import pandas as pd
import numpy as np


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


def fix_quote_type(sent):
    CLEANR = re.compile('<.*?>')

    def cleanhtml(raw_html):
        cleantext = re.sub(CLEANR, '', raw_html)
        return cleantext

    def normalize(text):
        text = '' if pd.isnull(text) else text
        text = re.sub('\s+', ' ', text)
        return cleanhtml(unidecode(text).strip())

    quote_type_mapper = {
        '': 'NO QUOTE',
        'PUBLIC SPEECH, NOT TO JOURNO': 'PUBLIC SPEECH',
        'COMMUNICATION, NOT TO JOURNO': 'COMMUNICATION',
        'LAWSUIT': 'COURT PROCEEDING',
        'TWEET': 'SOCIAL MEDIA POST',
        'PROPOSAL': 'PROPOSAL/ORDER/LAW',
        'Other: LAWSUIT': 'COURT PROCEEDING',
        'Other: Evaluation': 'QUOTE',
        'Other: DIRECT OBSERVATION': 'DIRECT OBSERVATION',
        'Other: Campaign filing': 'PUBLISHED WORK',
        'Other: VOTE/POLL': 'VOTE/POLL',
        'Other: PROPOSAL': 'PROPOSAL/ORDER/LAW',
        'Other: Campaign Filing': 'PUBLISHED WORK',
        'Other: Data analysis': 'DIRECT OBSERVATION',
        'Other: Analysis': 'DIRECT OBSERVATION',
        'Other: LAW': 'PROPOSAL/ORDER/LAW',
        'Other: Investigation': 'DIRECT OBSERVATION',
        'Other: Database': 'PUBLISHED WORK',
        'Other: Data Analysis': 'DIRECT OBSERVATION',
        'DOCUMENT': 'PUBLISHED WORK',
    }

    q = sent.get('quote_type', '')
    q = quote_type_mapper.get(q, q)
    if (q == 'QUOTE'):
        if ('"' in normalize(sent['sent'])):
            return 'INDIRECT QUOTE'
        else:
            return 'DIRECT QUOTE'
    return q


class TokenizedDataset(Dataset):
    def __init__(self, doc_list, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.labels = []
        self.categories = []
        self.sent_lens = []

        for doc in doc_list:
            doc_tokens = []
            doc_labels = []
            for sentence in doc:
                doc_tokens.append(tokenizer.encode(sentence['sent']))
                doc_labels.append(int(sentence['label']))
            doc_sent_lens = list(map(len, doc_tokens))
            if sum(doc_sent_lens) <= max_length:
                doc_tokens = functools.reduce(lambda a, b: a + b, doc_tokens)
                self.input_ids.append(torch.tensor(doc_tokens))
                self.labels.append(torch.tensor(doc_labels))
                self.sent_lens.append(doc_sent_lens)
                self.categories.append(list(map(lambda x: fix_quote_type(x), doc)))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'labels': self.labels[idx],
            'sentence_lens': self.sent_lens[idx],
            'categories': self.categories[idx]
        }


def collate_fn(dataset):
    """
    Takes in an instance of Torch Dataset.
    Returns:
     * input_ids: tensor, size: N x num toks in doc
     * label_ids: tensor, size: N x num sents in doc
     * sentence_lens: List[List[int]], size: N x num sents in doc
    """
    # transpose dict
    batch_by_columns = {}
    for key in dataset[0].keys():
        batch_by_columns[key] = list(map(lambda d: d[key], dataset))

    output = {}
    output['input_ids'] = pad_sequence(batch_by_columns["input_ids"], batch_first=True)
    output['labels'] = pad_sequence(batch_by_columns["labels"], batch_first=True).to(float)
    output['sentence_lens'] = batch_by_columns['sentence_lens']
    return output


###############################
# model components
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

class TransformerContextMixin(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # load transformer
        transformer_config = RobertaConfig.from_pretrained(self.config.pretrained_model_path)
        transformer_config.num_attention_heads = self.config.num_sent_attn_heads
        transformer_config.num_hidden_layers = self.config.num_contextual_layers
        transformer_config.hidden_size = self.config.hidden_dim
        transformer_config.max_position_embeddings = self.config.max_num_sentences + 20
        self.sentence_transformer = RobertaModel(config=transformer_config)

    def forward(self, cls_embeddings):
        # inputs_embeds: input of shape: (batch_size, sequence_length, hidden_size)
        contextualized_embeds, _ = self.sentence_transformer(inputs_embeds=cls_embeddings, return_dict=False)
        return contextualized_embeds


class LongRangeClassificationModel(nn.Module):
    def __init__(self, config, num_labels=1, hf_model=None):
        super().__init__()
        self.transformer = hf_model if hf_model is not None else AutoModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_labels = num_labels
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.loss_fct = CrossEntropyLoss() if self.num_labels > 1 else BCEWithLogitsLoss()
        self.pooling_method = config.classification_head['pooling_method']
        self.attention = AttentionCompression(config.hidden_size, config.hidden_dropout_prob)

    def pool_words(self, hidden: torch.Tensor, sentence_lens: List[List[int]]) -> List[torch.Tensor]:
        """Take the first tensor of each sentence."""
        pooled_tensors = []
        for doc, doc_sent_lens in zip(hidden, sentence_lens):
            bos_tok_positions = np.cumsum([0] + doc_sent_lens[:-1])
            eos_tok_positions = np.cumsum(doc_sent_lens)
            # make square from text
            sentences = []
            for b, e in zip(bos_tok_positions, eos_tok_positions):
                sent_word_embs = doc[b: e, :]
                if self.pooling_method == 'average':
                    sent_rep = sent_word_embs.mean(dim=0).unsqueeze(0)
                elif self.pooling_method == 'attention':
                    sent_rep = self.attention(sent_word_embs)
                else:  # 'cls'
                    sent_rep = sent_word_embs[0, :].unsqueeze(0)
                sentences.append(sent_rep)
            sent_embs = torch.vstack(sentences)
            pooled_tensors.append(sent_embs)

        # At this point, we can merge all the docs together, it doesn't matter
        return torch.vstack(pooled_tensors)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            sentence_lens: Optional[List[List[int]]] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        """
        Parameters:
             * `input_ids`: one document tokens (list of sentences. Each sentence is a list of ints.)
             * `labels`: list of y_preds [optional].
             * `attention`: list

        """
        if input_ids is not None and len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(dim=0)

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # pool word embeddings
        hidden = outputs[0]
        pooled_output = self.pool_words(hidden, sentence_lens)
        # optionally, in the future we can do something like add LSTM
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # calculate loss
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1, 1))

        output = (logits.view(1, -1),) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


if __name__ == "__main__":
    from transformers import AutoModel, AutoConfig, AutoTokenizer

    hf_model = AutoModel.from_pretrained('roberta-base')
    hf_config = AutoConfig.from_pretrained('roberta-base')
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    # roberta_tokenizer.encode('Hello.') + roberta_tokenizer.encode('My name is Alex.')
    # >>> [   0,  31414, 4,  2,  0, 2387, 766, 16, 2618, 4, 2    ]
    #        <s> Hello  . </s> <s>  My   name is  Alex  . </s>
    #        bos  1st sent  eos bos     2nd sent         . eos

    hf_config.num_labels = 1
    sentence_classifier = SentenceClassificationModel(hf_config, hf_model=hf_model)

    doc = ['Hello.', 'My name is Alex.']
    sentence_toks = list(map(tokenizer.encode, doc))
    sentence_lens = list(map(len, sentence_toks))
    sentence_toks = functools.reduce(lambda a, b: a + b, sentence_toks)
    sent_tensor = torch.tensor([sentence_toks])
    labels = [0, 1]
    labels_tensor = torch.tensor([labels]).to(float)
    #
    sentence_classifier(input_ids=sent_tensor, sentence_lens=[sentence_lens], labels=labels_tensor)