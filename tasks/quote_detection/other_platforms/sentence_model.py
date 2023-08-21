import torch.nn as nn
from typing import Optional, List

from torch.utils.data import Dataset
from transformers import AutoModel, AutoConfig, RobertaConfig, RobertaModel

import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence

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
    def __init__(self, doc_list, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.labels = []
        self.attention = []

        for doc in doc_list:
            doc_tokens = []
            doc_labels = []
            for sentence in doc:
                doc_tokens.append(tokenizer.encode(sentence['sent']))
                doc_labels.append(int(sentence['label']))
            if sum(map(len, doc_tokens)) <= max_length:
                doc_tokens = list(map(torch.tensor, doc_tokens))
                attention_mask = _get_attention_mask(doc_tokens)
                input_ids = pad_sequence(doc_tokens, batch_first=True)
                doc_labels = torch.tensor(doc_labels).unsqueeze(0).to(float) # we need have the labels like this for
                                                                             # this weird `nested_concat` function in the `evaluation_loop` of the `Trainer.py`
                self.input_ids.append(input_ids)
                self.attention.append(attention_mask)
                self.labels.append(doc_labels)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'labels': self.labels[idx],
            'attention_mask': self.attention[idx]
        }


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
        self.sentence_transformer = RobertaModel(config=config)

    def forward(self, cls_embeddings):
        # inputs_embeds: input of shape: (batch_size, sequence_length, hidden_size)
        contextualized_embeds, _ = self.sentence_transformer(inputs_embeds=cls_embeddings, return_dict=False)
        return contextualized_embeds


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


class SentenceClassificationModel(nn.Module):
    def __init__(self, config, num_labels=1, hf_model=None, pooling_method='average'):
        super().__init__()
        self.transformer = hf_model if hf_model is not None else AutoModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_labels = num_labels
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.loss_fct = CrossEntropyLoss() if self.num_labels > 1 else BCEWithLogitsLoss()
        self.pooling_method = pooling_method

    def pool_words(self, hidden, attention_mask):
        if self.pooling_method == 'average':
            return (hidden.T * attention_mask.T).T.mean(axis=1)
        elif self.pooling_method == 'cls':
            return hidden[:, 0, :]

    def process_one_doc(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None):
        if input_ids is not None and len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(dim=0)

        outputs = self.transformer(input_ids, attention_mask=attention_mask)

        # pool word embeddings
        hidden = outputs[0]
        pooled_output = self.pool_words(hidden, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # calculate loss
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1, 1))
        return loss, logits.view(1, -1)

    def forward(self, input_ids: List[torch.Tensor], attention_mask: List[torch.Tensor],
                labels: Optional[List[torch.Tensor]] = None,):
        outputs = list(map(lambda x: self.process_one_doc(*x), zip(input_ids, attention_mask, labels)))
        losses, logits = list(zip(*outputs))
        loss = None if losses[0] == None else sum(losses)
        logits = torch.vstack(logits)
        return (loss, logits) if loss is not None else logits


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

    import functools

    doc = ['Hello.', 'My name is Alex.']
    sentence_toks = list(map(tokenizer.encode, doc))
    sentence_lens = list(map(len, sentence_toks))
    sentence_toks = functools.reduce(lambda a, b: a + b, sentence_toks)
    sent_tensor = torch.tensor([sentence_toks])
    labels = [0, 1]
    labels_tensor = torch.tensor([labels]).to(float)
    #
    sentence_classifier(input_ids=sent_tensor, sentence_lens=[sentence_lens], labels=labels_tensor)