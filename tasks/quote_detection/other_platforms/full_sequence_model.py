import torch.nn as nn
from typing import Optional, List

from torch.utils.data import Dataset
from transformers import AutoModel, AutoConfig, RobertaConfig, RobertaModel
import functools

import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence

######################################
# data components

class TokenizedDataset(Dataset):
    def __init__(self, doc_list, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.labels = []
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

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'labels': self.labels[idx],
            'sentence_lens': self.sent_lens[idx]
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

    def pool_words(self, hidden: torch.Tensor, sentence_lens: List[List[int]]) -> List[torch.Tensor]:
        """Take the first tensor of each sentence."""
        pooled_tensors = []
        for doc, doc_sent_lens in zip(hidden, sentence_lens):
            bos_tok_positions = [0] + doc_sent_lens[:-1]
            pooled_tensors.append(doc[bos_tok_positions, :])

        # At this point, we can merge all the docs together, it doesn't matter
        return pooled_tensors

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
        pooled_output = torch.vstack(pooled_output)
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