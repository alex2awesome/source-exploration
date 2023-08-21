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

    batch_by_columns['input_ids'] = list(map(torch.tensor, batch_by_columns['input_ids']))
    batch_by_columns['attention_mask'] = _get_attention_mask(batch_by_columns['input_ids'])
    batch_by_columns['input_ids'] = pad_sequence(batch_by_columns['input_ids'], batch_first=True)
    batch_by_columns['labels'] = torch.tensor(batch_by_columns['labels'])

    return batch_by_columns



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
        self.max_length = 4000
        self.source_encoding_method = config.source_encoding_method
        self.process_data(doc_list)

    def process_data(self, doc_list):
        for doc in doc_list:
            input_ids, doc_label = self.process_one_doc(doc)
            if input_ids is not None:
                self.input_ids.append(input_ids)
                self.labels.append(doc_label)

    def process_one_doc(self, doc):
        doc_label = int(doc['label'])

        # encode and chunk
        doc_text = ' '.join(list(map(lambda x: x.strip(), doc['sent'])))
        doc_tokens = self.tokenizer.encode(doc_text)
        if len(doc_tokens) <= self.max_length:
            return doc_tokens, doc_label
        else:
            return None, None

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        output_dict = {
            'input_ids': self.input_ids[idx],
            'labels': self.labels[idx]
        }
        return output_dict