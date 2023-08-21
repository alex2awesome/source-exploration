from torch.utils.data import Dataset
import re
import itertools
import torch
from tqdm.auto import tqdm
from unidecode import unidecode
from typing import List, Optional
import pandas as pd
from torch.nn.utils.rnn import pad_sequence


CLEANR = re.compile('<.*?>')
def cleanhtml(raw_html):
    cleantext = re.sub(CLEANR, '', raw_html)
    return cleantext

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


def contains_ambiguous_source(doc):
    # contains ambiguous source
    sources = list(set(map(lambda x: x['head'], doc)))
    sources = list(filter(pd.notnull, sources))
    ambiguous_sources = list(filter(lambda x: re.search('-\d', x) is not None, sources))
    return len(ambiguous_sources) > 0


def normalize(text):
    text = '' if pd.isnull(text) else text
    text = re.sub('\s+', ' ', text)
    return cleanhtml(unidecode(text).strip())


def get_start_end_toks(text, doc_text, tokenized_obj, fail_on_not_found=True):
    text = normalize(text)
    try:
        start_char = doc_text.index(text)
        end_char = start_char + len(text) - 1
        return tokenized_obj.char_to_token(start_char), tokenized_obj.char_to_token(end_char)
    except ValueError:
        if fail_on_not_found:
            raise ValueError('substring not found')
        else:
            return None, None


def fix_quote_type(sent):
    quote_type_mapper = {
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


def clean_doc(doc):
    output_doc = []
    for sent in doc:
        output_doc.append({
            'head': normalize(sent['head']),
            'sent': normalize(sent['sent']),
            'quote_type': fix_quote_type(sent)
        })
    return output_doc


def collate_fn(dataset):
    """
    Takes in an instance of Torch Dataset.
    Returns:
     * input_ids:
     * sentence_ind_tokens:
     * start_position: List[int]
     * end_position: List[int]
    """
    # transpose dict
    batch_by_columns = {}
    for key in dataset[0].keys():
        batch_by_columns[key] = list(map(lambda d: d[key], dataset))

    output = {}
    to_tensorify_and_pad = ['input_ids', 'token_type_ids', 'labels']
    for col in to_tensorify_and_pad:
        if col in batch_by_columns:
            rows = list(map(torch.tensor, batch_by_columns[col]))
            output[col] = pad_sequence(rows, batch_first=True)
    return output


class TokenClassificationDataset(Dataset):
    def __init__(self, input_data, hf_tokenizer, max_length=2048,
                 include_nones_as_positives=False,):
        """
        Generate QA-style dataset for source-span detection.

        * `input_data`: list of documents where each corresponds to.
        * `hf_tokenizer`:
        * `max_length`:
        * `include_nones_as_positives`: also train on none.
        * `pretrain_salience`: include datapoints that don't have sentence data.
        * `loss_window`: reward model for near misses, within a window.
        * `decay`: how much to decay over the loss window.
        """
        self.hf_tokenizer = hf_tokenizer
        self.include_nones_as_positives = include_nones_as_positives
        self.max_length = max_length
        #
        self.data = self.process_data_file(input_data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def process_doc(self, doc):
        tokenized_doc = []
        if contains_ambiguous_source(doc):
            return []

        # augment doc and process token data.
        doc[0]['sent'] = 'journalist passive-voice ' + doc[0]['sent']
        doc = clean_doc(doc)
        doc_sents = list(map(lambda x: x['sent'], doc))
        doc_text = ' '.join(doc_sents)
        #
        encoded_data = self.hf_tokenizer(doc_text)
        doc_tokens = encoded_data.input_ids
        if len(doc_tokens) > self.max_length:
            return []

        # groupy by and process by source.
        doc = sorted(doc, key=lambda x: x['head'])  # sort by source
        for source_heads, source_sentences in itertools.groupby(doc, key=lambda x: x['head']):
            if (not self.include_nones_as_positives) and (source_heads == ''):
                continue
            source_sentences = list(source_sentences)
            for source_head in source_heads.split(';'):
                if source_head in doc_text:
                    source_start_tok, source_end_tok = get_start_end_toks(source_head, doc_text, encoded_data)

                    # for each sentences this source belongs to, generate a new tokenized datapoint
                    for source_sent in source_sentences:
                        sent = normalize(source_sent['sent'])
                        sent_ids = self.hf_tokenizer.encode(sent, add_special_tokens=False)
                        input_ids = self.hf_tokenizer.build_inputs_with_special_tokens(doc_tokens[1: -1], sent_ids)
                        token_type_ids = self.hf_tokenizer.create_token_type_ids_from_sequences(doc_tokens[1: -1],
                                                                                                sent_ids)
                        if len(input_ids) > self.max_length:
                            continue

                        labels = [0] * (source_start_tok) + \
                                 [1] * (source_end_tok - source_start_tok + 1) + \
                                 [0] * (len(input_ids) - source_end_tok - 1)

                        tokenized_chunk = {
                            'input_ids': input_ids,
                            'token_type_ids': token_type_ids,
                            'labels': labels,
                            'quote_type': source_sent['quote_type']
                        }
                        tokenized_doc.append(tokenized_chunk)
        return tokenized_doc

    def process_data_file(self, data):
        tokenized_data = []
        for doc in tqdm(data, total=len(data)):
            tokenized_data.extend(self.process_doc(doc))
        return tokenized_data

