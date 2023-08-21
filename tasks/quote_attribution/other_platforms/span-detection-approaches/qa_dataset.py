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
            'head': normalize(sent.get('head', '')),
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
    # transpose list of dicts -> dict of lists
    batch_by_columns = {}
    for key in dataset[0].keys():
        batch_by_columns[key] = list(map(lambda d: d[key], dataset))
    #
    output = {}
    to_tensorify_and_pad = ['input_ids', 'token_type_ids']
    to_tensorify = ['start_positions', 'end_positions']
    for col in to_tensorify:
        if isinstance(batch_by_columns[col][0], list):
            to_tensorify_and_pad.append(col)
            to_tensorify.remove(col)

    for col in to_tensorify_and_pad:
        if col in batch_by_columns:
            rows = list(map(torch.tensor, batch_by_columns[col]))
            output[col] = pad_sequence(rows, batch_first=True)
    for col in to_tensorify:
        output[col] = torch.tensor(batch_by_columns[col])
    return output


class QATokenizedDatasetCompressedSentenceToks(Dataset):
    def __init__(self, input_data, hf_tokenizer, max_length=2048, include_nones_as_positives=False):
        self.hf_tokenizer = hf_tokenizer
        self.include_nones_as_positives = include_nones_as_positives
        self.max_length = max_length
        self.data = self.process_data_file(input_data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def process_data_file(self, data, num_documents=None):
        if num_documents is not None:
            data = data[:num_documents]

        tokenized_data = []
        for doc in tqdm(data, total=len(data)):
            if contains_ambiguous_source(doc): continue

            # augment doc and process token data.
            doc[0]['sent'] = 'journalist passive-voice ' + doc[0]['sent']
            doc = clean_doc(doc)
            doc_sents = list(map(lambda x: x['sent'], doc))
            doc_text = ' '.join(doc_sents)
            #
            encoded_data = self.hf_tokenizer(doc_text)
            doc_tokens = encoded_data.input_ids
            if len(doc_tokens) > self.max_length:
                continue

            # groupy by and process by source.
            doc = sorted(doc, key=lambda x: x['head'])  # sort by source
            for source_heads, source_sentences in itertools.groupby(doc, key=lambda x: x['head']):
                if (not self.include_nones_as_positives) and (source_heads == ''): continue

                source_sentences = list(source_sentences)
                for source_head in source_heads.split(';'):
                    if source_head in doc_text:
                        source_start_tok, source_end_tok = get_start_end_toks(source_head, doc_text, encoded_data)

                        # for each sentences this source belongs to, generate a new tokenized datapoint
                        for source_sent in source_sentences:
                            sent_start_tok, sent_end_tok = get_start_end_toks(source_sent['sent'], doc_text, encoded_data)
                            sent_indicator_tokens = [0] * (sent_start_tok) + \
                                                    [1] * (sent_end_tok - sent_start_tok + 1) + \
                                                    [0] * (len(doc_tokens) - sent_end_tok - 1)

                            tokenized_chunk = {
                                'start_positions': source_start_tok,
                                'end_positions': source_end_tok,
                                'input_ids': doc_tokens,
                                'sentence_ind_tokens': sent_indicator_tokens,
                            }
                            tokenized_data.append(tokenized_chunk)

        return tokenized_data


class QATokenizedDataset(Dataset):
    def __init__(self, input_data=None, hf_tokenizer=None, max_length=4096,
                 include_nones_as_positives=False, pretrain_salience=False,
                 loss_window=None, decay=.5, do_score=False
                 ):
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
        self.loss_window = int(loss_window or 0)
        self.decay = decay
        self.pretrain_salience = pretrain_salience
        #
        if (not do_score) and (input_data is not None):
            self.data = self.process_data_file(input_data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def get_loss_window(self, start_position, end_position, num_input_toks):
        if start_position is None and end_position is None:
            return start_position, end_position

        if self.loss_window == 0:
            return start_position, end_position

        start_vec = [0] * num_input_toks
        end_vec = [0] * num_input_toks
        for w in range(-self.loss_window, self.loss_window + 1):
            if ((start_position + w) >= 0) and ((start_position + w) < num_input_toks):
                start_vec[start_position + w] = self.decay ** abs(w)
            if ((end_position + w) >= 0) and ((end_position + w) < num_input_toks):
                end_vec[end_position + w] = self.decay ** abs(w)
        return start_vec, end_vec

    def get_doc_text(self, doc, has_labels=True):
        if has_labels and contains_ambiguous_source(doc):
            return None

        # augment doc and process token data.
        doc[0]['sent'] = 'journalist passive-voice ' + doc[0]['sent']
        doc = clean_doc(doc)
        doc_sents = list(map(lambda x: x['sent'], doc))
        doc_text = ' '.join(doc_sents)

        #
        encoded_data = self.hf_tokenizer(doc_text)
        doc_tokens = encoded_data.input_ids
        return doc_text, doc_tokens, encoded_data

    def process_one_doc_training(self, doc):
        tokenized_doc = []
        doc_text, doc_tokens, encoded_data = self.get_doc_text(doc)

        if len(doc_tokens) > self.max_length:
            return None, None

        # group by and process by source.
        doc = sorted(doc, key=lambda x: x['head'])  # sort by source
        for source_heads, source_sentences in itertools.groupby(doc, key=lambda x: x['head']):
            if (not self.include_nones_as_positives) and (source_heads == ''): continue

            source_sentences = list(source_sentences)
            for source_head in source_heads.split(';'):
                if source_head in doc_text:
                    source_start_tok, source_end_tok = get_start_end_toks(source_head, doc_text, encoded_data)

                    # Only used during training: add training examples where no sentence is specified, just
                    # the source. This is to pretrain the model to pay attention to the salience of potential sources.
                    if self.pretrain_salience:
                        # add a loss window
                        y_s, y_e = self.get_loss_window(source_start_tok, source_end_tok, len(doc_tokens))
                        tokenized_chunk = {
                            'start_positions': y_s,
                            'end_positions': y_e,
                            'input_ids': doc_tokens,
                        }
                        tokenized_doc.append(tokenized_chunk)

                    # for each sentences this source belongs to, generate a new tokenized datapoint
                    for source_sent in source_sentences:
                        tokenized_chunk = self.process_one_sentence_doc(
                            sent=source_sent['sent'],
                            doc_tokens=doc_tokens,
                            source_start_tok=source_start_tok,
                            source_end_tok=source_end_tok,
                            quote_type=source_sent['quote_type']
                        )
                        tokenized_doc.append(tokenized_chunk)

        return tokenized_doc

    def process_one_doc(self, doc):
        tokenized_chunks = []
        doc_text, doc_tokens, encoded_data = self.get_doc_text(doc, has_labels=False)
        if len(doc_tokens) > self.max_length:
            return None
        for sent in doc:
            tokenized_chunk = self.process_one_sentence_doc(sent=sent['sent'], doc_tokens=doc_tokens)
            tokenized_chunks.append(tokenized_chunk)
        return tokenized_chunks

    def process_one_sentence_doc(self, sent, doc=None, doc_tokens=None, source_start_tok=None,
                                 source_end_tok=None, quote_type=None):
        sent = normalize(sent)
        if doc_tokens is None:
            doc_tokens = self.hf_tokenizer(doc)

        sent_ids = self.hf_tokenizer.encode(sent, add_special_tokens=False)
        input_ids = self.hf_tokenizer.build_inputs_with_special_tokens(doc_tokens[1: -1], sent_ids)
        token_type_ids = self.hf_tokenizer.create_token_type_ids_from_sequences(doc_tokens[1: -1], sent_ids)
        if len(input_ids) > self.max_length:
            return None

        y_s, y_e = self.get_loss_window(source_start_tok, source_end_tok, len(input_ids))
        tokenized_chunk = {
            'start_positions': y_s,
            'end_positions': y_e,
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'quote_type': quote_type
        }
        return tokenized_chunk

    def process_data_file(self, data):
        tokenized_data = []
        for doc in tqdm(data, total=len(data)):
            tokenized_doc = self.process_one_doc_training(doc)
            if tokenized_doc is not None:
                tokenized_data.extend(tokenized_doc)

        return tokenized_data

