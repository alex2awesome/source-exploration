import torch
import re
import csv
import torch.optim
import torch.utils.data as data
import pytorch_lightning as pl
import os
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, GPT2Tokenizer, RobertaTokenizer
from models_neural.src.utils_general import (
    reformat_model_path,
    format_local_vars,
    transpose_dict,
)
import numpy as np
from unidecode import unidecode
import itertools
import spacy
import random
from tqdm.auto import tqdm

try: # version 3.0.2
    from transformers.tokenization_gpt2 import AddedToken
except: # version > 3.0.2
    pass

from .utils_data_processing_helpers import (
    get_source_candidates,
    reconcile_candidates_and_annotations,
    cache_doc_tokens,
    generate_indicator_lists,
    build_source_lookup_table,
    generate_training_data,
    # for QA
    cache_doc_tokens_for_qa,
    find_source_offset,
    generate_training_chunk_from_source_offset
)


class Dataset(data.Dataset):
    def __init__(self, X, y=None, split=None):
        """Reads source and target sequences from txt files."""
        self.X = X
        self.y = y
        self.split = split

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        data = {}
        data["input_ids"] = self.X[index]
        if self.y is not None:
            data['labels'] = self.y[index]
        return data


class SourceDataset(data.Dataset):
    def __init__(self, input, split=None):
        """Input is a list of dictionaries containing keys such as:
        {
        'text': list of tokens,
        'target_sentence_ids': list of ints in [0, 1],
        'target_source_ids': list of ints in [0, 1]
        }
        """
        self.input = input
        self.split = split

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx]


def _get_split(row):
    if '/test/' in row:
        return 'test'
    elif '/train/' in row:
        return 'train'
    elif '/validation/' in row:
        return 'val'


max_num_tokens_in_doc = 2045
answer_token_sep = '<ANSWER>'

class BaseFineTuningDataModule(pl.LightningDataModule):
    def __init__(self, config=None, *args, **kwargs):
        super().__init__()
        self.config = config
        self.data_fp = kwargs.get('data_fp')
        self.add_eos_token = (kwargs.get('model_type') == "gpt2")
        self.max_length_seq = kwargs.get('max_length_seq')
        self.max_num_sentences = kwargs.get('max_num_sentences', 100)
        self.batch_size = kwargs.get('batch_size')
        self.num_cpus = kwargs.get('num_cpus')
        self.split_type = kwargs.get('split_type')
        self.split_perc = kwargs.get('split_perc', .9)
        self.load_tokenizer(
            model_type=kwargs.get('model_type'),
            pretrained_model_path=kwargs.get('pretrained_model_path')
        )

    def load_tokenizer(self, model_type, pretrained_model_path):
        if model_type == "gpt2":
            self.tokenizer = GPT2Tokenizer.from_pretrained(reformat_model_path(pretrained_model_path))
            # self.tokenizer.add_special_tokens({
            #     'sep_token': AddedToken('<|sep|>', rstrip=False, lstrip=False, single_word=True, normalized=True)
            # })

        elif model_type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(reformat_model_path(pretrained_model_path))
        elif model_type == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained(reformat_model_path(pretrained_model_path))
        else:
            print('Model path not in {bert, roberta, gpt2}.')

    def prepare_data(self):
        """
        Checks if the data path exists.

        Occurs only on the master GPU.
        """
        if not os.path.exists(self.data_fp):
            raise FileNotFoundError('Data files... make sure to download them from S3!')

    def setup(self, stage=None):
        """
            Download and split the dataset before training/testing.
            For Nonsequential datasets, this just splits on the sentences.
            For Sequential datasets (which are nested lists of sentences), this splits on the documents.

            Occurs on every GPU.
        """
        if stage in ('fit', None):
            d = self.get_dataset()
            # split randomly
            if self.split_type == 'random':
                train_size = int(self.split_perc * len(d))
                test_size = len(d) - train_size
                self.train_dataset, self.test_dataset = torch.utils.data.random_split(d, [train_size, test_size])

            # split by filename
            elif self.split_type == 'key':
                zipped = [d.X, d.split]
                if d.y is not None:
                    zipped.append(d.y)
                else:
                    zipped.append([None] * len(d.X))
                train_dataset = list(filter(lambda x: x[1] in ['train', 'val'], zip(*zipped)))
                train_X, _, train_y = zip(*train_dataset)
                self.train_dataset = Dataset(X=train_X, y=train_y)
                test_dataset = list(filter(lambda x: x[1] in ['test'], zip(*zipped)))
                test_X, _, test_y = zip(*test_dataset)
                self.test_dataset = Dataset(X=test_X, y=test_y)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_cpus
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_cpus
        )

    def get_dataset(self):
        """
        Read in dataset as a list of "label \t text" entries.
        Output flat lists of X = [sent_1, sent_2, ...], y = [label_1, label_2]
        """
        split, X, y = [], [], []
        with open(self.data_fp) as f:
            csv_reader = csv.reader(f, delimiter="\t")
            csv_data = list(csv_reader)

        for doc_idx, doc in itertools.groupby(csv_data, key=lambda x: x[2]):  # group by doc_id
            sorted_doc = sorted(doc, key=lambda x: int(x[3]))                 # sort by sent_id
            if len(sorted_doc) > self.max_num_sentences:
                continue

            doc_seqs, doc_labels = [], []
            for sentence in sorted_doc:
                text = sentence[0]
                label = sentence[1]
                processed_text, processed_labels = self.process_row(text, label)
                doc_seqs.append(processed_text)
                doc_labels.append(processed_labels)

            if len(torch.cat(doc_seqs)) > max_num_tokens_in_doc:
                continue

            # append processed data
            X.append(doc_seqs)
            y.append(doc_labels)

            # record dataset built-in splits
            split.append(_get_split(doc_idx))
        return Dataset(X, y, split=split)


class SourceConditionalGenerationDataset(BaseFineTuningDataModule):
    def __init__(
            self, data_fp, model_type, max_length_seq,
            batch_size, pretrained_model_path, num_cpus=10,
            split_type='random', split_perc=.9, **kwargs,
    ):
        super().__init__(**format_local_vars(locals()))

        self.tokenizer.add_special_tokens({'additional_special_tokens': [answer_token_sep]})
        self.answer_token_id = self.tokenizer.additional_special_tokens_ids[0]

    def process_row(self, text, label, has_label=True):
        x_seq = self.tokenizer.encode(text)
        label_seq = self.tokenizer.encode(label)
        x_seq = x_seq[:self.max_length_seq]
        if has_label:
            x_seq = x_seq + [self.answer_token_id]
            y_seq = [-100] * len(x_seq)

            x_seq = torch.tensor(x_seq + label_seq + [self.tokenizer.eos_token_id])
            y_seq = torch.tensor(y_seq + label_seq + [self.tokenizer.eos_token_id])
        else:
            y_seq = [-100] * len(x_seq)
            x_seq = torch.tensor(x_seq)
            y_seq = torch.tensor(y_seq)

        return x_seq, y_seq

    def collate_fn(self, dataset):
        """
        Takes in an instance of Torch Dataset (or a subclassed instance).
        Expects dataset[i]['X'] to be a list of list of sentences
            and dataset[i]['y'] to be a list of list of labels.
        Returns dict with key:
             input_ids: list of tensors of len # docs where each tensor is an entire document.
             labels: same as `input_ids`, for language modeling.
        """
        columns = transpose_dict(dataset)
        X_batch = list(map(lambda sents: torch.cat(sents), columns["input_ids"]))
        Y_batch = list(map(lambda labels: torch.cat(labels), columns["labels"]))
        return {
            "input_ids": torch.cat(X_batch).unsqueeze(dim=0),
            "labels": torch.cat(Y_batch).unsqueeze(dim=0)
        }


class SourceClassificationDataModule(BaseFineTuningDataModule):
    def __init__(
            self, data_fp, model_type, max_length_seq,
            batch_size, pretrained_model_path, num_cpus=10,
            split_type='random', split_perc=.9, spacy_path=None, *args, **kwargs
    ):

        super().__init__(**format_local_vars(locals()))
        if spacy_path is not None:
            self.nlp = spacy.load(spacy_path)
        else:
            self.nlp = None

    def get_dataset(self):
        """
        Read in dataset as a list of "label \t text" entries.
        Output flat lists of X = [sent_1, sent_2, ...], y = [label_1, label_2]
        """
        split, data_chunk = [], []
        with open(self.data_fp) as f:
            csv_reader = csv.reader(f, delimiter="\t")
            csv_data = list(csv_reader)

        grouped = []
        for doc_idx, doc in itertools.groupby(csv_data, key=lambda x: x[3]):  # group by doc_id
            sorted_doc = sorted(doc, key=lambda x: int(x[2]))  # sort by sent_id
            grouped.append((doc_idx, sorted_doc))

        if self.config.shuffle_data:
            random.shuffle(grouped)

        i = 0
        for doc_idx, doc in tqdm(grouped, total=len(grouped)):
            sorted_doc = sorted(doc, key=lambda x: int(x[2]))                 # sort by sent_id
            sorted_doc = sorted_doc[:self.max_num_sentences]

            # contains ambiguous source
            sources = list(set(map(lambda x: x[1], sorted_doc)))
            ambiguous_sources = list(filter(lambda x: re.search('-\d', x) is not None, sources))
            if len(ambiguous_sources) > 0:
                continue

            sorted_doc[0][0] = 'journalist passive-voice ' + sorted_doc[0][0]
            doc_tok_by_word, doc_tok_by_sent, blank_toks_by_sent, all_doc_tokens = cache_doc_tokens(sorted_doc, self.tokenizer, self.nlp)

            if self.config.num_documents is not None:
                if i > self.config.num_documents:
                    break
                i += 1

            s = _get_split(doc_idx)
            source_cand_df = get_source_candidates(sorted_doc, self.nlp)
            annot_to_cand_mapper = reconcile_candidates_and_annotations(source_cand_df, sorted_doc, self.nlp, split=s)
            source_ind_list, sent_ind_list = generate_indicator_lists(blank_toks_by_sent, doc_tok_by_word, source_cand_df, sorted_doc)
            source_cand_df = build_source_lookup_table(source_cand_df, source_ind_list)
            training_data = generate_training_data(
                sorted_doc, annot_to_cand_mapper, source_cand_df, sent_ind_list, all_doc_tokens, self.config.downsample_negative_data
            )

            # append processed data
            data_chunk.extend(training_data)

            # record dataset built-in splits
            split.extend([s] * len(training_data))

        return SourceDataset(input=data_chunk, split=split)

    def tensorfy_and_pad(self, list_of_lists):
        tensors = list(map(torch.tensor, list_of_lists))
        return pad_sequence(tensors, batch_first=True)[:, :max_num_tokens_in_doc]

    def collate_fn(self, dataset):
        """
        Takes in an instance of Torch Dataset (or a subclassed instance).
        Expects dataset[i]['X'] to be a list of list of sentences
            and dataset[i]['y'] to be a list of list of labels.
        Returns dict with key:
             input_ids: list of tensors of len # docs where each tensor is an entire document.
             labels: same as `input_ids`, for language modeling.
        """
        columns = transpose_dict(dataset)
        X_lens = list(map(len, columns['doc_tokens']))
        X_input_ids = self.tensorfy_and_pad(columns["doc_tokens"])
        X_source_ids = self.tensorfy_and_pad(columns['source_ind_tokens'])
        X_sent_ids = self.tensorfy_and_pad(columns['sentence_ind_tokens'])
        max_len = max(X_lens)
        attention_mask = list(map(lambda x_len: [1] * x_len + [0] * (max_len - x_len), X_lens))
        attention_mask = torch.tensor(attention_mask)
        labels = list(map(int, columns['label']))
        return {
            "input_ids": X_input_ids,
            "target_sentence_ids": X_sent_ids,
            "target_person_ids": X_source_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }


class SourceQADataModule(BaseFineTuningDataModule):
    def __init__(
            self, data_fp, model_type, max_length_seq,
            batch_size, pretrained_model_path, num_cpus=10,
            split_type='random', split_perc=.9, spacy_path=None, *args, **kwargs
    ):

        super().__init__(**format_local_vars(locals()))
        if spacy_path is not None:
            self.nlp = spacy.load(spacy_path)
        else:
            self.nlp = None

    def get_dataset(self):
        """
        Read in dataset as a list of "label \t text" entries.
        Output flat lists of X = [sent_1, sent_2, ...], y = [label_1, label_2]
        """
        split, data_chunk = [], []
        with open(self.data_fp) as f:
            csv_reader = csv.reader(f, delimiter="\t")
            csv_data = list(csv_reader)

        grouped = []
        for doc_idx, doc in itertools.groupby(csv_data, key=lambda x: x[3]):  # group by doc_id
            sorted_doc = sorted(doc, key=lambda x: int(x[2]))  # sort by sent_id
            sorted_doc = list(map(lambda x: [x[0].strip(), x[1], x[2], x[3]], sorted_doc))
            grouped.append((doc_idx, sorted_doc))

        if self.config.shuffle_data:
            random.shuffle(grouped)

        i = 0
        training_data = []
        for doc_idx, doc_to_group in tqdm(grouped, total=len(grouped)):
            s = _get_split(doc_idx)

            # contains ambiguous source
            sources = list(set(map(lambda x: x[1], doc_to_group)))
            ambiguous_sources = list(filter(lambda x: re.search('-\d', x) is not None, sources))
            if len(ambiguous_sources) > 0:
                continue

            # check to see if we're only running this on a small sample
            if self.config.num_documents is not None:
                if i > self.config.num_documents:
                    break
                i += 1

            # otherwise, continue
            doc_to_group[0][0] = 'journalist passive-voice ' + doc_to_group[0][0]
            (
                doc_tok_by_word,
                doc_tok_by_sent,
                all_doc_tokens,
                word_len_cumsum,
                sent_lens,
                sent_len_cumsum
            ) = cache_doc_tokens_for_qa(doc_to_group, self.tokenizer, self.nlp)

            doc_to_group = sorted(doc_to_group, key=lambda x: x[1])  # sort by source

            for source_heads, source_sentences in itertools.groupby(doc_to_group, key=lambda x: x[1]):
                if (not self.config.train_on_none) and (source_heads == 'None'):
                    continue

                for source_head in source_heads.split(';'):
                    source_head = unidecode(source_head).strip()
                    source_chunk = find_source_offset(
                        source_head, source_sentences, doc_to_group, word_len_cumsum, sent_len_cumsum
                    )
                    training_chunks = generate_training_chunk_from_source_offset(
                        source_chunk, source_sentences, all_doc_tokens, sent_lens
                    )
                    training_data.extend(training_chunks)
                    split.append(s)

        return SourceDataset(input=training_data, split=split)

    def tensorfy_and_pad(self, list_of_lists):
        tensors = list(map(torch.tensor, list_of_lists))
        return pad_sequence(tensors, batch_first=True)[:, :max_num_tokens_in_doc]

    def collate_fn(self, dataset):
        """
        Takes in an instance of Torch Dataset (or a subclassed instance).
        Expects dataset[i]['X'] to be a list of list of sentences
            and dataset[i]['y'] to be a list of list of labels.
        Returns dict with key:
             input_ids: list of tensors of len # docs where each tensor is an entire document.
             labels: same as `input_ids`, for language modeling.
        """
        columns = transpose_dict(dataset)
        output = {}
        output['start_positions'] = torch.tensor(columns['start_position'])
        output['end_positions'] = torch.tensor(columns['end_position'])
        output['input_ids'] = self.tensorfy_and_pad(columns["context"])
        output['sentence_ids'] = self.tensorfy_and_pad(columns['sentence_indicator_tokens'])

        output['attention_mask'] = list(map(lambda x: [1] * len(x), columns['context']))
        output['attention_mask'] = self.tensorfy_and_pad(output['attention_mask'])
        return output