import csv
import torch
import torch.optim
import torch.utils.data as data
import pytorch_lightning as pl
from transformers import BertTokenizer, GPT2Tokenizer, RobertaTokenizer

# import sys, os
# here = os.path.dirname(__file__)
# sys.path.insert(0, here)
from .utils_general import (
    reformat_model_path,
    get_idx2class,
    _get_attention_mask,
    format_local_vars,
    transpose_dict,
    get_fh
)
from torch.nn.utils.rnn import pad_sequence
import itertools


class Dataset(data.Dataset):
    def __init__(self, X, y, v=None, split=None):
        """Reads source and target sequences from txt files."""
        self.X = X
        self.y = y
        self.v = v # version number
        self.split = split

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        data = {}
        data["X"] = self.X[index]
        data["y"] = self.y[index]
        if self.v is not None:
            data['add_features'] = self.v[index]
        return data

    def __iter__(self):
        if self.v is not None:
            return zip(self.X, self.y, self.split, self.v)
        else:
            return zip(self.X, self.y, self.split, [None] * len(self.X))


def _get_split(row):
    if '/test/' in row:
        return 'test'
    elif '/train/' in row:
        return 'train'
    elif '/validation/' in row:
        return 'val'


class BaseDiscourseDataModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data_fp = kwargs.get('data_fp')
        self.add_eos_token = (kwargs.get('model_type') == "gpt2")
        self.max_length_seq = kwargs.get('max_length_seq')
        self.max_num_sentences = kwargs.get('max_num_sentences')
        self.batch_size = kwargs.get('batch_size')
        self.num_cpus = kwargs.get('num_cpus')
        self.split_type = kwargs.get('split_type')
        self.load_tokenizer(
            model_type=kwargs.get('model_type'), pretrained_model_path=kwargs.get('pretrained_model_path')
        )
        self.config = kwargs.get('config')

    def load_tokenizer(self, model_type, pretrained_model_path):
        if model_type == "gpt2":
            self.tokenizer = GPT2Tokenizer.from_pretrained(reformat_model_path(pretrained_model_path))
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

    def process_row(self, text):
        seq = self.tokenizer.encode(text)
        if self.add_eos_token:
            seq.append(self.tokenizer.eos_token_id)
        seq = torch.tensor(seq, dtype=torch.long)
        seq = seq[:self.max_length_seq]
        return seq

    def setup(self, stage=None):
        """
            Download and split the dataset before training/testing.
            For Nonsequential datasets, this just splits on the sentences.
            For Sequential datasets (which are nested lists of sentences), this splits on the documents.

            Occurs on every GPU.
        """
        self.idx2class, self.class2idx = get_idx2class(self.data_fp, self.config)

        if stage in ('fit', None):
            d = self.get_dataset()
            # split randomly
            if self.split_type == 'random':
                train_size = int(0.9 * len(d))
                test_size = len(d) - train_size
                self.train_dataset, self.test_dataset = torch.utils.data.random_split(d, [train_size, test_size])
            # split by filename
            elif self.split_type == 'key':
                train_dataset = list(filter(lambda x: x[2] in ['train'], iter(d)))
                train_X, train_y, _, add_features = zip(*train_dataset)
                self.train_dataset = Dataset(X=train_X, y=train_y, v=add_features)
                test_dataset = list(filter(lambda x: x[2] in ['test'], iter(d)))
                test_X, test_y, _, add_features = zip(*test_dataset)
                self.test_dataset = Dataset(X=test_X, y=test_y, v=add_features)

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

    def _get_attention_mask(self, x):
        return _get_attention_mask(x, self.max_length_seq)


class NonSequentialDiscourseDataModule(BaseDiscourseDataModule):
    def __init__(self, data_fp, model_type, max_length_seq, batch_size,
                 pretrained_model_path, num_cpus=10, split_type='random',
                 **kwargs,
                 ):
        super().__init__(**format_local_vars(locals()))

    def get_dataset(self):
        """
        Read in dataset as a list of "label \t text" entries.
        Output flat lists of X = [sent_1, sent_2, ...], y = [label_1, label_2]
        """
        x, y, split = [], [], []
        with get_fh(self.data_fp) as f:
            csv_reader = csv.reader(f, delimiter="\t")
            for idx, row in enumerate(csv_reader):
                if idx == 0:
                    if ('event_tag' in row) and ('sentence' in row):
                        continue
                    if ('t_id' in row) and ('s' in row):
                        continue
                if idx < 10:
                    print(row)
                if len(row) > 2:
                    label, text, file = row[0], row[1], row[2]
                else:
                    label, text = row[0], row[1]
                    file = None
                if row:
                    processed_seq = self.process_row(text=text)
                    x.append(processed_seq)
                    y.append(self.class2idx[label])
                if file:
                    split.append(_get_split(file))

        return Dataset(x, y, split=split)

    def collate_fn(self, dataset):
        """
        Takes in an instance of Torch Dataset (or a subclassed instance).
        Expects dataset[i]['X'] to be a list of sentences and dataset[i]['y'] to be a list of labels.
        Returns tensors X_batch, y_batch
        """
        output = {}
        columns = transpose_dict(dataset)
        output['input_ids'] = pad_sequence(columns["X"], batch_first=True)
        output['labels'] = torch.tensor(columns["y"], dtype=torch.long)
        output['attention_mask'] = self._get_attention_mask(columns["X"])
        if self.config.do_version:
            output['add_features'] = torch.tensor(columns['add_features'])
        return output


class SequentialDiscourseDataModule(BaseDiscourseDataModule):
    def __init__(self, data_fp,
                 model_type,
                 max_length_seq,
                 max_num_sentences,
                 batch_size,
                 pretrained_model_path,
                 num_cpus=10,
                 split_type='random',
                 **kwargs
                 ):
        super().__init__(**format_local_vars(locals()))

    def get_dataset(self):
        """
        Read in data as a list of "label \t text \t doc_id \t sent_idx
        Output nested list of:
            * X = [[sent_{1, 1}, sent_{1, 2}...], ...]
            * y = [[label_{1, 1}, label_{1, 2}...], ...]
        """
        split, X, y, v = [], [], [], []
        with get_fh(self.data_fp) as f:
            csv_reader = csv.reader(f, delimiter="\t")
            csv_data = list(csv_reader)
            if csv_data[0] == ['source', 's', 't_id', 's_idx']:
                csv_data = csv_data[1:]

        idx = 0
        for doc_idx, doc in itertools.groupby(csv_data, key=lambda x: x[2]):  # group by doc_id
            # if self.config.local and (idx > 100):
            #     break
            # if idx > 100_000:
            #     break
            sorted_doc = sorted(doc, key=lambda x: int(x[3]))                 # sort by sent_id
            if len(sorted_doc) > self.max_num_sentences:
                continue
            doc_seqs, doc_labels = [], []
            doc_versions = []
            if self.config.use_headline:
                headline = sorted_doc[0][4]
                processed_headline = self.process_row(headline)
                doc_seqs.append(processed_headline)
                doc_labels.append(self.class2idx['headline'])
            for sentence in sorted_doc:
                label, text = sentence[0], sentence[1]
                if self.config.do_doc_pred:  # then, the row is an entire document
                    for s in text.split('<SENT>'):
                        if len(s) > 2:
                            doc_seqs.append(self.process_row(s))
                else:
                    processed_text = self.process_row(text)
                    doc_seqs.append(processed_text)
                if self.config.do_version:
                    doc_versions.append(int(float(doc_idx.split('-')[2])))
                if self.config.do_multitask:
                    label_vec = []
                    for l_idx, l in enumerate(label.split('|||')):
                        label_vec.append(self.class2idx[l_idx][l])
                    doc_labels.append(label_vec)
                else:
                    doc_labels.append(self.class2idx[label])

            # append processed data
            if self.config.separate_heads:
                if self.config.num_labels_pred_window != -1:
                    n_back = min(self.config.num_labels_pred_window, len(doc_labels))
                else:
                    n_back = len(doc_labels)
                trials = ['main'] + \
                         list(map(lambda i: 'backwards %s' % (i + 1), range(n_back))) + \
                         list(map(lambda i: 'forward %s' % (i + 1), range(n_back)))
                for t in trials:
                    X.append(doc_seqs)
                    y.append(doc_labels)
                    v.append(t)
                    split.append(_get_split(doc_idx))
            else:
                X.append(doc_seqs)
                y.append(doc_labels)
                v.append(doc_versions)
                # record dataset built-in splits
                split.append(_get_split(doc_idx))
            idx += 1

        return Dataset(X, y, v=v, split=split)

    def collate_fn(self, dataset):
        """
        Takes in an instance of Torch Dataset (or a subclassed instance).
        Expects dataset[i]['X'] to be a list of list of sentences
            and dataset[i]['y'] to be a list of list of labels.
        Returns lists of tensors X_batch, y_batch
        """
        output = {}
        columns = transpose_dict(dataset)
        output["input_ids"] = list(map(lambda sents: pad_sequence(sents, batch_first=True), columns["X"]))
        output['labels'] = list(map(lambda labels: torch.tensor(labels, dtype=torch.long), columns["y"]))
        output["attention_mask"] = list(map(lambda sents: self._get_attention_mask(sents), columns['X']))
        if self.config.do_version or self.config.separate_heads:
            output['add_features'] = list(map(lambda a: torch.tensor(a, dtype=torch.long), columns["add_features"]))
        return output


class FlatDocumentDiscourseDataModule(SequentialDiscourseDataModule):
    """
    Outputs dataset where each datapoint is an entire document with a sequence of tags.

    Downstream, contextualized word embeddings are generated from the entire document, then split up by `input_lens`
    into different sentences. The sentence embeddings are generated, contextualized, and classified.
    """
    def __init__(self, data_fp,
                 model_type,
                 max_length_seq,
                 max_num_sentences,
                 batch_size,
                 pretrained_model_path,
                 num_cpus=10,
                 split_type='random',
                 **kwargs
                 ):
        super().__init__(**format_local_vars(locals()))

    def collate_fn(self, dataset):
        """
        Takes in an instance of Torch Dataset (or a subclassed instance).
        Expects dataset[i]['X'] to be a list of list of sentences
            and dataset[i]['y'] to be a list of list of labels.
        Returns dict with key:
             input_ids: list of tensors of len # docs where each tensor is an entire document.
             labels: list of len # docs where each item is labels of len # sentences.
             input_lens: list of len # docs where each item is lengths of each sent in doc.
        """
        columns = transpose_dict(dataset)
        X_lens = []
        for doc in columns['X']:
            sent_lens = list(map(lambda sent: len(sent), doc))
            sent_lens = torch.tensor(sent_lens, dtype=torch.long)
            X_lens.append(sent_lens)

        output = {}
        output['input_ids'] = list(map(lambda sents: torch.cat(sents), columns["X"]))
        output['labels'] = list(map(lambda labels: torch.tensor(labels, dtype=torch.long), columns["y"]))
        output['attention_mask'] = list(map(lambda sents: self._get_attention_mask(sents), columns['X']))
        output['input_lens'] = X_lens
        if self.config.do_version or self.config.separate_heads:
            # output['add_features'] = list(map(lambda a: torch.tensor(a, dtype=torch.long), columns["add_features"]))
            output['add_features'] = columns["add_features"]
        return output


class FlatDocumentDiscourseDataModuleWordSplit(FlatDocumentDiscourseDataModule):
    """
    Outputs dataset where each datapoint is an entire document with a sequence of tags.

    Downstream, contextualized word embeddings are generated from the entire document, then split up by `input_lens`
    into different sentences. The sentence embeddings are generated, contextualized, and classified.

    Like `FlatDocumentDiscourseDataModule` except each document is split up into N training samples based on the number
     of words in the document.
    """
    def __init__(self, data_fp,
                 model_type,
                 max_length_seq,
                 max_num_sentences,
                 batch_size,
                 pretrained_model_path,
                 num_cpus=10,
                 split_type='random',
                 **kwargs
                 ):
        super().__init__(**format_local_vars(locals()))

    def get_dataset(self):
        """
        Read in data as a list of "label \t text \t doc_id \t sent_idx
        Output nested list of:
            * X = [[sent_{1, 1}, sent_{1, 2}...], ...]
            * y = [[label_{1, 1}, label_{1, 2}...], ...]
        """
        split, X, y = [], [], []
        with get_fh(self.data_fp) as f:
            csv_reader = csv.reader(f, delimiter="\t")
            csv_data = list(csv_reader)

        for doc_idx, doc in itertools.groupby(csv_data, key=lambda x: x[2]):  # group by doc_id
            sorted_doc = sorted(doc, key=lambda x: int(x[3]))                 # sort by sent_id
            if len(sorted_doc) > self.max_num_sentences:
                continue
            doc_seqs, doc_labels = [], []
            for sentence in sorted_doc:
                label, text = sentence[0], sentence[1]
                processed_text = self.process_row(text)
                doc_seqs.append(processed_text)
                doc_labels.append(self.class2idx[label])

            # append processed data
            for sent_idx in range(len(doc_seqs)):
                previous_sents, previous_labels = doc_seqs[:sent_idx], doc_labels[:sent_idx]
                curr_sent, curr_label = doc_seqs[sent_idx], doc_labels[sent_idx]

                for w_idx in range(1, len(curr_sent)):
                    datapoint_seqs = previous_sents + [curr_sent[:w_idx]]
                    datapoint_labels = previous_labels + [curr_label]

                    # record generated datapoint
                    X.append(datapoint_seqs)
                    y.append(datapoint_labels)

                    # record dataset built-in splits
                    split.append(_get_split(doc_idx))

        return Dataset(X, y, split=split)
