import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
# import sys, os
# here = os.path.dirname(__file__)
# sys.path.insert(0, here)

from .config_helper import TransformersConfig
from torch.nn.functional import pad
import os
import csv
import gzip


def freeze_all_params(subgraph):
    for p in subgraph.parameters():
        p.requires_grad = False


def format_layer_freezes(layers):
    """
    Format the input argument --freeze_encoder_layers
    When run locally, it's a list of strings: ['1', '2',...]
    When run remotely, because of reformatting, it's a list of joined strings: ['1 2 3...']

    """
    if not isinstance(layers, list):
        try:
            return int(layers)
        except:
            return

    if len(layers) == 0:
        return

    if len(layers) == 1:
        if isinstance(layers[0], str) and layers[0].isdigit():
            return int(layers[0])

        layers = layers[0].split()

    return list(map(int, layers))


def format_loss_weighting(loss_weight):
    """
    Format the input argument --freeze_encoder_layers
    When run locally, it's a list of strings: ['1', '2',...]
    When run remotely, because of reformatting, it's a list of joined strings: ['1 2 3...']

    """
    if not isinstance(loss_weight, list):
        try:
            return float(loss_weight)
        except:
            return

    if len(loss_weight) == 0:
        return

    if len(loss_weight) == 1:
        if isinstance(loss_weight[0], str) and loss_weight[0].isdigit():
            return float(loss_weight[0])

        loss_weight = loss_weight[0].split()

    return list(map(float, loss_weight))

def get_config(config=None, kwargs={}):
    if config is None:
        config = kwargs.get('config', None)

    if len(kwargs) > 0:
        kwargs.pop('tb_logger', None)
        if config is None:
            config = TransformersConfig.from_dict(kwargs)
        else:
            for k, v in kwargs.items():
                if (k != 'config') and getattr(config, k, None) != v and v is not None: #  k  != 'config'
                    if ('file' in k or 'dir' in k or 'path' in k) and (kwargs['local'] or kwargs['env'] == 'local'):
                        continue
                    config.__setattr__(k, v)

    # temporary defaults for backwards compatibility.
    if getattr(config, 'model_type', '') == '':
        if getattr(config, 'pretrained_files_s3') is not None:
            if 'roberta' in config.pretrained_files_s3:
                config.model_type = 'roberta'
            else:
                config.model_type = 'gpt2'
        else:
            config.model_type = 'gpt2'

    if getattr(config, 'embedding_dim', 0) == 0:
        config.embedding_dim = 1024
    if getattr(config, 'num_contextual_layers', None) is None:
        config.num_contextual_layers = 2
    if getattr(config, 'num_sent_attn_heads', None) is None:
        config.num_sent_attn_heads = 2
    if getattr(config, 'context_layer', None) is None:
        config.context_layer = 'gpt2-sentence'
    if getattr(config, 'max_num_word_positions', None) is None:
        config.max_num_word_positions = 2048

    if getattr(config, 'learning_rate', None) is None:
        config.learning_rate = 5e-05
    if getattr(config, 'num_warmup_steps', None) is None:
        config.num_warmup_steps = 0
    if getattr(config, 'num_steps_per_epoch', None) is None:
        config.num_steps_per_epoch = 800

    if getattr(config, 'freeze_transformer', None) is None:
        config.freeze_transformer = True
    if getattr(config, 'freeze_embedding_layer', None) is None:
        config.freeze_embedding_layer = False
    if getattr(config, 'freeze_encoder_layers', None) is None:
        config.freeze_encoder_layers = False
    if getattr(config, 'use_tsa', None) is None:
        config.use_tsa = False
    return config


















##

def process_eval_output(eval_output):
    preds = eval_output.predictions  ## these are the logits from the model
    labels = eval_output.label_ids
    if len(preds.shape) == 2:
        preds = preds.argmax(axis=1)
    return preds, labels


def f1_prediction_multiclass(eval_output):
    preds, labels = process_eval_output(eval_output)
    return classification_report(labels, preds, output_dict=True)


def f1_and_confusion_matrix(eval_output):
    output = {}
    preds, labels = process_eval_output(eval_output)
    output['classification_report'] = f1_prediction_multiclass(eval_output)
    c = confusion_matrix(preds, labels)
    output['confusion_matrix'] = str(c.tolist())
    return output


# mappers
def get_log_lines(s3_obj):
    log_str = s3_obj.get()['Body'].read().decode('utf-8')
    return log_str.split('\n')


def split_eval_lines(log_lines=None,  s3_obj=None):
    import ast
    import pandas as pd
    if not log_lines:
        log_lines = get_log_lines(s3_obj)
    ## eval lines
    search_term = "{'eval_loss"
    eval_lines = list(filter(lambda x: search_term in x, log_lines))
    eval_lines = list(map(lambda x: search_term + x.split(search_term)[1], eval_lines))
    eval_dicts = list(map(ast.literal_eval, eval_lines))
    if 'eval_classification_report' in eval_dicts[0]:
        list(map(lambda x: x.update(x.pop('eval_classification_report')), eval_dicts))
    eval_df = pd.DataFrame(eval_dicts)
    return eval_df


def split_param_lines(log_lines=None,  s3_obj=None):
    import ast
    if not log_lines:
        log_lines = get_log_lines(s3_obj)
    ## parse param lines
    params_lines = []
    in_params = False
    for line in log_lines:
        if in_params:
            params_lines.append(line)
        if 'MODEL PARAMS:' in line:
            in_params = True
        if line == '}':
            in_params = False
    params_lines = params_lines[1:]
    params_lines = ['{'] + params_lines
    param_str = ' '.join(params_lines).replace('true', 'True').replace('false', 'False').replace('null', 'None')
    param_dict = ast.literal_eval(param_str)
    return param_dict


class LabelMappers():
    def __init__(self):
        self.get_label_mappers()
        self.label_order = ['eval_6',
                            'eval_7',
                            'eval_0',
                            'eval_1',
                            'eval_5',
                            'eval_2',
                            'eval_3',
                            'eval_4',
                            'eval_8',
                            'eval_macro avg',
                            'eval_weighted avg'
                            ]

        self.model_label_map = {
            'eval_6': 'Main Event',
            'eval_7': 'Consequence',
            'eval_0': 'Previous Event',
            'eval_1': 'Current Context',
            'eval_5': 'Historical Event',
            'eval_2': 'Anecdotal Event',
            'eval_3': 'Evaluation',
            'eval_4': 'Expectation',
            'eval_8': 'Error Tag',
            'eval_macro avg': 'Macro Avg.',
            'eval_weighted avg': 'Weighted Avg.'
        }

        self.dataset_to_paper_map = {
            'Main': 'Main Event',
            'Main_Consequence': 'Consequence',
            'Cause_General': 'Previous Event',
            'Cause_Specific': 'Current Context',
            'Distant_Historical': 'Historical Event',
            'Distant_Anecdotal': 'Anecdotal Event',
            'Distant_Evaluation': 'Evaluation',
            'Distant_Expectations_Consequences': 'Expectation',
            'macro avg': 'Macro Avg.',
            'weighted avg': 'Weighted Avg.'
        }
        self.label_mapper = [
            (6,  '6',            'eval_6',            'Main',               'Main Event',       'M1'),
            (7,  '7',            'eval_7',            'Main_Consequence',   'Consequence',      'M2'),
            (0,  '0',            'eval_0',            'Cause_General',      'Previous Event',   'C2'),
            (1,  '1',            'eval_1',            'Cause_Specific',     'Current Context',  'C1'),
            (5,  '5',            'eval_5',            'Distant_Historical', 'Historical Event', 'D1'),
            (2,  '2',            'eval_2',            'Distant_Anecdotal',  'Anecdotal Event',  'D2'),
            (3,  '3',            'eval_3',            'Distant_Evaluation', 'Evaluation',       'D3',),
            (4,  '4',            'eval_4',            'Distant_Expectations_Consequences', 'Expectation', 'D4'),
            (8,  '8',            'eval_8',            'Error_Tag',          'Error Tag',         'E'),
            (9,  'macro avg',    'eval_macro avg',    'eval_macro avg',     'Macro Avg.',        'Macro'),
            (10, 'weighted avg', 'eval_weighted avg', 'eval_weighted avg',  'Weighted Avg.',     'Weighted')
        ]
        self.cols = ['idx', 'new_model_output', 'model_output',  'annotated_label', 'paper_label', 'paper_shorthand']
        self.col_to_idx =  {
            'Cause_General': 0,
            'Cause_Specific': 1,
            'Distant_Anecdotal': 2,
            'Distant_Evaluation': 3,
            'Distant_Expectations_Consequences': 4,
            'Distant_Historical': 5,
            'Main': 6,
            'Main_Consequence': 7,
            'error': 8
        }

    def get_label_mappers(self):
        import pandas as pd
        self.label_mapper_df = pd.DataFrame(self.label_mapper, columns=self.cols)

        ##
        label_cols = ['M1', 'M2', 'C1', 'C2', 'D1', 'D2', 'D3', 'D4']
        agg_cols = ['eval_macro avg', 'eval_micro avg']
        self.col_order = label_cols + agg_cols

        self.model_to_paper_short = (self.label_mapper_df
         .assign(output_map=lambda df: df['idx'].apply(lambda x: 'eval_%s' % x))
         .set_index('output_map')['paper_shorthand'].to_dict()
        )
        self.paper_short_to_model = {v:k for k,v in self.model_to_paper_short.items()}

        self.full_label_mapper_df = pd.merge(
            left=pd.Series(self.paper_short_to_model).to_frame('model_output'),
            right=self.label_mapper_df,
            left_index=True,
            right_on='paper_shorthand'
        )

    # todo: make a function that formats the kind of output table that you want.
    def process_eval_s(self, s):
        import pandas as pd
        output_dict = pd.Series([])
        for col in self.col_order[:-2]:
            output_dict['%s f1-score' % col] = s[self.col_mapper[col]]['f1-score']
        macro_to_get = ['precision', 'recall', 'f1-score']
        for to_get in macro_to_get:
            output_dict['macro avg %s' % to_get] = s['eval_macro avg'].get(to_get)
        output_dict['weighted avg f1-score'] = s['eval_weighted avg']['f1-score']
        return output_dict


# mixin handlers
class AutoMixinMeta(type):
    """
        Helps us conditionally include Mixins, which is useful if we want to switch between different
        combinations of models (ex. SBERT with Doc Embedding, RoBERTa with positional embeddings).

        class Sub(metaclass = AutoMixinMeta):
            def __init__(self, name):
            self.name = name
    """

    def __call__(cls, *args, **kwargs):
        try:
            mixin = kwargs.pop('mixin')
            if isinstance(mixin, list):
                mixin_names = list(map(lambda x: x.__name__, mixin))
                mixin_name = '.'.join(mixin_names)
                cls_list = tuple(mixin + [cls])
            else:
                mixin_name = mixin.__name__
                cls_list = tuple([mixin, cls])

            name = "{}With{}".format(cls.__name__, mixin_name)
            cls = type(name, cls_list, dict(cls.__dict__))
        except KeyError:
            pass
        return type.__call__(cls, *args, **kwargs)


class Mixer(metaclass = AutoMixinMeta):
    """ Class to mix different elements in.

            model = Mixer(config=config, mixin=[SBERTMixin, BiLSTMMixin, TransformerBase])
    """
    pass


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def reshape_and_pad_sequence(hidden, sequence_lens, device=None):
    """
    Take in a flattened sequence of sentences and reshape it into a padded cube.

    Params:
        * hidden: input token sequence of shape (# tokens in doc X embedding dim )
        * sequence_lens: list of sentence lengths to split the sequence into.

    Returns:
        * output_hidden: matrix of shape (# sentences, max(# tokens in sentence), embedding_dim)
            unless
    """
    if device is None:
        device = get_device()

    # if multiple documents are passed in, we assume every document is the same length.
    # this is true if we're testing candidate sentences.
    max_seq_len = max(sequence_lens)
    cum_seq_lens = torch.cumsum(sequence_lens, dim=0)
    start_idxs = torch.cat((torch.zeros(1, device=device), cum_seq_lens[:-1])).to(torch.int16)
    stop_idxs = cum_seq_lens
    # one document is passed in
    if len(hidden.shape) == 2:
        num_sents, embedding_dim = hidden.shape
        max_seq_len = max(sequence_lens)
        output_hidden = torch.zeros((len(sequence_lens), max_seq_len, embedding_dim), device=device)
        for idx, (s, e) in enumerate(zip(start_idxs, stop_idxs)):
            sentence_emb = hidden[s:e]
            padded_emb = pad(sentence_emb, (0, 0, 0, max_seq_len - (e - s)))
            output_hidden[idx] = padded_emb

    # multiple documents passed in.
    elif len(hidden.shape) == 3:
        num_docs, num_sents, embedding_dim = hidden.shape
        output_hidden = torch.zeros((num_docs, len(sequence_lens), max_seq_len, embedding_dim), device=device)
        for doc_idx in range(num_docs):
            for sent_idx, (s, e) in enumerate(zip(start_idxs, stop_idxs)):
                curr_sent_len = (e - s)
                sentence_emb = hidden[doc_idx][s:e]
                padded_emb = pad(sentence_emb, (0, 0, 0, max_seq_len - curr_sent_len))
                output_hidden[doc_idx][sent_idx] = padded_emb

    return output_hidden


def vec_or_nones(vec, output_len):
    return vec if vec is not None else [None] * output_len


def _get_len(x):
    # x is already a length
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, list):
        return len(x)
    else:
        return x.shape.numel()


def _get_attention_mask(x, max_length_seq=10000):
    max_len = max(map(lambda y: _get_len(y), x))
    max_len = min(max_len, max_length_seq)
    attention_masks = []
    for x_i in x:
        input_len = _get_len(x_i)
        if input_len < max_length_seq:
            mask = torch.cat((torch.ones(input_len), torch.zeros(max_len - input_len)))
        else:
            mask = torch.ones(max_length_seq)
        attention_masks.append(mask)
    return torch.stack(attention_masks)

def reformat_model_path(x, args=None):
    fp_marker = './'
    if (
            (os.environ.get('env') == 'bb' or (args is not None and getattr(args, 'env', 'local') == 'bb'))
            and (not x.startswith(fp_marker))
    ):
        return os.path.join(fp_marker, x)
    else:
        return x


def get_fh(fp):
    if '.gz' in fp:
        fh = gzip.open(fp, 'rt')
    else:
        fh = open(fp)
    return fh

def get_idx2class(dataset_fp, config=None, use_headline=False):
    if (config is not None) and config.do_multitask:
        tasks_idx2class = []
        with get_fh(dataset_fp) as f:
            csv_reader = csv.reader(f, delimiter="\t")
            for row_idx, row in enumerate(csv_reader):
                if row:
                    ls = row[0].split('|||')
                    if row_idx == 0:
                        for _ in ls:
                            tasks_idx2class.append(set())
                    for l_idx, l in enumerate(ls):
                        tasks_idx2class[l_idx].add(l)
        tasks_class2idx = list(map(lambda idx2class: {v:k for k,v in enumerate(idx2class)}, tasks_idx2class))
        return tasks_idx2class, tasks_class2idx
    else:
        classes = set()
        with get_fh(dataset_fp) as f:
            csv_reader = csv.reader(f, delimiter="\t")
            for row in csv_reader:
                if row:
                    classes.add(row[0])
        if use_headline or ((config is not None) and config.use_headline):
            classes.add('headline')
        idx2class = sorted(classes)
        class2idx = {v:k for k,v in enumerate(idx2class)}
        return idx2class, class2idx


def format_local_vars(locals):
    locals.pop('self', '')
    locals.pop('__class__', '')
    for k, v in locals.get('kwargs', {}).items():
        locals[k] = v
    return locals


def transpose_dict(dicts):
    """Take a dictionary in record-format and translate it into a columnar dict.

    [{'a': 1, 'b':2}, {'a':2, 'b':3}] -> {'a': [1,2], 'b': [2, 3]}
    """
    columns = {}
    for key in dicts[0].keys():
        columns[key] = list(map(lambda d: d[key], dicts))
    return columns