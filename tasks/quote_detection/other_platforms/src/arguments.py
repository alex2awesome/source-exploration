from dataclasses import dataclass, field
from typing import Optional, List

MAX_SEQUENCE_LEN = 2048
MODEL_TYPES = [
    'google/bigbird-roberta-large',
    'google/bigbird-roberta-base',
    'allenai/longformer-base-4096',
    'allenai/longformer-large-4096',
]


@dataclass
class RunnerArguments:
    """
    Arguments that help us run the script, coordinate environments, etc.
    """
    platform: Optional[str] = field(
        default='local',
        metadata={
            "help": "Where we're running. Either: {'local', 'gcp', 'pluslab', 'bb'}"
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    freeze_layers: List[int] = field(
        default=None,
        metadata={
            'help': 'Which layers of the model to freeze.'
        }
    )
    pooling_method: Optional[str] = field(
        default='average',
        metadata={
            'help': 'Which pooling method to use.'
        }
    )
    context_layer: Optional[str] = field(
        default=None,
        metadata={'help': 'How to contextualize the sentence vectors.'}
    )


@dataclass
class DatasetArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    max_sequence_len: Optional[int] = field(
        default=MAX_SEQUENCE_LEN,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    dataset_order: Optional[str] = field(
        default=None,
        metadata={'help': 'One of {None, "shortest-first", "longest-first"}'}
    )
