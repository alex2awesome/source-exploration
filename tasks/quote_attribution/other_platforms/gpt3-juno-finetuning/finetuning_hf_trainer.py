import os
import sys
import json
import torch
from dataclasses import dataclass, field
from typing import Optional, List
from torch.utils.data import Dataset
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoTokenizer, AutoConfig, AutoModelForCausalLM,
    Trainer, TrainingArguments, HfArgumentParser
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
import math

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
MAX_SEQUENCE_LEN = 2048

## notes:
## -----------------------------
# gpt neo has 32 layers
# gpt2 medium has 24 layers
# gpt juno has 28 layers

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
    datum_order: Optional[str] = field(
        default=None,
        metadata={'help': 'One of {None, "shortest-first", "longest-first"}'}
    )


def load_data(file):
    train_input, val_input = [], []
    with open(file) as f:
        for line in f:
            if line != '':
                dat = json.loads(line)
                assert dat['split'] in ['train', 'test']
                if dat['split'] == 'train':
                    train_input.append(dat)
                elif dat['split'] == 'test':
                    val_input.append(dat)
    return train_input, val_input


def freeze_all_params(subgraph):
    for p in subgraph.parameters():
        p.requires_grad = False


class TokenizedDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.labels = []

        for dat in txt_list:
            prompt_toks = tokenizer.encode(dat['prompt'])
            completion_toks = tokenizer.encode(dat['completion'])
            input_ids = prompt_toks + completion_toks
            labels = [-100] * len(prompt_toks) + completion_toks
            if len(input_ids) <= max_length:
                self.input_ids.append(torch.tensor(input_ids))
                self.labels.append(torch.tensor(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'label_ids': self.labels[idx]
        }


if __name__ == '__main__':
    parser = HfArgumentParser((RunnerArguments, ModelArguments, DatasetArguments, TrainingArguments,))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        runner_args, model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        runner_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # weird bug
    if training_args.report_to == ['null']:
        training_args.report_to = []

    # Load the models
    if runner_args.platform == 'gcp':
        model_args.cache_dir = '/dev'
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    configuration = AutoConfig.from_pretrained(
        model_args.model_name_or_path, output_hidden_states=False, cache_dir=model_args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, config=configuration, cache_dir=model_args.cache_dir)

    # freeze layers
    if model_args.freeze_layers is not None:
        for layer in model_args.freeze_layers:
            freeze_all_params(model.transformer.h[layer])

    # Load the data
    train_input, val_input = load_data(data_args.dataset_name)
    if data_args.datum_order is not None:
        if data_args.datum_order == 'longest-first':
            train_input = sorted(train_input, key=lambda x: -len(x['prompt']))
        elif data_args.datum_order == 'shortest-first':
            train_input = sorted(train_input, key=lambda x: len(x['prompt']))

    train_dataset = TokenizedDataset(train_input, tokenizer, max_length=data_args.max_sequence_len)
    eval_dataset = TokenizedDataset(val_input, tokenizer, max_length=data_args.max_sequence_len)
    print('{:>5,} training samples'.format(len(train_dataset)))
    print('{:>5,} validation samples'.format(len(eval_dataset)))

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(
                train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        print("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(
            eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))
        perplexity = math.exp(metrics["eval_loss"])
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)