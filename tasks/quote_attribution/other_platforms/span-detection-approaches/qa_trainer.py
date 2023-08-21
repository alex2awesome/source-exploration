import os
import jsonlines
import evaluate
from transformers import (
    AutoTokenizer, AutoConfig, AutoModel, Trainer, TrainingArguments, HfArgumentParser,
    RobertaModel
)
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from arguments import RunnerArguments, ModelArguments, DatasetArguments
import sys
sys.path.insert(0, '.')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np
import json

def load_data(args):
    here = os.path.dirname(__file__)
    file = os.path.join(here, args.dataset_name)
    train_input, val_input = [], []
    with jsonlines.open(file) as f:
        for dat in f:
            assert dat['split'] in ['train', 'test']
            if dat['split'] == 'train':
                train_input.append(dat['data'])
            elif dat['split'] == 'test':
                val_input.append(dat['data'][:args.max_num_sentences])
    if args.datum_order == 'shortest-first':
        train_input = sorted(train_input, key=len)
        val_input = sorted(val_input, key=len)

    return train_input[:args.max_train_samples], val_input[:args.max_val_samples]


def sequence_overlap(start_pred, end_pred, start_label, end_label, window=2):
    """Calculates the match of a single example using the following heuristic:
        * There must be at least one token in the true source predicted.
        * The start/end points cannot be more than `w` tokens away from the true start/end points.
    """
    has_sequence_overlap = (start_pred <= start_label and end_pred >= start_label) or \
                  (start_pred <= end_label and end_pred >= end_label)
    start_and_end_within_window = (abs(start_pred - start_label) <= window) and \
                                  (abs(end_pred - end_label) <= window)

    return has_sequence_overlap and start_and_end_within_window


def sequence_f1(start_pred, end_pred, start_label, end_label):
    start_pred, end_pred = min(start_pred, end_pred), max(start_pred, end_pred)
    common_start = max(start_pred, start_label)
    common_end = min(end_pred, end_label)

    len_common_tokens = max(common_end - common_start, 0)
    len_pred_tokens = max(end_pred - start_pred, 0)
    len_truth_tokens = end_label - start_label

    if (len_pred_tokens == 0) or (len_truth_tokens == 0): return 0

    prec = len_common_tokens / len_pred_tokens
    rec = len_common_tokens / len_truth_tokens

    if prec + rec == 0: return 0
    return 2 * (prec * rec) / (prec + rec)



def compute_metrics(eval_preds):
    def drop_nans(arr):
        if isinstance(arr, list):
            arr = np.vstack(arr)
        arr = arr.flatten()
        return arr[arr != -100]

    (start_logits, end_logits), (start_labels, end_labels) = eval_preds
    start_preds, end_preds = start_logits.argmax(axis=1), end_logits.argmax(axis=1)
    f1s, es = [], []

    for s_pred, e_pred, s_label, e_label in zip(start_preds, end_preds, start_labels, end_labels):
        s_pred, e_pred = min(s_pred, e_pred), max(s_pred, e_pred)
        f1s.append( sequence_f1(s_pred, e_pred, s_label, e_label) )
        es.append( sequence_overlap(s_pred, e_pred, s_label, e_label))

    return {'f1': np.mean(f1s), 'e': np.mean(es)}


def get_last_checkpoint_with_asserts(training_args):
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
    return last_checkpoint


def model_name_or_checkpoint(last_checkpoint, model_args):
    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
        checkpoint = model_args.model_name_or_path
    else:
        checkpoint = None
    return checkpoint


class EvaluateCallback(TrainerCallback):

    def __init__(self, trainer, eval_datasets) -> None:
        """
        Run an evaluation loop on multiple datasets.

        * `eval_datasets` is a list of tuples where each item is:
            (`dataset prefix`, dataset)
        """
        super().__init__()
        self._trainer = trainer
        self.eval_datasets = eval_datasets

    def on_evaluate(self, args, state, control, **kwargs):
        # if control.should_evaluate:
        metrics_to_dump = {}
        for prefix, dataset in self.eval_datasets:
            # self._trainer.evaluate(eval_dataset=dataset, metric_key_prefix=prefix)
            dataloader = self._trainer.get_eval_dataloader(dataset)
            eval_loop_output = self._trainer.evaluation_loop(dataloader=dataloader, description=prefix, metric_key_prefix=prefix)
            metrics = eval_loop_output.metrics
            print(json.dumps(metrics))
            metrics_to_dump.update(metrics)

        output_dir = self._trainer.args.output_dir
        with open(os.path.join(output_dir, f'callback-metrics-state-{state.global_step}.json'), 'w') as f:
            json.dump(metrics_to_dump, f)


from collections import defaultdict
def get_other_datasets(dataset):
    """Expects the key `quote_type` to be in each item of the dataset."""
    other_datasets = defaultdict(list)
    for doc in dataset:
        quote_type = doc.get('quote_type', '')
        if quote_type != '':
            other_datasets[quote_type].append(doc)
    other_datasets['full'] = dataset
    return other_datasets

from typing import Dict, Union, Any
import torch
class QATrainer(Trainer):
    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        inputs['global_step'] = self.state.global_step
        inputs['max_steps'] = self.state.max_steps
        return inputs


if __name__ == '__main__':
    parser = HfArgumentParser((RunnerArguments, ModelArguments, DatasetArguments, TrainingArguments,))
    runner_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.model_type == 'salience':
        from qa_model import QAModelWithSalience as ModelClass
        from qa_dataset import QATokenizedDataset as DatasetClass, collate_fn
    else:
        from qa_model import QAModel as ModelClass
        from qa_dataset import QATokenizedDataset as DatasetClass, collate_fn

    # weird bug
    if training_args.report_to == ['null']:
        training_args.report_to = []

    if 'wandb' in training_args.report_to:
        import wandb
        wandb.init(project="source-exploration-discrimination")

    # Load the models
    if runner_args.platform == 'gcp':
        model_args.cache_dir = '/dev'

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    config.freeze_layers = model_args.freeze_layers
    config.qa_head = {}
    config.include_nones_as_positives = data_args.include_nones_as_positives
    config.attention_type = model_args.attention_type
    config.loss_window = model_args.loss_window

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir, config=config)
    hf_model = AutoModel.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir, config=config)

    model = ModelClass(config=config, hf_model=hf_model)

    # Load the data
    train_input, val_input = load_data(data_args)
    if training_args.do_train:
        train_dataset = DatasetClass(
            train_input, hf_tokenizer=tokenizer, max_length=data_args.max_sequence_len,
            include_nones_as_positives=data_args.include_nones_as_positives,
            pretrain_salience=model_args.model_type == 'salience',
            loss_window=model_args.loss_window
        )
        print('{:>5,} training samples'.format(len(train_dataset)))

    other_datasets = {}
    if training_args.do_eval:
        eval_dataset = DatasetClass(
            val_input, hf_tokenizer=tokenizer, max_length=data_args.max_sequence_len,
            include_nones_as_positives=data_args.include_nones_as_positives,
        )
        other_datasets = get_other_datasets(eval_dataset)

        print('{:>5,} validation samples'.format(len(eval_dataset)))

    trainer = QATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        data_collator=collate_fn  # for datacollator with padding
    )
    trainer.add_callback(EvaluateCallback(trainer, eval_datasets=other_datasets.items()))

    # Detecting last checkpoint.
    last_checkpoint = get_last_checkpoint_with_asserts(training_args)

    # Evaluation
    if training_args.do_eval:
        print("*** Pre-run evaluation ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("pre-training eval", metrics)
        trainer.save_metrics("pre-training eval", metrics)

    # Training
    if training_args.do_train:
        checkpoint = model_name_or_checkpoint(last_checkpoint, model_args)
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

        preds, labels, metrics = trainer.predict(eval_dataset)

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(
            eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("post-training eval", metrics)
        trainer.save_metrics("post-training eval", metrics)

        prediction_output = []
        for preds_doc, labels_doc in zip(preds, labels):
            preds_doc = preds_doc[preds_doc != -100]
            labels_doc = labels_doc[labels_doc != -100]
            prediction_output.append([
                {'pred': float(p),
                 'label': float(l)}
                for p,l in zip(preds_doc, labels_doc)
            ])
        with open(os.path.join(training_args.output_dir, 'prediction_output.jsonl'), 'w') as f:
            jsonlines.Writer(f).write_all(prediction_output)


