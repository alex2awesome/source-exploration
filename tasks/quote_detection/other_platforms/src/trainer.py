import os
import sys
import json, jsonlines
import evaluate
from transformers import (
    AutoTokenizer, AutoConfig, AutoModel, Trainer, TrainingArguments, HfArgumentParser,
    RobertaModel
)
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
import math
from scipy.special import expit

import sys
sys.path.insert(0, '.')
from arguments import RunnerArguments, ModelArguments, DatasetArguments
from transformers.trainer_utils import EvalPrediction



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
                val_input.append(dat['data'])

    # order, in case we want to check worst-case memory performance
    if args.dataset_order is not None:
        if args.dataset_order == 'longest-first':
            train_input = sorted(train_input, key=lambda x: -len(x))
        elif args.dataset_order == 'shortest-first':
            train_input = sorted(train_input, key=len)

    return train_input[:args.max_train_samples], val_input[:args.max_val_samples]

import numpy as np
def compute_metrics(eval_preds):
    def flatten_drop_nans(arr):
        if isinstance(arr, list):
            arr = np.vstack(arr)
        arr = arr.flatten()
        return arr[arr != -100]
    metric = evaluate.load("f1")
    logits, labels = eval_preds
    logits, labels = flatten_drop_nans(logits), flatten_drop_nans(labels)
    probas = expit(logits)
    predictions = (probas > .5).astype(int)
    return metric.compute(predictions=predictions, references=labels)


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

    def __init__(self, trainer, eval_dataset) -> None:
        """
        Run an evaluation loop on multiple datasets.

        * `eval_datasets` is a list of tuples where each item is:
            (`dataset prefix`, dataset)
        """
        super().__init__()
        self._trainer = trainer
        self.eval_dataset = eval_dataset

    def on_evaluate(self, args, state, control, **kwargs):
        # if control.should_evaluate:
        metrics_to_dump = {}
        all_labels = defaultdict(list)
        all_logits = defaultdict(list)

        dataloader = self._trainer.get_eval_dataloader(self.eval_dataset)
        for step, inputs in enumerate(dataloader):
            loss, logits, labels = self._trainer.prediction_step(self._trainer.model, inputs, prediction_loss_only=False)
            logits = logits.squeeze().cpu().detach().numpy().tolist()
            labels = labels[0].squeeze().cpu().detach().numpy().tolist()
            categories = self.eval_dataset[step]['categories']
            for sent_idx, (logit, label, cat) in enumerate(zip(logits, labels, categories)):
                all_labels[cat].append(label)
                all_logits[cat].append(logit)
                all_labels['full'].append(label)
                all_logits['full'].append(logit)

        for cat in all_labels.keys():
            labels = np.array(all_labels[cat])
            logits = np.array(all_logits[cat])
            metrics = compute_metrics(EvalPrediction(logits, labels))
            for k, m in metrics.items():
                metrics_to_dump[cat + '_' + k] = m

        print(json.dumps(metrics_to_dump))
        output_dir = self._trainer.args.output_dir
        with open(os.path.join(output_dir, f'callback-metrics-state-{state.global_step}.json'), 'w') as f:
            json.dump(metrics_to_dump, f)


from collections import defaultdict


if __name__ == '__main__':
    parser = HfArgumentParser((RunnerArguments, ModelArguments, DatasetArguments, TrainingArguments,))
    runner_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.model_type == 'full-sequence':
        from full_sequence_model import LongRangeClassificationModel as ModelClass
        from full_sequence_model import TokenizedDataset, collate_fn
    else:
        from sentence_model import SentenceClassificationModel as ModelClass
        from sentence_model import TokenizedDataset, collate_fn

    # weird bug
    if training_args.report_to == ['null']:
        training_args.report_to = []

    if 'wandb' in training_args.report_to:
        import wandb
        wandb.init(project="source-exploration-discrimination")

    # Load the models
    if runner_args.platform == 'gcp':
        model_args.cache_dir = '/dev'

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    hf_model = AutoModel.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)

    config.context_layer = model_args.context_layer
    config.frozen_layers = model_args.freeze_layers
    config.classification_head = {
        'num_labels': 1,
        'pooling_method': model_args.pooling_method,
    }

    model = ModelClass(config=config, hf_model=hf_model)

    # Load the data
    train_input, val_input = load_data(data_args)
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
        compute_metrics=compute_metrics,
        data_collator=collate_fn, ## for datacollator with padding
    )
    trainer.add_callback(EvaluateCallback(trainer, eval_dataset=eval_dataset))

    # Detecting last checkpoint.
    last_checkpoint = get_last_checkpoint_with_asserts(training_args)

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


