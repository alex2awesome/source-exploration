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
import numpy as np
from collections import defaultdict

def load_data(args):
    here = os.path.dirname(__file__)

    file = os.path.join(here, args.dataset_name)
    train_input, val_input = [], []
    other_datasets = defaultdict(list)

    with jsonlines.open(file) as f:
        for dat in f:
            assert dat['split'] in ['train', 'test']
            if dat['split'] == 'train':
                train_input.append(dat)
            elif dat['split'] == 'test':
                val_input.append(dat)
                if dat.get('category') is not None:
                    other_datasets[dat['category']].append(dat)

    if args.gold_label_dataset_name is not None:
        with jsonlines.open(args.gold_label_dataset_name) as f:
            for dat in f:
                assert dat['split'] in ['train', 'test']
                if dat['split'] == 'train':
                    other_datasets['gold-label-train'].append(dat)
                elif dat['split'] == 'test':
                    other_datasets['gold-label-test'].append(dat)

    if args.datum_order == 'shortest-first':
        train_input = sorted(train_input, key=lambda x: len(x['sent']))
        val_input = sorted(val_input, key=lambda x: len(x['sent']))

    return train_input[:args.max_train_samples], val_input[:args.max_val_samples], other_datasets


def compute_metrics(eval_preds):
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")
    logits, labels = eval_preds
    predictions = logits.argmax(axis=1)
    if isinstance(labels, list) and len(labels) == 1:
        labels = labels[0]
    return {
        'f1': f1_metric.compute(predictions=predictions, references=labels)['f1'],
        'accuracy': accuracy_metric.compute(predictions=predictions, references=labels)['accuracy'],
        'num_1_preds': sum(predictions),
        'num_0_preds': sum(predictions == 0),
        'sample': str(list(predictions[:20])),
        'all-ones-accuracy': (1 == labels).mean(),
        'all-ones-f1': f1_metric.compute(predictions=[1] * len(labels), references=labels)['f1'],
    }


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


if __name__ == '__main__':
    parser = HfArgumentParser((RunnerArguments, ModelArguments, DatasetArguments, TrainingArguments,))
    runner_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.model_type == 'full-sequence':
        from baseline_document_classification import TokenizedDataset, collate_fn
        from transformers import AutoModelForSequenceClassification as ModelClass
    else:
        from sentence_model import DocClassificationModelSentenceLevel as ModelClass
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
    config.num_contextual_layers = model_args.num_contextual_layers
    config.frozen_layers = model_args.freeze_layers
    config.freeze_embeddings = model_args.freeze_embeddings
    config.word_pooling_method = model_args.word_pooling_method
    config.sent_pooling_method = model_args.sent_pooling_method
    config.type_vocab_size = 2  # to specify source-style inputs
    config.source_encoding_method = model_args.source_encoding_method
    config.max_num_sources = model_args.max_num_sources
    config.max_sequence_len = data_args.max_sequence_len
    config.use_input_ids = model_args.use_input_ids
    config.use_source_ids = model_args.use_source_ids
    config.problem_type = 'single_label_classification'

    if model_args.model_type == 'full-sequence':
        model = ModelClass.from_pretrained(model_args.model_name_or_path, config=config, cache_dir=model_args.cache_dir)
    else:
        model = ModelClass(config=config, hf_model=hf_model)

    # Load the data
    train_input, val_input, other_input = load_data(data_args)
    train_dataset = TokenizedDataset(doc_list=train_input, config=config, tokenizer=tokenizer)
    eval_dataset = TokenizedDataset(doc_list=val_input, config=config, tokenizer=tokenizer)
    other_datasets = {
        k: TokenizedDataset(doc_list=v, config=config, tokenizer=tokenizer) for k, v in other_input.items()
    }

    print('{:>5,} training samples'.format(len(train_dataset)))
    print('{:>5,} validation samples'.format(len(eval_dataset)))

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,  # for datacollator with padding
    )
    trainer.add_callback(EvaluateCallback(trainer, eval_datasets=other_datasets.items()))

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


