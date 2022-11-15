# set up logging before imports

# import sys, os
# here = os.path.dirname(__file__)
# sys.path.insert(0, here)

import json
from models_neural.src.config_helper import training_args
from models_neural.quote_attribution.utils_parser import attach_model_arguments
from models_neural.src.utils_general import format_loss_weighting, reformat_model_path
from models_neural.src.config_helper import TransformersConfig
from models_neural.quote_attribution.language_models import LMModel
from models_neural.quote_attribution.utils_dataset import (
    SourceConditionalGenerationDataset,
    SourceClassificationDataModule,
    SourceClassificationExtraTokens,
    SourceQADataModule
)
from models_neural.quote_attribution.classification_models import (
    SourceClassifier,
    SourceClassifierWithSourceSentVecs,
    SourceQA
)


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers
import torch
from transformers import AutoConfig

import logging
logging.basicConfig(level=logging.INFO)

experiments = {
    'roberta_generation': ('roberta', SourceConditionalGenerationDataset, LMModel),
    'roberta_classification': ('roberta', SourceClassificationDataModule, SourceClassifier),
    'roberta_classification_vecs': ('roberta', SourceClassificationDataModule, SourceClassifierWithSourceSentVecs),
    'roberta_classification_toks': ('roberta', SourceClassificationExtraTokens, SourceClassifier),
    'roberta_qa': ('roberta', SourceQADataModule, SourceQA)
}

local_output_fp = './runs/'


def main(
        args,
        config=None,
        experiment='roberta_classification',
        output_fp='.',
        num_nodes=1,
        num_gpus=1,
        notes='',
        **kwargs
    ):

    accelerator = 'dp'
    accelerator = accelerator if ((args.num_nodes > 1) or (args.num_gpus > 1)) else None

    if not os.path.exists(output_fp):
        os.makedirs(output_fp)

    lm_type, datasetclass, lm_class = experiments[experiment]

    ####
    # load dataset and model classes
    dataset = datasetclass(
        config=config,
        data_fp=config.train_data_file,
        pretrained_model_path=config.pretrained_model_path,
        num_cpus=config.num_dataloader_cpus,
        split_type=args.split_type,
        split_perc=.95,
        model_type=lm_type,
        batch_size=args.batch_size,
        max_length_seq=args.max_length_seq,
        spacy_path=args.spacy_model_file
    )
    dataset.prepare_data()
    dataset.setup(stage='fit')
    config.num_steps_per_epoch = len(dataset.train_dataset)

    model = lm_class(config=config)  # our experimental setup


    #########
    # get TB logger
    if os.environ.get('TENSORBOARD_LOGDIR'):
        tb_logger = loggers.TensorBoardLogger(
            save_dir=os.environ['TENSORBOARD_LOGDIR'],
        )
        # tb_logger.log_hyperparams(config.to_dict())
        tb_logger.log_hyperparams({
                'notes': args.notes,
                'embedding_model_type': config.model_type,
                'dataset_size': len(dataset.train_dataset),
                'experiment': args.experiment,
            }
        )
        tb_logger.save()
    else:
        tb_logger = None

    #################
    #  Train model
    checkpoint_callback = ModelCheckpoint(
        monitor='Validation loss',
        dirpath=local_output_fp,
        filename='trial-%s__epoch={epoch:02d}-perplexity={Validation Perplexity:.2f}' % args.notes,
        save_top_k=1,
        mode='min',
        auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        gpus=num_gpus,
        num_nodes=num_nodes,
        accelerator=accelerator if not args.use_deepspeed else None,
        max_epochs=10,
        logger=tb_logger,
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=config.max_grad_norm,
        # plugins='ddp_sharded',
        plugins='deepspeed_stage_2' if args.use_deepspeed else None,
        precision=16 if args.use_deepspeed else 32
    )
    print('NUM GPUs USING:')
    print(trainer.gpus)

    trainer.fit(model, datamodule=dataset)

    # upload best model
    best_model_path = checkpoint_callback.best_model_path
    if not args.local:
        from models_neural.src.utils_data_access import get_fs
        fs = get_fs()
        # upload model
        fname = os.path.basename(best_model_path)
        remote_path = os.path.join('aspangher', 'source-exploration', output_fp, fname)
        print('uploading model file at %s to: %s...' % (best_model_path, remote_path))
        fs.put(best_model_path, remote_path)

        # upload config
        local_config_path = 'config-%s.json' % args.notes
        remote_config_path = os.path.join('aspangher', 'source-exploration', output_fp, local_config_path)
        with open(local_config_path, 'w') as f:
            config_dict = config.to_dict()
            json.dump(config_dict, f)
        fs.put(local_config_path, remote_config_path)


    # log best metric score
    best_metric = checkpoint_callback.best_model_score
    print('BEST MODEL SCORE: %s' % best_metric)


if __name__ == "__main__":
    import os, argparse, glob

    parser = argparse.ArgumentParser()
    parser = attach_model_arguments(parser)
    parser.add_argument('--local', action='store_true')

    args = parser.parse_args()
    print(args)

    # load data
    here = os.path.dirname(os.path.realpath(__file__))
    args.eval_data_file = None
    if args.local:
        # train and eval files
        args.num_gpus = 0
        args.train_data_file = os.path.join(here, args.train_data_file)
    else:
        from models_neural.src.utils_data_access import download_all_necessary_files
        download_all_necessary_files(args)

    # config
    config = TransformersConfig(cmd_args=args)
    config.pretrained_model_path = reformat_model_path(config.pretrained_model_path)
    config.num_train_epochs = config.num_train_epochs if hasattr(config, 'num_train_epochs') else training_args.num_train_epochs
    config.loss_weighting = format_loss_weighting(config.loss_weighting)
    config.use_cache = False
    config.num_output_tags = 1

    if not hasattr(config, 'env'):
        config.env = os.environ.get('env')

    # set up model
    logging.info('MODEL PARAMS:')
    logging.info(config.to_json_string(use_diff=False))
    dump_config_now = True
    if dump_config_now:
        local_config_path = 'config-%s.json' % args.notes
        with open(local_config_path, 'w') as f:
            json.dump(config.to_dict(), f)
    logging.info('END MODEL PARAMS')

    main(
        args=args,
        config=config,
        **vars(args),
    )