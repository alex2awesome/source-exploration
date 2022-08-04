# set up logging before imports

# import sys, os
# here = os.path.dirname(__file__)
# sys.path.insert(0, here)

from models_neural.src.config_helper import training_args
from models_neural.quote_attribution.utils_parser import attach_model_arguments
from models_neural.src.utils_general import format_loss_weighting, reformat_model_path
from models_neural.src.config_helper import TransformersConfig, get_transformer_config
from models_neural.quote_attribution.language_models import LMModel, GPT2LMHeadModel, RobertaForCausalLM
from models_neural.quote_attribution.utils_dataset import (
    SourceConditionalGenerationDataset,
    SourceClassificationDataModule,
    SourceQADataModule
)
from models_neural.quote_attribution.classification_models import (
    SourceClassifier,
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
    'roberta_qa': ('roberta', SourceQADataModule, SourceQA)
}

local_output_fp = './runs/'

def get_config(pretrained_path=None, cmd_args=None, config=None):
    """
    """
    pretrained_config = AutoConfig.from_pretrained(pretrained_path)

    # update pretrained config with our Argparse config.
    for k in cmd_args.__dict__:
        pretrained_config.__dict__[k] = cmd_args.__dict__[k]

    if config is not None:
        for k in config.__dict__:
            pretrained_config.__dict__[k] = config.__dict__[k]

    # return
    return pretrained_config


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
    #
    config = get_config(args.pretrained_model_path, args, config=config)
    config.pretrained_cache_dir = args.pretrained_model_path
    config.use_cache = False
    config.num_output_tags = 1

    ####
    # load dataset and model classes
    dataset = datasetclass(
        config=config,
        data_fp=config.main_data_file,
        pretrained_model_path=config.pretrained_cache_dir,
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
        tb_logger.log_hyperparams({
                'notes': args.notes,
                'embedding_model_type': config.model_type,
                'dataset_size': len(dataset.train_dataset),
                'experiment': args.experiment,
                # trainer params
                'batch_size': config.batch_size,
                'num_warmup_steps': config.num_warmup_steps,
                'learning_rate': config.learning_rate,
                'gradient_accumulation': config.accumulate_grad_batches,
            }
        )
    else:
        tb_logger = None

    #################
    #  Train model
    checkpoint_callback = ModelCheckpoint(
        monitor='Validation Perplexity',
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

    # cache
    # upload best model
    best_model_path = checkpoint_callback.best_model_path
    if args.env == 'bb':
        fs = get_fs()
        fname = os.path.basename(best_model_path)
        remote_path = os.path.join('aspangher', 'source-exploration', output_fp, fname)
        print('uploading model file at %s to: %s...' % (best_model_path, remote_path))
        fs.put(best_model_path, remote_path)
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
        args.pretrained_path = args.pretrained_model_path
    else:
        from models_neural.src.utils_data_access import download_all_necessary_files
        download_all_necessary_files(args)

    # config
    config = TransformersConfig(cmd_args=args)
    config.pretrained_cache_dir = reformat_model_path(args.pretrained_path)
    config.main_data_file = os.path.join(here, args.train_data_file)
    config.max_position_embeddings = args.max_num_word_positions
    config.num_warmup_steps = training_args.warmup_steps
    config.num_train_epochs = config.num_train_epochs if hasattr(config, 'num_train_epochs') else training_args.num_train_epochs
    config.loss_weighting = format_loss_weighting(config.loss_weighting)
    if not hasattr(config, 'env'):
        config.env = os.environ.get('env')

    t_config = get_transformer_config(config.pretrained_cache_dir)
    config.embedding_dim = t_config.hidden_size

    # set up model
    logging.info('MODEL PARAMS:')
    logging.info(config.to_json_string())
    logging.info('END MODEL PARAMS')

    main(
        args=args,
        config=config,
        **vars(args),
    )