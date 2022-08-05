# set up logging before imports

# import sys, os
# here = os.path.dirname(__file__)
# sys.path.insert(0, here)

from models_neural.src.config_helper import training_args
from models_neural.quote_detection.utils_parser import attach_model_arguments
from models_neural.src.utils_general import format_loss_weighting, reformat_model_path
from models_neural.src.config_helper import TransformersConfig, get_transformer_config
from models_neural.quote_detection.models_discriminator_full import Discriminator
from models_neural.src.utils_oldmethods import BaselineDiscriminator, DiscriminatorGPT2Baseline
from models_neural.quote_detection.utils_dataset_discriminator import (
    NonSequentialDiscourseDataModule, SequentialDiscourseDataModule, FlatDocumentDiscourseDataModule,
    FlatDocumentDiscourseDataModuleWordSplit
)
#
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers
import torch
from transformers import AutoConfig

import logging
logging.basicConfig(level=logging.INFO)

experiments = {
    'baseline_non-sequential': (NonSequentialDiscourseDataModule, BaselineDiscriminator),
    'lstm_sequential': (SequentialDiscourseDataModule, Discriminator),
    'gpt2_sequential_baseline': (SequentialDiscourseDataModule, DiscriminatorGPT2Baseline),
    'contextual_flatmap': (FlatDocumentDiscourseDataModule, Discriminator),
    'contextual_flatmap_wordsplit': (FlatDocumentDiscourseDataModuleWordSplit, Discriminator)
}

def main(
        training_args,
        config,
        experiment='lstm_sequential',
        output_fp='.',
        num_nodes=1,
        num_gpus=1,
        notes='',
        **kwargs
    ):

    accelerator = 'dp'
    accelerator = accelerator if ((num_nodes > 1) or (num_gpus > 1)) else None

    if not os.path.exists(output_fp):
        os.makedirs(output_fp)

    datasetclass, discriminator_class = experiments[experiment]
    dataset = datasetclass(
        data_fp=config.main_data_file,
        pretrained_model_path=config.pretrained_model_path,
        num_cpus=config.num_dataloader_cpus,
        config=config,
        **kwargs
    )
    dataset.prepare_data()
    dataset.setup(stage='fit')

    # some config handling
    config.pad_id = dataset.tokenizer.pad_token_id or 0
    if config.do_multitask:
        config.num_output_tags = list(map(len, dataset.idx2class))
    else:
        config.num_output_tags = len(dataset.idx2class)
    config.num_steps_per_epoch = len(dataset.train_dataset)
    config.total_steps = training_args.num_train_epochs * config.num_steps_per_epoch
    # evaluate at the end of every epoch.
    training_args.eval_steps = config.num_steps_per_epoch

    with open('config__%s.json' % notes, 'w') as f:
        import json
        config_dict = config.to_dict()
        json.dump(config_dict, f)
        print('dumping config...')

    # initialize model
    model = discriminator_class(config=config)

    #################
    # logging/checkpoint setup
    #
    checkpoint_callback = ModelCheckpoint(
        # monitor='f1 macro',
        dirpath=output_fp,
        filename='trial-%s__epoch={epoch:02d}-f1_macro={f1 macro:.2f}' % notes,
        save_top_k=1,
        mode='max',
        auto_insert_metric_name=False
    )
    if os.environ.get('TENSORBOARD_LOGDIR'):
        tb_logger = loggers.TensorBoardLogger(
            save_dir=os.environ['TENSORBOARD_LOGDIR'],
        )
        tb_logger.log_hyperparams({
                'embedding_model_type': config.model_type,
                'lstm_bidirectional': config.bidirectional,
                'lstm_num_hidden_layers': config.num_contextual_layers,
                'contextual_layer_type': config.context_layer,
                'num_sent_attn_heads': config.num_sent_attn_heads,
                'use_doc_embs': config.use_doc_emb,
                'use_pos_embs': config.use_positional,
                'use_headline_embs': config.use_headlines,
                'dataset_size': len(dataset.train_dataset),
                'sentence_embedding_method': config.sentence_embedding_method,
                'experiment': experiment,
                # trainer params
                'warmup_steps': config.warmup_steps,
                'learning_rate': config.learning_rate,
                'gradient_accumulation': config.accumulate_grad_batches,
            }
        )
    else:
        tb_logger = None

    #################
    #  Train model
    #
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        gpus=num_gpus,
        num_nodes=num_nodes,
        accelerator=accelerator,
        max_epochs=10,
        logger=tb_logger,
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=config.max_grad_norm,
    )
    trainer.fit(model, datamodule=dataset)

    # upload best model
    best_model_path = checkpoint_callback.best_model_path
    if not args.local:
        fs = get_fs()
        fname = os.path.basename(best_model_path)
        remote_path = os.path.join('aspangher', 'source-exploration', 'models', output_fp, fname)
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
    eval_data_file = None
    if args.local:
        # train and eval files
        args.num_gpus = 0
        main_data_file = os.path.join(here, '..', args.train_data_file_s3)
        if args.eval_data_file_s3 is not None:
            eval_data_file = os.path.join(here, '..', args.eval_data_file_s3)
        pretrained_path = args.pretrained_files_s3
    else:
        from models_neural.src import (
            get_fs, download_model_files_bb, download_file_to_filepath
        )
        # train (and eval df)
        print('Downloading data...')
        fn = 'input_data.csv.gz' if args.train_data_file_s3.endswith('.gz') else 'input_data.csv'
        main_data_file = os.path.join(here, fn)
        download_file_to_filepath(remote_file_name=args.train_data_file_s3, local_path=main_data_file)
        if args.eval_data_file_s3 is not None:
            eval_data_file = os.path.join(here, 'eval_data.csv')
            download_file_to_filepath(remote_file_name=args.eval_data_file_s3, local_path=eval_data_file)

        # download model files
        print('Downloading pretrained discriminator...')
        pretrained_path = args.pretrained_files_s3
        print('downloading pretrained model %s->%s' % (args.pretrained_files_s3, pretrained_path))
        if '/' not in args.pretrained_files_s3:
            download_model_files_bb(remote_model=args.pretrained_files_s3, local_path=here)
        else:
            download_file_to_filepath(remote_file_name=args.pretrained_files_s3)

        print(glob.glob(os.path.join(pretrained_path, '*')))
        if args.finetuned_lm_file is not None:
            from models_neural.src import LMModel
            download_file_to_filepath(remote_file_name=args.finetuned_lm_file)
            config = AutoConfig.from_pretrained(pretrained_path)
            # print(config)
            fine_tuned_model = LMModel.load_from_checkpoint(
                args.finetuned_lm_file,
                loading_from_checkpoint=True,
                config=config,
                model_type=args.model_type,
            )
            if hasattr(fine_tuned_model, 'hf_model'):
                fine_tuned_model = fine_tuned_model.hf_model
            torch.save(fine_tuned_model.state_dict(), os.path.join(pretrained_path, 'pytorch_model.bin'))
            print('new directory with fine-tuned model...')
            print(glob.glob(os.path.join(pretrained_path, '*')))

    # config
    config = TransformersConfig(cmd_args=args)
    config.pretrained_model_path = reformat_model_path(pretrained_path)
    config.main_data_file = main_data_file
    config.warmup_steps = training_args.warmup_steps
    config.num_train_epochs = config.num_train_epochs if hasattr(config, 'num_train_epochs') else training_args.num_train_epochs
    config.loss_weighting = format_loss_weighting(config.loss_weighting)
    if not hasattr(config, 'env'):
        config.env = os.environ.get('env')

    t_config = get_transformer_config(config.pretrained_model_path)
    config.embedding_dim = t_config.hidden_size

    # set up model
    logging.info('MODEL PARAMS:')
    logging.info(config.to_json_string())
    logging.info('END MODEL PARAMS')

    main(
        training_args=training_args,
        config=config,
        **vars(args),
    )