from fine_tuning.language_models import LMModel, GPT2Wrapper, GPT2LMHeadModel, RobertaForMaskedLM
from pytorch_lightning.callbacks import ModelCheckpoint
import glob
import torch
import torch.optim
import torch.utils.data as data
import pytorch_lightning as pl
import os

MAX_TOKENS = 2045
learning_rate = 5e-4
warmup_steps = 1e2
epsilon = 1e-8


def load_data(file):
    import json
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


def transpose_dict(dicts):
    """Take a dictionary in record-format and translate it into a columnar dict.

    [{'a': 1, 'b':2}, {'a':2, 'b':3}] -> {'a': [1,2], 'b': [2, 3]}
    """
    columns = {}
    for key in dicts[0].keys():
        columns[key] = list(map(lambda d: d[key], dicts))
    return columns


class Dataset(data.Dataset):
    def __init__(self, X, y=None, split=None):
        """Reads source and target sequences from txt files."""
        self.X = X
        self.y = y
        self.split = split

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        data = {}
        data["input_ids"] = self.X[index]
        data['labels'] = self.y[index]
        return data


class FinetuningDataModule(pl.LightningDataModule):
    def __init__(self, data_fp, tokenizer, max_length=2048, batch_size=1, num_cpus=1):
        super().__init__()
        self.data_fp = data_fp
        self.max_length = max_length
        self.num_cpus = num_cpus
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def prepare_data(self):
        if not os.path.exists(self.data_fp):
            raise FileNotFoundError('Data files... make sure to download them from S3!')

    def setup(self, stage=None):
        if stage in ('fit', None):
            train_input, val_input = load_data(self.data_fp)
            self.train_dataset = self.get_dataset(train_input)
            self.test_dataset = self.get_dataset(val_input)

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

    def get_dataset(self, data_input):
        """
        Read in dataset as a list of "label \t text" entries.
        Output flat lists of X = [sent_1, sent_2, ...], y = [label_1, label_2]
        """
        X, y = [], []
        for dat in data_input:
            prompt_toks = self.tokenizer.encode(dat['prompt'])
            completion_toks = self.tokenizer.encode(dat['completion'])
            input_ids = prompt_toks + completion_toks
            labels = [-100] * len(prompt_toks) + completion_toks
            if len(input_ids) <= self.max_length:
                X.append(torch.tensor(input_ids))
                y.append(torch.tensor(labels))

        return Dataset(X, y)

    def collate_fn(self, dataset):
        """
        Takes in an instance of Torch Dataset (or a subclassed instance).
        Expects dataset[i]['X'] to be a list of list of sentences
            and dataset[i]['y'] to be a list of list of labels.
        Returns dict with key:
             input_ids: list of tensors of len # docs where each tensor is an entire document.
             labels: same as `input_ids`, for language modeling.
        """
        columns = transpose_dict(dataset)
        X_batch = list(map(lambda sents: torch.cat(sents), columns["input_ids"]))
        y_batch = list(map(lambda sents: torch.cat(sents), columns['labels']))
        return {
            "input_ids": torch.cat(X_batch).unsqueeze(dim=0),
            'labels': torch.cat(y_batch).unsqueeze(dim=0)
        }



local_output_fp = './runs/'

def main(args, output_fp='.'):
    accelerator = 'dp'
    accelerator = accelerator if ((args.num_nodes > 1) or (args.num_gpus > 1)) else None

    if not os.path.exists(output_fp):
        os.makedirs(output_fp)

    #########
    # load dataset and model classes
    dataset = FinetuningDataModule(
        data_fp=args.dataset,
        num_cpus=args.num_dataloader_cpus,
        batch_size=args.batch_size,
        max_length=args.max_length_seq,
    )
    dataset.prepare_data()
    dataset.setup(stage='fit')
    config.num_steps_per_epoch = len(dataset.train_dataset)

    if lm_type == 'gpt2':
        model = GPT2LMHeadModel.from_pretrained(config.pretrained_cache_dir, config=config)
    else:
        print(config)
        model = RobertaForMaskedLM.from_pretrained(config.pretrained_cache_dir, config=config)
    #
    model.resize_token_embeddings(len(dataset.tokenizer))
    lm = lm_class(config=config, model=model)  # our experimental setup


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
        gpus=args.num_gpus,
        num_nodes=args.num_nodes,
        accelerator=accelerator if not args.use_deepspeed else None,
        max_epochs=10,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.max_grad_norm,
        # plugins='ddp_sharded',
        plugins='deepspeed_stage_2' if args.use_deepspeed else None,
        precision=16 if args.use_deepspeed else 32
    )

    print('NUM GPUs USING:')
    print(trainer.gpus)
    # if args.num_gpus > 1:
    #     lm.parallelize()

    trainer.fit(lm, datamodule=dataset)

    # cache
    # upload best model
    best_model_path = checkpoint_callback.best_model_path
    if args.env == 'bb':
        fname = os.path.basename(best_model_path)

    # log best metric score
    best_metric = checkpoint_callback.best_model_score
    print('BEST MODEL SCORE: %s' % best_metric)


if __name__ == '__main__':
    import argparse
    from fine_tuning.utils_parser import attach_model_args
    parser = argparse.ArgumentParser()
    parser = attach_model_args(parser)
    args = parser.parse_args()

    # load data
    here = os.path.dirname(os.path.realpath(__file__))
    if args.env == 'local':
        # train and eval files
        args.dataset = os.path.join(here, '..', args.dataset)
    else:
        # train (and eval df)
        print('Downloading data...')
        data_fp = os.path.join(here, 'input_data.csv')
        download_file_to_filepath(remote_file_name=args.dataset, local_path=data_fp)
        args.dataset = data_fp

    # download model files
    if args.env == 'local':
        pretrained_path = args.pretrained_model_path
    else:
        if '/' not in args.pretrained_model_path:
            download_model_files_bb(remote_model=args.pretrained_model_path, local_path=here)
        else:
            download_file_to_filepath(remote_file_name=args.pretrained_model_path)
        output_path = os.path.join(here, args.pretrained_model_path, '*')
        print('files in: %s' % output_path)
        print(glob.glob(output_path))
        args.pretrained_path = reformat_model_path(os.path.join(here, args.pretrained_model_path), args)

    if args.experiment is None:
        if 'gpt2' in args.pretrained_model_path:
            args.experiment = 'discourse-gpt2'
        elif 'roberta' in args.pretrained_model_path:
            args.experiment = 'discourse-roberta'
        else:
            print('No args.experiment set, or can\'t infer it from args.pretrained_model_path!!!')

    # run fine-tuning
    main(args)
