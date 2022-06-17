# import sys, os
# here = os.path.dirname(__file__)
# sys.path.insert(0, here)

from .config_helper import training_args
from .utils_parser import attach_model_arguments
from .utils_general import get_device
from .utils_general import reformat_model_path, _get_attention_mask
from .config_helper import TransformersConfig, get_transformer_config
import torch
import logging
from .utils_model_loader import ModelLoader
from torch.nn.utils.rnn import pad_sequence


idx_to_label_high_level = {
    0: 'Cause',
    1: 'Distant',
    2: 'Error',
    3: 'Main'
}


class Scorer(ModelLoader):
    def predict(self, input_sentences, add_features=None):
        if isinstance(input_sentences, pd.Series):
            input_sentences = input_sentences.squeeze()
        # prepare input
        input_ids = list(map(self.tokenizer.encode, input_sentences))
        seq_lens = list(map(len, input_ids))
        input_ids = list(map(lambda x: torch.tensor(x, device=self.device), input_ids))
        attention_mask = _get_attention_mask(seq_lens, max_length_seq=512).to(self.device)
        input_ids = pad_sequence(input_ids, batch_first=True).to(self.device)[:, :512]
        if add_features is not None:
            add_features = torch.tensor(add_features, device=self.device)

        # run discriminator
        loss, preds, _ = self.full_discriminator.predict_one_doc(
            input_ids=input_ids, attention_mask=attention_mask,
            add_features=add_features
        )
        if preds.shape[1] == 4:
            idx_to_label = idx_to_label_high_level

        preds = torch.argmax(preds, axis=1)
        preds = preds.cpu().detach().numpy().tolist()
        if self.kwargs.get('config') and self.kwargs['config'].map_tags:
            preds = list(map(idx_to_label.get, preds))
        return preds

if __name__=="__main__":
    from util.utils_data_access import (
        download_all_necessary_files,
        download_file_to_filepath,
        download_model_files_bb,
        upload_file_to_filepath
    )
    import os, argparse, glob

    parser = argparse.ArgumentParser()
    parser = attach_model_arguments(parser)
    parser.add_argument('--local', action='store_true')

    args = parser.parse_args()
    print(args)

    if not args.local:
        download_all_necessary_files(args)

    # load data
    here = os.path.dirname(os.path.realpath(__file__))
    if args.local:
        # train and eval files
        main_data_file = os.path.join(here, '..', args.train_data_file_s3)
    else:
        # train (and eval df)
        print('Downloading data...')
        filename = 'input_data.csv' if '.gz' not in args.train_data_file_s3 else 'input_data.csv.gz'
        main_data_file = os.path.join(here, filename)
        download_file_to_filepath(remote_file_name=args.train_data_file_s3, local_path=main_data_file)

    # download model files
    if args.local:
        pretrained_path = args.pretrained_files_s3
    else:
        print('Downloading pretrained discriminator...')
        pretrained_path = args.pretrained_files_s3
        print('downloading pretrained model %s->%s' % (args.pretrained_files_s3, pretrained_path))
        if '/' not in args.pretrained_files_s3:
            download_model_files_bb(remote_model=args.pretrained_files_s3, local_path=here)
        else:
            download_file_to_filepath(remote_file_name=args.pretrained_files_s3)

    print(glob.glob(os.path.join(pretrained_path, '*')))
    # config
    config = TransformersConfig(cmd_args=args)
    config.pretrained_cache_dir = reformat_model_path(pretrained_path)
    config.main_data_file = main_data_file
    config.discriminator_path = args.discriminator_path
    config.num_warmup_steps = training_args.warmup_steps
    config.num_train_epochs = config.num_train_epochs if hasattr(config, 'num_train_epochs') else training_args.num_train_epochs
    if not hasattr(config, 'env'):
        config.env = os.environ.get('env')

    t_config = get_transformer_config(config.pretrained_cache_dir)
    config.embedding_dim = t_config.hidden_size

    # set up model
    logging.info('MODEL PARAMS:')
    logging.info(config.to_json_string())
    logging.info('END MODEL PARAMS')

    s = Scorer(
        lm_model_type=config.model_type,
        pretrained_lm_model_path=None,
        pretrained_model_path=config.pretrained_cache_dir,
        discriminator_path=config.discriminator_path,
        training_args=training_args,
        experiment=args.experiment,
        config=config,
        device=get_device()
    )

    ## edits run - change as necessary
    import pandas as pd
    df = pd.read_csv(main_data_file)
    if config.do_doc_pred:
        docs = df.assign(sentence=lambda df: df['sentences'].str.split('<SENT>'))
        docs = docs.set_index(['entry_id', 'version'])
    else:
        group_cols = ['entry_id', 'version']
        if 'source' in df.columns:
            group_cols.append('source')
        df = df.loc[lambda df: df.notnull().all(axis=1)]
        docs = df.groupby(group_cols).aggregate(list)
    from tqdm.auto import tqdm
    # file
    local_name = 'output_file.txt'
    f = open(local_name, 'w')
    # run process
    for idx in tqdm(docs.index, total=len(docs)):
        cols = docs.loc[idx]
        sents = cols['sentence']
        if args.do_version:
            if args.do_doc_pred:
                v = [idx[1]]
            else:
                v = [idx[1]] * len(sents)
        else:
            v = None
        try:
            label_col = pd.Series({}, dtype=object) if len(args.label_col) == 0 else cols[args.label_col]
            f.write(str(idx))
            f.write('\n')
            pred_tags = s.predict(input_sentences=sents, add_features=v)
            f.write(str(pred_tags))
            f.write(label_col.to_json())
            f.write('\n')
        except Exception as e:
            print(e)
            continue

    if not args.local:
        upload_file_to_filepath(local_name, args.processed_data_fname)