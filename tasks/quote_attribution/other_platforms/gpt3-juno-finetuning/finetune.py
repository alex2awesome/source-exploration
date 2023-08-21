import os
import time
import datetime

import pandas as pd
import seaborn as sns
import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler

torch.manual_seed(42)

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

learning_rate = 5e-4
warmup_steps = 1e2
epsilon = 1e-8


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
        return self.input_ids[idx], self.labels[idx]


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


def set_seeds(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device('cpu')


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

def train_model(m, train_dat, val_dat, num_epochs):
    device = get_device()
    m = m.to(device)
    optimizer = AdamW(m.parameters(), lr=learning_rate, eps=epsilon)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    training_stats = []
    for epoch_i in range(0, num_epochs):

        # ========================================
        #               Training
        # ========================================
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, num_epochs))
        print('Training...')

        t0 = time.time()
        total_train_loss = 0
        m.train()

        for step, batch in tqdm(enumerate(train_dat)):
            b_input_ids = batch[0].to(device)
            b_labels = batch[1].to(device)

            m.zero_grad()

            outputs = m(b_input_ids, labels=b_labels)
            loss = outputs[0]
            batch_loss = loss.item()
            total_train_loss += batch_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            del b_input_ids
            del b_labels
            if device == torch.device('cuda'):
                torch.cuda.empty_cache()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================
        print("")
        print("Running Validation...")

        t0 = time.time()
        m.eval()
        total_eval_loss = 0

        # Evaluate data for one epoch
        for batch in tqdm(val_dat):
            b_input_ids = batch[0].to(device)
            b_labels = batch[1].to(device)

            with torch.no_grad():
                outputs = m(b_input_ids, labels=b_labels)

            loss = outputs[0]

            del b_input_ids
            del b_labels
            if device == torch.device('cuda'):
                torch.cuda.empty_cache()

        batch_loss = loss.item()
        total_eval_loss += batch_loss

    avg_val_loss = total_eval_loss / len(validation_dataloader)

    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append({
        'epoch': epoch_i + 1,
        'Training Loss': avg_train_loss,
        'Valid. Loss': avg_val_loss,
        'Training Time': training_time,
        'Validation Time': validation_time
    })

    return m, training_stats


if __name__ == '__main__':
    ACCEPTABLE_MODEL_NAME_LIST = [
        'gpt2',
        'gpt2-large',
        'EleutherAI/gpt-neo-2.7B',
        'EleutherAI/gpt-j-6B'
    ]


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data_file')
    parser.add_argument('--pretrained_model_name', default='gpt2-large')
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--output_dir', default='./model_save/')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--platform', type=str, default=None)
    args = parser.parse_args()

    # Load the models
    cache_dir = None
    if args.platform == 'gcp':
        cache_dir = '/dev'
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name, cache_dir=cache_dir)
    configuration = AutoConfig.from_pretrained(args.pretrained_model_name, output_hidden_states=False, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name, config=configuration, cache_dir=cache_dir)

    # Load the data
    train_input, val_input = load_data(args.training_data_file)

    train_dataset = TokenizedDataset(train_input, tokenizer, max_length=args.max_length)
    val_dataset = TokenizedDataset(val_input, tokenizer, max_length=args.max_length)
    print('{:>5,} training samples'.format(len(train_dataset)))
    print('{:>5,} validation samples'.format(len(val_dataset)))
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=args.batch_size  # Trains with this batch size.
    )
    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=args.batch_size  # Evaluate with this batch size.
    )

    # train model
    model, training_stats = train_model(model, train_dataloader, validation_dataloader, args.epochs)

    # Create output directory if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("Saving model to %s" % args.output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))