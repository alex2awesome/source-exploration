from transformers import AutoConfig, AutoTokenizer
import torch
import sys
import jsonlines
from tqdm.auto import tqdm
sys.path.insert(0, '.')
from copy import copy


def get_tokenizer_name(model_name, tokenizer_name):
    if tokenizer_name is not None:
        return tokenizer_name
    if 'roberta-base' in model_name:
        return 'roberta-base'
    elif 'roberta-large' in model_name:
        return 'roberta-large'


def get_model_and_dataset_class(model_name):
    if 'salience' in model_name:
        from qa_model import QAModelWithSalience as model_class
        from qa_dataset import QATokenizedDataset as dataset_class
    else:
        from qa_model import QAModel as model_class
        from qa_dataset import QATokenizedDataset as dataset_class
    return model_class, dataset_class


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def get_test_data(datafile):
    data = list(jsonlines.open(datafile))
    if isinstance(data[0], dict) and ('data' in data[0]):
        test_data = list(filter(lambda x: x['split'] == 'test', data))
        return list(map(lambda x: x['data'], test_data))
    else:
        return data

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--outfile', type=str)
    parser.add_argument('--config_name', default=None, type=str)
    parser.add_argument('--tokenizer_name', default=None, type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--platform', default=None)
    args = parser.parse_args()

    # set naming
    tokenizer_name = get_tokenizer_name(args.model_name_or_path, args.tokenizer_name)
    model_class, dataset_class = get_model_and_dataset_class(args.model_name_or_path)

    # load model
    from transformers import modeling_utils
    config = AutoConfig.from_pretrained(args.config_name or args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # load in dataset
    data = get_test_data(args.dataset_name)
    dataset = dataset_class(data, hf_tokenizer=tokenizer)

    device = get_device()
    model.eval()
    model = model.to(device)
    output_data = []
    for idx in tqdm(range(len(dataset)), total=len(dataset)):
        datum = dataset[idx]
        datum = {k: torch.tensor([v]).to(device) for k, v in datum.items()}

        # score
        output = model(**datum)
        if 'start_positions' in datum:
            _, start_logits, end_logits = output
        else:
            start_logits, end_logits = output[0]
        start_pred, end_pred = start_logits.argmax(axis=1)[0], end_logits.argmax(axis=1)[0]
        start_pred, end_pred = min(start_pred, end_pred), max(start_pred, end_pred)

        span_window = datum['input_ids'][0][start_pred - 2: end_pred + 3].cpu().detach().numpy()
        pred_text_window = tokenizer.decode(span_window)
        #
        span = datum['input_ids'][0][start_pred: end_pred + 1].cpu().detach().numpy()
        pred_text = tokenizer.decode(span)

        # process data
        output_packet = {'pred_text_window': pred_text_window, 'pred_text': pred_text}
        if 'start_positions' in datum:
            s = datum['start_positions']
            e = datum['end_positions']
            ids = datum['input_ids']
            true_head = tokenizer.decode(ids[0, s: e+1])
            output_packet['label'] = true_head
        output_data.append(output_packet)

    #
    with open(args.outfile, 'w') as f:
        jsonlines.Writer(f).write_all(output_data)