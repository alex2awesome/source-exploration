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


def get_model_and_dataset_class(model_type):
    if model_type == 'sentence' :
        from sentence_model import SentenceClassificationModel as model_class
        from sentence_model import TokenizedDataset as dataset_class
    else:
        from full_sequence_model import LongRangeClassificationModel as model_class
        from full_sequence_model import TokenizedDataset as dataset_class
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
    parser.add_argument('--model_type', default='sentence', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--outfile', type=str)
    parser.add_argument('--config_name', default=None, type=str)
    parser.add_argument('--tokenizer_name', default=None, type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    args = parser.parse_args()

    # set naming
    tokenizer_name = get_tokenizer_name(args.model_name_or_path, args.tokenizer_name)
    model_class, dataset_class = get_model_and_dataset_class(args.model_type)

    # load model
    from transformers import modeling_utils
    config = AutoConfig.from_pretrained(args.config_name or args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # load in dataset
    data = get_test_data(args.dataset_name)
    dataset = dataset_class(data, tokenizer=tokenizer, do_score=True)

    device = get_device()
    model.eval()
    model = model.to(device)
    output_data = []
    for doc in tqdm(data, total=len(data)):
        input_ids, attention_mask, _ = dataset.process_one_doc(doc)
        datum = {
            'input_ids': input_ids.to(device),
            'attention_mask': attention_mask.to(device)
        }

        # score
        score = model.get_proba(**datum)
        scores = score.cpu().detach().numpy().flatten()

        # process data
        output_datum = []
        for sent_idx, sent in enumerate(doc):
            output_packet = {
                'pred': float(scores[sent_idx]),
                'sent': sent['sent']
            }
            if 'label' in sent:
                output_packet['label'] = sent['label']
            output_datum.append(output_packet)
        output_data.append(output_datum)

    #
    with open(args.outfile, 'w') as f:
        jsonlines.Writer(f).write_all(output_data)
