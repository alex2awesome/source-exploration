from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import re, json, string
import numpy as np
import unidecode
from tqdm.auto import tqdm
from transformers.generation_stopping_criteria import StoppingCriteria


sep = '\n\n##\n\n'
end = ' END'
CLEANR = re.compile('<.*?>')
def cleanhtml(raw_html):
    cleantext = re.sub(CLEANR, '', raw_html)
    return cleantext


def clean(x):
    if pd.isnull(x):
        return x
    x = x.lower()
    words_to_remove = ['the']
    for w in words_to_remove:
        x = (' %s ' % x).replace(' %s ' % w, ' ')
    x = re.sub('\s+', ' ', x)
    x = re.sub('\d+', '', x)
    for p in string.punctuation:
        x = x.replace(p, '')
    return x.strip()


def test_in(true_label, gpt3_guess):
    if pd.isnull(true_label) or pd.isnull(gpt3_guess):
        return np.nan

    true_label, gpt3_guess = clean(true_label), clean(gpt3_guess)
    if true_label == gpt3_guess:
        return True
    if true_label in gpt3_guess:
        return True
    if gpt3_guess in true_label:
        return True
    return False


def make_non_packed_prompts(one_doc_df, tokenizer):
    prompt_template = '"""%s""".\n\nTo which source can we attribute this sentence:\n\n"""%s"""\n\n##\n\n'

    output_data = []
    one_doc_df = one_doc_df.copy()
    one_doc_df['sent'] = one_doc_df['sent'].apply(cleanhtml)
    article = ' '.join(one_doc_df['sent'])
    quotes = (
        one_doc_df
            .fillna('')
            .loc[lambda df: ~df['quote_type'].isin(['', 'NARRATIVE', 'BACKGROUND'])]
            .loc[lambda df: ~(df['sent'].apply(unidecode.unidecode) == '"')]
    )

    for idx, (sent, sent_idx, head, quote_type, source_type, doc_idx) in quotes.iterrows():
        prompt = prompt_template % (article, sent)
        completion = head + end
        tokens = tokenizer.encode(prompt + completion)
        if len(tokens) < 2000:
            to_append = {
                "prompt": tokens,
                "completion": completion,
                'sent_idx': sent_idx,
                'doc_idx': doc_idx
            }
            output_data.append(to_append)

    # return
    return output_data


def load_data(filename):
    output = []
    with open(filename) as f:
        for line in f:
            json_obj = json.loads(line)
            output.append(json_obj)
    output = list(map(pd.DataFrame, output))
    return output


def get_device():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    return device


class EndTokenStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_tokens, device='cpu'):
        super().__init__()
        if isinstance(stop_tokens, list):
            stop_tokens = torch.tensor(stop_tokens)
        stop_tokens = stop_tokens.to(device)
        self.stop_tokens = stop_tokens

    def __call__(self, input_ids: torch.LongTensor, **kwargs):
        return input_ids[:, :-len(self.stop_tokens)] == self.stop_tokens

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--model_name_or_path', type=str, default=None)
    parser.add_argument('--tokenizer_name', type=str, default=None)

    # dataset arguments
    parser.add_argument('--dataset_name', type=str, help='jsonl file of original docs.', default=None)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--end', type=str, default=None)

    args = parser.parse_args()

    generation_config = {
        'do_sample': True,
        'top_k': 10,
        'max_length': 2048,
        'max_new_tokens': 4,
        'top_p': .95,
        'num_return_sequences': 2
    }

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    device = get_device()
    model = model.to(device)

    test_doc_dfs = load_data(args.dataset_name)

    # Generate Text
    model.eval()
    test_sample_with_answers = []
    stopping_criteria = StoppingCriteria()
    for one_doc_df in tqdm(test_doc_dfs, total=len(test_doc_dfs)):
        prompts = make_non_packed_prompts(one_doc_df, tokenizer)

        # get the model output
        for p in prompts:
            prompt_tokens = torch.tensor(p["prompt"], device=device).unsqueeze(0)
            model_output = model.generate(prompt_tokens, **generation_config)
            test_sample_with_answers.append({
                'y_pred': model_output,
                'y_true': p['completion'],
                'sent_idx': p['sent_idx'],
                'doc_idx': p['doc_idx']
            })

    # pack up and write to disk
    test_answer_df = pd.DataFrame(test_sample_with_answers).drop_duplicates(['doc_idx', 'sent_idx'])
    all_docs_df = pd.concat(test_doc_dfs)
    test_answer_df_w_doc = (
        all_docs_df
            .loc[lambda df: df['doc_id'].isin(test_answer_df['doc_idx'])]
            .merge(test_answer_df, how='left', left_on=['doc_id', 'sent_idx'], right_on=['doc_idx', 'sent_idx'])
            .drop(['doc_idx', ], axis=1)
            .assign(y_true=lambda df: df['y_true'].str.replace(' END', ''))
            .assign(match=lambda df: df.apply(lambda x: test_in(x['y_pred'], x['y_true']), axis=1))
            [['doc_id', 'sent_idx', 'sent', 'quote_type', 'head', 'y_pred', 'match']]
            .assign(sent=lambda df: df['sent'].apply(unidecode.unidecode).apply(cleanhtml))
    )

    test_answer_df_w_doc.to_csv(args.output_file)