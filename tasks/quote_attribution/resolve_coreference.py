from fastcoref import spacy_component
import spacy
import torch
import numpy as np
from typing import List, Dict
from scripts.coref_resolution_util import get_cluster_mappers, convert_to_sent_idx_and_sent_char_span
from copy import copy, deepcopy
import json, os
import re

CLEANR = re.compile('<.*?>')
def cleanhtml(raw_html):
    cleantext = re.sub(CLEANR, '', raw_html)
    return cleantext


def get_clusters(text, coref_approach):
    if coref_approach in ['lingmess', 'fastcoref']:
        nlp = get_nlp(coref_approach)
        doc = nlp(text)
        return doc, doc._.coref_clusters


def slim_down_s2h_mapper(s2h_mapper):
    """Remove span=>head pairs with the following logic:
    * remove (span, head) pairs that overlap and keep the longest of the two.
    * remove (span, head) pairs where `span` == `head`
    """
    s2h_to_replace = {k: v for k, v in s2h_mapper.items() if k != v}
    s2h_to_replace = {k: s2h_to_replace[k] for k in sorted(s2h_to_replace, key=lambda x: x[0])}

    # make sure we don't have nest corefs
    old_s2h_len = len(s2h_to_replace)
    while True:
        k = list(s2h_to_replace.keys())
        # compare adjacent sorted spans to make sure the start-index of one isn't less than the end-index of another
        nested_coref = list(filter(lambda x: x[1][0] < x[0][1], zip(k[:-1], k[1:])))
        to_remove = list(
            map(lambda x: min(x, key=lambda y: y[1] - y[0]), nested_coref))  # if it is, take the larger span
        s2h_to_replace = {k: v for k, v in s2h_to_replace.items() if
                          k not in to_remove}  # repeat until there are no more overlapping spans
        new_s2h_len = len(s2h_to_replace)
        if new_s2h_len == old_s2h_len:
            break
        else:
            old_s2h_len = new_s2h_len
    return s2h_to_replace


def our_core_logic(sents, s2h_to_replace, doc ):
    """Resolves the coreferences in a list of sents.

    Maybe not as complete as the AllenNLP resolution logic but also maybe more efficient."""
    # get sent charspans
    sent_lens = list(map(len, sents))
    sent_idxs = np.cumsum([0] + sent_lens)
    sent_char_spans = list(zip(sent_idxs[:-1], sent_idxs[1:]))

    test_sents = copy(sents)
    for k in reversed(list(s2h_to_replace)):
        v = s2h_to_replace[k]
        k_sent_idx, (k_sent_s, k_sent_e) = convert_to_sent_idx_and_sent_char_span(k, sent_bins=sent_char_spans)
        v_sent_idx, (v_sent_s, v_sent_e) = convert_to_sent_idx_and_sent_char_span(v, sent_bins=sent_char_spans)

        # determine what to replace
        to_replace = test_sents[v_sent_idx][v_sent_s: v_sent_e]
        final_token = doc.char_span(*k)[-1]
        if final_token.tag_ in ["PRP$", "POS"]:
            to_replace = to_replace + "'s"

        test_sents[k_sent_idx] = (
                test_sents[k_sent_idx][:k_sent_s] + to_replace + test_sents[k_sent_idx][k_sent_e:]
        )

    assert len(test_sents) == len(sents)
    return test_sents


def get_coreference_one_datum(sents: List[str], coref_approach='lingmess'):
    """
    Resolve coreferences in a single document.
        * `sents` is expected as a list of strings, where each `str` is a sentence.
        * coref_approach {'lingmess', 'fastcoref'}. Should support 'fuzzy' soon.
    """
    # get clusters
    text = ''.join(sents)
    doc, clusters = get_clusters(text, coref_approach)

    # from clusters, get cluster mappers of heads -> spans
    s2h_mapper, _ = get_cluster_mappers(doc, clusters, is_char_clusters=True, use_non_nouns_as_head=True)
    s2h_to_replace = slim_down_s2h_mapper(s2h_mapper)

    # resolve the corefs
    return our_core_logic(sents, s2h_to_replace, doc)


_nlp = None
def get_nlp(coref_approach):
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(args.spacy_model)
        add_component(_nlp, coref_approach)
    return _nlp


def add_component(model, component_type):
    if component_type == 'lingmess':
        model.add_pipe(
            "fastcoref",
            config={
                'model_architecture': 'LingMessCoref',
                'model_path': 'biu-nlp/lingmess-coref',
                'device': get_device()
            }
        )
    elif component_type == 'fastcoref':
        model.add_pipe("fastcoref")


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--spacy_model', default='en_core_web_lg')
    parser.add_argument('--coref_approach', default='lingmess')
    args = parser.parse_args()

    # relative import
    if '..' in args.input_file:
        args.input_file = os.path.join(os.path.dirname(__file__), args.input_file)

    # output file
    if args.output_file is None:
        input_fn = os.path.basename(args.input_file)
        fn, fend = input_fn.split('.')
        args.output_file = fn + '__coref-resolved.' + fend

    data_to_convert = []
    with open(args.input_file) as f:
        for line in f:
            data_to_convert.append(json.loads(line))

    coref_data = []
    for datum in data_to_convert:
        datum_copy = deepcopy(datum)
        sents = list(map(lambda x: x['sent'], datum_copy))
        sents = list(map(lambda x: cleanhtml(x.strip()) + ' ', sents))
        resolved_sents = get_coreference_one_datum(sents)
        for idx, sent in enumerate(resolved_sents):
            datum_copy[idx]['sent'] = sent
        coref_data.append(datum_copy)

    with open(args.output_file, 'w') as f:
        for datum in coref_data:
            f.write(json.dumps(datum))
            f.write('\n')
