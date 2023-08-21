from collections import defaultdict
from spacy.tokens import Token
from more_itertools import unique_everseen
from copy import copy
import pandas as pd
from unidecode import unidecode
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from spacy.tokens import Doc, Span

import sys, os

here = os.path.dirname(__file__)
sys.path.insert(0, here)

_nlp_coref = None
def get_nlp_coref():
    global _nlp_coref
    if _nlp_coref is None:
        import neuralcoref
        import spacy
        try:
            _nlp_coref = spacy.load('en_core_web_lg')
        except:
            _nlp_coref = spacy.load('en_core_web_sm')
        neuralcoref.add_to_pipe(_nlp_coref, max_dist=500)
    return _nlp_coref


def core_logic_part(document: Doc, coref: List[int], resolved: List[str], mention_span: Span):
    final_token = document[coref[1]]
    if final_token.tag_ in ["PRP$", "POS"]:
        resolved[coref[0]] = mention_span.text + "'s" + final_token.whitespace_
    else:
        resolved[coref[0]] = mention_span.text + final_token.whitespace_
    for i in range(coref[0] + 1, coref[1] + 1):
        resolved[i] = ""
    return resolved


def get_span_noun_indices(doc: Doc, cluster: List[List[int]], char_clusters: bool = False) -> List[int]:
    """
    Gets spans that contain nouns

    * doc: a Spacy doc
    * cluster
    * char_clusters: whether to retrieve spans by char index or word index
    """
    if char_clusters:  # each (span_start, span_end) indicates char index
        spans = [doc.char_span(span[0], span[1]) for span in cluster]
    else:  # cluster (span_start, span_end) indicats words, not characters
        spans = [doc[span[0]:span[1] + 1] for span in cluster]
    #
    span_ents = [span.ents for span in spans]
    spans_pos = [[token.pos_ for token in span] for span in spans]
    span_noun_indices = [
        i for i, (span_pos, span_ents) in enumerate(zip(spans_pos, span_ents))
        if any(pos in span_pos for pos in ['NOUN', 'PROPN'])
        # and len(span_ents) > 0
    ]
    return span_noun_indices


def get_cluster_head(doc: Doc, cluster: List[List[int]],
                     is_char_cluster: bool = False,
                     use_non_nouns_as_heads: bool = True):
    """Get the head of the coreference cluster.
        Optionally, only consider heads that contain NOUN or PROPER NOUN.
    """

    # if there's a noun present in one of the spans, use that as the head
    noun_indices = get_span_noun_indices(doc, cluster, is_char_cluster)
    if (noun_indices is not None) and (len(noun_indices) > 0):
        head_idx = noun_indices[0]
    else:
        if not use_non_nouns_as_heads:
            return None, None
        head_idx = 0

    head_start, head_end = cluster[head_idx]

    # get the text span
    if is_char_cluster:
        head_span = doc.char_span(head_start, head_end)
    else:
        head_span = doc[head_start:head_end + 1]

    return head_span, [head_start, head_end]


def get_cluster_mappers(doc, clusters, is_char_clusters, use_non_nouns_as_head):
    """
    Return mappers from:
        head => [spans]
        span => head

    (my method, not part of original intersection strategy).
    """
    span_to_head_mapper = {}
    head_to_span_mapper = defaultdict(list)
    for cluster in clusters:
        _, mention = get_cluster_head(doc, cluster, is_char_clusters, use_non_nouns_as_head)
        if mention is not None: # this only happens when there's a cluster where no element is a noun
            for span in cluster:
                span_to_head_mapper[tuple(span)] = tuple(mention)
                head_to_span_mapper[tuple(mention)].append(tuple(span))
    return span_to_head_mapper, head_to_span_mapper


def is_containing_other_spans(span: List[int], all_spans: List[List[int]]):
    return any([s[0] >= span[0] and s[1] <= span[1] and s != span for s in all_spans])


def improved_replace_corefs(document, clusters):
    resolved = list(tok.text_with_ws for tok in document)
    all_spans = [span for cluster in clusters for span in cluster]  # flattened list of all spans

    for cluster in clusters:
        noun_indices = get_span_noun_indices(document, cluster)

        if noun_indices:
            mention_span, mention = get_cluster_head(document, cluster, noun_indices)

            for coref in cluster:
                if coref != mention and not is_containing_other_spans(coref, all_spans):
                    core_logic_part(document, coref, resolved, mention_span)

    return "".join(resolved)


def improved_replace_corefs_sentence_level(document, clusters, sents):
    """Resolve coreference. Taken from:

    https://github.com/NeuroSYS-pl/coreference-resolution/blob/main/improvements_to_allennlp_cr.ipynb

    Includes several improvements.

    Modified to work on the sentence level and return a list of resolved sentences.
    """
    resolved = copy(sents)
    all_spans = [span for cluster in clusters for span in cluster]  # flattened list of all spans

    for cluster in clusters:
        noun_indices = get_span_noun_indices(document, cluster)

        if noun_indices:
            mention_span, mention = get_cluster_head(document, cluster, noun_indices)
            for coref in cluster:
                if coref != mention and not is_containing_other_spans(coref, all_spans):
                    core_logic_part(document, coref, resolved, mention_span)

    return list(map(lambda x: "".join(x), resolved))



def core_logic_part_sentence_level(
        document: Doc,
        coref: Tuple[int],
        resolved: List[List[str]],
        mention_span: Span,
        sent_bins: List[Tuple[int]]
):
    """Do the actual coreference resolving.

    params:
        * document: spacy Doc
        * coref: (s, e) span for the block of text we want to remove (to replace)
        * resolved: a sequence of sentences, where each sentence is characters/whitespaces
        * mention_span: (s, e) span for the block of text we wish to replace.

    Modified to work on the sentence level and return a list of resolved sentences.
    """
    sent_idx, (sent_span_s, sent_span_e) = convert_to_sent_idx_and_sent_char_span(coref, sent_bins)
    # pack in the resolved part to the first character of the `coref` span.
    final_token = document.char_span(*coref)[-1]
    if final_token.tag_ in ["PRP$", "POS"]:
        to_replace = mention_span.text + "'s" + final_token.whitespace_
    else:
        to_replace = mention_span.text + final_token.whitespace_
    # fill in the rest with null characters
    resolved[sent_idx][sent_span_s] = to_replace
    for i in range(coref[0] + 1, coref[1] + 1):
        resolved[sent_idx][i] = ""
    return resolved


def in_bin(index, bins):
    """Check if and index is in a list of bins [(start, end)]"""
    for bin_idx, (b_s, b_e) in enumerate(bins):
        if index >= b_s and index < b_e:
            return bin_idx


def convert_to_sent_idx_and_sent_char_span(doc_span, sent_bins):
    """
    Take a span's start/end character position within in a document and return:
    1. the sentence of the span.
    2. the position of the start/end character position in the sentence.
    """
    doc_span_s, doc_span_e = doc_span
    bin_idx_of_start = in_bin(doc_span_s, sent_bins)
    bin_idx_of_end = in_bin(doc_span_e, sent_bins)

    # the start and end of the span are both in the sentence
    assert (bin_idx_of_start is not None) and (bin_idx_of_end is not None), 'couldn\'t place span in bin'
    if bin_idx_of_start != bin_idx_of_end:
        print('coref crosses a sentence boundary...')
        second_sent_start = sent_bins[bin_idx_of_end][0]
        # If the end of the span is closer to the sentence boundary than the start of it,
        # then set the sent_span to end at the end of the first sentence (i.e. the length of the sentence).
        if abs(second_sent_start - doc_span_s) > abs(doc_span_e - second_sent_start):
            bin_idx = bin_idx_of_start
            bin_s, bin_e = sent_bins[bin_idx]
            sent_span = (doc_span_s - bin_s, bin_e - bin_s)

        # If the start of the span is closer to the sentence boundary than the end of it,
        # then set the sent_span to start at the beginning of the second sentence (i.e. character 0).
        else:
            bin_idx = bin_idx_of_end
            bin_s, bin_e = sent_bins[bin_idx]
            sent_span = (0,  doc_span_e - bin_s)
    else:
        bin_idx = bin_idx_of_start
        bin_s, _ = sent_bins[bin_idx]
        sent_span = (doc_span_s - bin_s, doc_span_e - bin_s)
    return bin_idx, sent_span

class IntersectionStrategy(ABC):

    def __init__(self, allen_model, hugging_model):
        self.allen_clusters = []
        self.hugging_clusters = []
        self.allen_model = allen_model
        self.hugging_model = hugging_model
        self.document = []
        self.doc = None

    @abstractmethod
    def get_intersected_clusters(self):
        raise NotImplementedError

    @staticmethod
    def get_span_noun_indices(doc: Doc, cluster: List[List[int]]):
        spans = [doc[span[0]:span[1]+1] for span in cluster]
        spans_pos = [[token.pos_ for token in span] for span in spans]
        span_noun_indices = [i for i, span_pos in enumerate(spans_pos)
            if any(pos in span_pos for pos in ['NOUN', 'PROPN'])]
        return span_noun_indices

    @staticmethod
    def get_cluster_head(doc: Doc, cluster: List[List[int]], noun_indices: List[int]):
        head_idx = noun_indices[0]
        head_start, head_end = cluster[head_idx]
        head_span = doc[head_start:head_end+1]
        return head_span, [head_start, head_end]

    @staticmethod
    def is_containing_other_spans(span: List[int], all_spans: List[List[int]]):
        return any([s[0] >= span[0] and s[1] <= span[1] and s != span for s in all_spans])

    def coref_resolved_improved(self, doc: Doc, clusters: List[List[List[int]]]):
        resolved = [tok.text_with_ws for tok in doc]
        all_spans = [span for cluster in clusters for span in cluster]  # flattened list of all spans

        for cluster in clusters:
            noun_indices = self.get_span_noun_indices(doc, cluster)
            if noun_indices:
                mention_span, mention = self.get_cluster_head(doc, cluster, noun_indices)

                for coref in cluster:
                    if coref != mention and not self.is_containing_other_spans(coref, all_spans):
                        final_token = doc[coref[1]]
                        if final_token.tag_ in ["PRP$", "POS"]:
                            resolved[coref[0]] = mention_span.text + "'s" + final_token.whitespace_
                        else:
                            resolved[coref[0]] = mention_span.text + final_token.whitespace_

                        for i in range(coref[0] + 1, coref[1] + 1):
                            resolved[i] = ""

        return "".join(resolved)

    def clusters(self, text):
        self.acquire_models_clusters(text)
        return self.get_intersected_clusters()

    def resolve_coreferences(self, text: str):
        clusters = self.clusters(text)
        resolved_text = self.coref_resolved_improved(self.doc, clusters)
        return resolved_text

    def acquire_models_clusters(self, text: str):
        allen_prediction = self.allen_model.predict(text)
        self.allen_clusters = allen_prediction['clusters']
        self.document = allen_prediction['document']
        self.doc = self.hugging_model(text)
        hugging_clusters = self._transform_huggingface_answer_to_allen_list_of_clusters()
        self.hugging_clusters = hugging_clusters

    def _transform_huggingface_answer_to_allen_list_of_clusters(self):
        list_of_clusters = []
        for cluster in self.doc._.coref_clusters:
            list_of_clusters.append([])
            for span in cluster:
                list_of_clusters[-1].append([span[0].i, span[-1].i])
        return list_of_clusters


class PartialIntersectionStrategy(IntersectionStrategy):
    def get_intersected_clusters(self):
        intersected_clusters = []
        for allen_cluster in self.allen_clusters:
            intersected_cluster = []
            for hugging_cluster in self.hugging_clusters:
                allen_set = set(tuple([tuple(span) for span in allen_cluster]))
                hugging_set = set(tuple([tuple(span) for span in hugging_cluster]))
                intersect = sorted([list(el) for el in allen_set.intersection(hugging_set)])
                if len(intersect) > 1:
                    intersected_cluster += intersect
            if intersected_cluster:
                intersected_clusters.append(intersected_cluster)
        return intersected_clusters


class FuzzyIntersectionStrategy(PartialIntersectionStrategy):
    """ Is treated as a PartialIntersectionStrategy, yet first must map AllenNLP spans and Huggingface spans. """

    @staticmethod
    def flatten_cluster(list_of_clusters):
        return [span for cluster in list_of_clusters for span in cluster]

    def _check_whether_spans_are_within_range(self, allen_span, hugging_span):
        allen_range = range(allen_span[0], allen_span[1]+1)
        hugging_range = range(hugging_span[0], hugging_span[1]+1)
        allen_within = allen_span[0] in hugging_range and allen_span[1] in hugging_range
        hugging_within = hugging_span[0] in allen_range and hugging_span[1] in allen_range
        return allen_within or hugging_within

    def _add_span_to_list_dict(self, allen_span, hugging_span):
        if (allen_span[1] - allen_span[0] > hugging_span[1] - hugging_span[0]):
            self._add_element(allen_span, hugging_span)
        else:
            self._add_element(hugging_span, allen_span)

    def _add_element(self, key_span, val_span):
        if tuple(key_span) in self.swap_dict_list.keys():
            self.swap_dict_list[tuple(key_span)].append(tuple(val_span))
        else:
            self.swap_dict_list[tuple(key_span)] = [tuple(val_span)]

    def _filter_out_swap_dict(self):
        swap_dict = {}
        for key, vals in self.swap_dict_list.items():
            if self.swap_dict_list[key] != vals[0]:
                swap_dict[key] = sorted(vals, key=lambda x: x[1]-x[0], reverse=True)[0]
        return swap_dict

    def _swap_mapped_spans(self, list_of_clusters, model_dict):
        for cluster_idx, cluster in enumerate(list_of_clusters):
            for span_idx, span in enumerate(cluster):
                if tuple(span) in model_dict.keys():
                    list_of_clusters[cluster_idx][span_idx] = list(model_dict[tuple(span)])
        return list_of_clusters

    def get_mapped_spans_in_lists_of_clusters(self):
        self.swap_dict_list = {}
        for allen_span in self.flatten_cluster(self.allen_clusters):
            for hugging_span in self.flatten_cluster(self.hugging_clusters):
                if self._check_whether_spans_are_within_range(allen_span, hugging_span):
                    self._add_span_to_list_dict(allen_span, hugging_span)
        swap_dict = self._filter_out_swap_dict()

        allen_clusters_mapped = self._swap_mapped_spans(deepcopy(self.allen_clusters), swap_dict)
        hugging_clusters_mapped = self._swap_mapped_spans(deepcopy(self.hugging_clusters), swap_dict)
        return allen_clusters_mapped, hugging_clusters_mapped

    def get_intersected_clusters(self):
        allen_clusters_mapped, hugging_clusters_mapped = self.get_mapped_spans_in_lists_of_clusters()
        self.allen_clusters = allen_clusters_mapped
        self.hugging_clusters = hugging_clusters_mapped
        return super().get_intersected_clusters()


def get_device():
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


# model loaders
_predictor = None
def get_predictor():
    global _predictor
    if _predictor is None:
        ## allenNLP
        from allennlp.predictors.predictor import Predictor
        model_url = 'https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz'
        _predictor = Predictor.from_path(model_url)  # load the model
        device = get_device()
        _predictor._model = _predictor._model.to(device)
    return _predictor


_fuzzy = None
def get_cluster_model():
    global _fuzzy
    if _fuzzy is None:
        predictor = get_predictor()
        nlp = get_nlp_coref()
        _fuzzy = FuzzyIntersectionStrategy(predictor, nlp)
    return _fuzzy


def get_clusters(text):
    return get_cluster_model().clusters(text)


# helper functions
def get_span(k):
    if isinstance(k, Span):
        span = [k.start, k.end]
    elif isinstance(k, Token):
        span = [k.i, k.i + 1]
    return span


def fuzzy_span_match(s1, s2):
    if (s1[0] <= s2[0]) and (s1[1] >= s2[0]):
        return True
    if (s1[0] <= s2[1]) and (s1[1] >= s2[1]):
        return True
        # other direction
    if (s2[0] <= s1[0]) and (s2[1] >= s1[0]):
        return True
    if (s2[0] <= s1[1]) and (s2[1] >= s1[1]):
        return True
    return False
