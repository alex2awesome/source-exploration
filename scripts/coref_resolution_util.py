from collections import defaultdict
from spacy.tokens import Token, Span
from more_itertools import unique_everseen
from copy import deepcopy, copy
import pandas as pd
from unidecode import unidecode
from copy import deepcopy
import neuralcoref
from abc import ABC, abstractmethod
from os import environ
from warnings import warn
from typing import Dict, List
from spacy.tokens import Doc, Span

def core_logic_part(document: Doc, coref: List[int], resolved: List[str], mention_span: Span):
    final_token = document[coref[1]]
    if final_token.tag_ in ["PRP$", "POS"]:
        resolved[coref[0]] = mention_span.text + "'s" + final_token.whitespace_
    else:
        resolved[coref[0]] = mention_span.text + final_token.whitespace_
    for i in range(coref[0] + 1, coref[1] + 1):
        resolved[i] = ""
    return resolved

def get_span_noun_indices(doc: Doc, cluster: List[List[int]]) -> List[int]:
    spans = [doc[span[0]:span[1]+1] for span in cluster]
    span_ents = [span.ents for span in spans ]
    spans_pos = [[token.pos_ for token in span] for span in spans]
    span_noun_indices = [
        i for i, (span_pos, span_ents) in enumerate(zip(spans_pos, span_ents))
        if any(pos in span_pos for pos in ['NOUN', 'PROPN'])
#         and len(span_ents) > 0
    ]
    return span_noun_indices

def get_cluster_head(doc: Doc, cluster: List[List[int]], noun_indices: List[int]):
    head_idx = noun_indices[0]
    head_start, head_end = cluster[head_idx]
    head_span = doc[head_start:head_end+1]
    return head_span, [head_start, head_end]

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
        if (allen_span[1]-allen_span[0] > hugging_span[1]-hugging_span[0]):
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

### model loaders
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


_nlp = None
def get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load('en_core_web_lg')
        except:
            _nlp = spacy.load('en_core_web_sm')
    return _nlp


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

### helper functions
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


def flatten_list_of_lists(l_of_l):
    return [i for s in l_of_l for i in s]


def get_sent_idx_from_sent(sent, doc=None, sents=None):
    if sents is None:
        sents = doc.sents
    s = list(filter(lambda x: sent == x[1], enumerate(sents)))
    return s[0][0]


def get_cluster_mappers(doc, clusters):
    span_to_head_mapper = {}
    head_to_span_mapper = defaultdict(list)
    for cluster in clusters:
        noun_indices = get_span_noun_indices(doc, cluster)

        if noun_indices:
            mention_span, mention = get_cluster_head(doc, cluster, noun_indices)
            for span in cluster:
                span_to_head_mapper[tuple(span)] = tuple(mention)
                head_to_span_mapper[tuple(mention)].append(tuple(span))
    return span_to_head_mapper, head_to_span_mapper


# 1. extract all named entities
# 2. fuzzy match named entities with coreference cluster heads
# 3. cluster named entities by lexical overlap, extract NE head to use as singular head
# 4. merge all coreference/cluster identities
# 5. extract all quoted entites and merge these with any member of the NEs/coreference cluster

###########################################
## extract quotes
##################
def extract_quotes_from_nsubj(doc, return_dict=False):
    entities = defaultdict(lambda: {'background sentence': [], 'quote sentence': []})
    spans = []
    ## get quotes
    to_break = False
    for s_idx, sent in enumerate(doc.sents):
        ##
        text_sentence = ' '.join([word.text for word in sent]).strip()

        # hack to pick up common phrasal signifiers
        common_multiword_phrases = ['according to', 'pointed out', 'points out', 'called for', 'calls for']
        for c in common_multiword_phrases:
            if c in text_sentence:
                sent = get_nlp()(text_sentence.replace(c, 'said'))

        ## extract all nsubj of VERB if VERB is 'said', 'says' or 'say'
        for possible_subject in sent:
            if (
                    possible_subject.dep_ == 'nsubj' and
                    possible_subject.head.pos_ == 'VERB' and
                    possible_subject.head.text in (
                    'say', 'says', 'said',
                    'describe', 'describes', 'described',
                    'claims', 'claims', 'claimed',
                    'explained', 'explains', 'explain',
                    'mentioned', 'mentions', 'mention',
                    'articulated', 'articulates', 'articulate',
                    'called', 'calls', 'call',
                    'declared', 'declares', 'declare',
                    'worried', 'worries', 'worry',
                    'asserted', 'asserts', 'assert',
            )
            ):
                # map subject to a noun phrase or named entity
                found = False
                if possible_subject.text.lower() in ['he', 'she', 'they']:
                    subj_to_use = possible_subject
                    found = True

                if not found:
                    for named_ent in sent.ents:
                        if named_ent.label_ == 'PERSON':
                            if possible_subject in named_ent:
                                found = True
                                subj_to_use = named_ent

                if not found:
                    for noun_phrase in sent.noun_chunks:
                        if possible_subject in noun_phrase:
                            subj_to_use = noun_phrase
                            found = True

                if not found:
                    subj_to_use = possible_subject

                subj_span = tuple(get_span(subj_to_use))
                spans.append(subj_span)
                entities[subj_span]['quote sentence'].append((s_idx, sent))
                entities[subj_span]['span_text'] = subj_to_use

    if return_dict:
        return entities
    else:
        return spans

def get_adjacent_quotes(already_extracted_quote_chunks, doc):
    # extract all quote sentences
    in_quote = False
    word_idx_in_quote = []
    for word_idx, word in enumerate(doc):
        if unidecode(word.text) == '"':
            in_quote = bool((in_quote + 1) % 2)
        if in_quote:
            word_idx_in_quote.append(word_idx)

    quote_sent = list(map(lambda x: doc[x].sent, word_idx_in_quote))
    quote_sent = list(unique_everseen(quote_sent))
    quote_sent_idxs = list(map(lambda x: get_sent_idx_from_sent(x, doc), quote_sent))

    #
    new_quote_sent_chunks = deepcopy(already_extracted_quote_chunks)
    if not isinstance(new_quote_sent_chunks, list):
        new_quote_sent_chunks = new_quote_sent_chunks.tolist()
    seen_set = flatten_list_of_lists(new_quote_sent_chunks)
    seen_set = set(map(lambda x: x[0], seen_set))
    adjacent_candidates = list(set(quote_sent_idxs) - seen_set)

    for i in adjacent_candidates:
        for q_idx, q in enumerate(new_quote_sent_chunks):
            for (q_sent_idx, q_span) in q:
                if i == q_sent_idx + 1:
                    new_quote_sent_chunks[q_idx].append((i, copy(q_span)))

    new_quote_sent_chunks = list(map(lambda x: list(set(x)), new_quote_sent_chunks))
    return new_quote_sent_chunks

## helper functions
def format_span_with_word_list(adj, sent, span, span_color=None, bold=False):
    span_s, span_e = span[0] - adj, span[1] - adj
    if (span_s > 0) and (span_e > 0) and (span_s < len(sent)) and (span_e <= len(sent)):
        span_text = sent[span_s:span_e]
        span_str = ' '.join(span_text)
        if span_color is not None:
            sent = (
                    sent[:span_s] +
                    ['<span style="background-color: %s">%s</span>' % (span_color, span_str)] +
                    sent[span_e:]
            )
        if bold:
            sent = (
                    sent[:span_s] +
                    ['<span style="font-weight: bold">%s</span>' % span_str] +
                    sent[span_e:]
            )
        adj = adj - (len(span_text) - 1)
    return sent, adj


def format_sent_with_word_list(sent, sent_color):
    if isinstance(sent, list):
        sent = ' '.join(sent)
    return '<span style="background-color: %s">%s</span>' % (sent_color, sent)


import util
def perform_quote_extraction_and_clustering(text):
    # 1. Cluster coref
    fuzzy = get_cluster_model()
    c = fuzzy.clusters(text)
    doc = fuzzy.doc if fuzzy.doc is not None else get_nlp()(text)

    # 2. Get all named entities
    person_ents = []
    for s_idx, sent in enumerate(doc.sents):
        # get person-entities
        for ent in sent.ents:
            if ent.label_ == 'PERSON':
                person_ents.append({
                    's_idx': s_idx,
                    'ent': ent.text,
                    'sent': sent.text,
                    'span': tuple(get_span(ent))
                })
    person_ents_df = pd.DataFrame(person_ents)

    # 3. Determine cluster heads
    span_to_head_mapper, head_to_span_mapper = get_cluster_mappers(doc, c)
    head_to_span_mapper_spacy_corr = {
        (k[0], k[1] + 1): list(map(lambda x: (x[0], x[1] + 1), v)) for k, v in head_to_span_mapper.items()
    }

    head_keys = list(head_to_span_mapper_spacy_corr.keys())

    # 4. Cluster NE's based on string matching
    name_c = util.cluster_by_last_name_equality(person_ents_df['ent'])
    _, h_to_c, _, c_to_h = util.get_name_cluster_head_by_length(name_c)

    # 5. Identify which NSUBJs are quotes
    q = extract_quotes_from_nsubj(doc, return_dict=True)
    quote_spans = list(q.keys())

    # 6. Match them using NE clusters and CoRef clusters
    t = (
        person_ents_df
            .assign(coref_heads=lambda df:
        df['span'].apply(lambda x: list(filter(lambda y: fuzzy_span_match(x, y), head_keys)))
                    )
            .assign(head_ref=pd.Series(c_to_h).sort_index())
            .assign(head_span=lambda df: df.apply(lambda x: df.loc[x['head_ref']]['span'], axis=1))
            .assign(head_s_idx=lambda df: df.apply(lambda x: df.loc[x['head_ref']]['s_idx'], axis=1))
            .assign(all_corefs=lambda df:
        df['coref_heads']
                    .apply(lambda x: list(map(lambda y: head_to_span_mapper_spacy_corr[y], x)))
                    .apply(lambda x: flatten_list_of_lists(x))
                    )
            .groupby('head_ref')
            .aggregate(list)
            .drop('coref_heads', axis=1)
            .assign(all_corefs=lambda df: df['all_corefs'].apply(flatten_list_of_lists))
            .rename(columns={
            's_idx': 'ne_sent_idxs',
            'sent': 'ne_sent',
            'span': 'ne_span',
            'all_corefs': 'coref_span',
            'ent': 'ne_ent'
        })
            .assign(coref_sent_idxs=lambda df: df['coref_span'].apply(
            lambda x: list(set(map(lambda y: get_sent_idx_from_sent(doc[y[0]:y[1]].sent, doc), x))))
                    )
            .assign(head_span=lambda df: df['head_span'].str.get(0))
            .assign(head_s_idx=lambda df: df['head_s_idx'].str.get(0))
            # get sentences/entities that are quotes
            .assign(
            quote_span=lambda df: df
                .apply(lambda x: list(set(x['ne_span'] + x['coref_span'])), axis=1)
                .apply(lambda x: flatten_list_of_lists(
                list(map(lambda y: list(filter(lambda z: fuzzy_span_match(y, z), quote_spans)), x))))
        )
            .assign(quote_sent_idxs=lambda df: df['quote_span'].apply(
            lambda x: list(map(lambda y: get_sent_idx_from_sent(doc[y[0]:y[1]].sent, doc), x)))
                    )
            # chunk the quotes
            .assign(
            quote_chunks=lambda df: df.apply(lambda x: list(zip(x['quote_sent_idxs'], x['quote_span'])), axis=1))
            .assign(quote_chunks=lambda df: get_adjacent_quotes(df['quote_chunks'], doc))
    )

    sentences_with_quotes = (t['quote_chunks']
                             .apply(lambda x: list(map(lambda y: y[0], x)))
                             .pipe(lambda df: set(flatten_list_of_lists(df)))
                             )

    t = (t
         .assign(ne_chunks=lambda df: df.apply(lambda x: list(zip(x['ne_sent_idxs'], x['ne_span'])), axis=1))
         .assign(coref_chunks=lambda df: df.apply(lambda x: list(zip(x['coref_sent_idxs'], x['coref_span'])), axis=1))
         .assign(background_chunks=lambda df:
    df.apply(
        lambda x: list(filter(lambda x: x[0] not in sentences_with_quotes, set(x['ne_chunks'] + x['coref_chunks']))),
        axis=1)
                 )
         )[['head_span', 'head_s_idx', 'quote_chunks', 'background_chunks']]

    sent_words = list(map(lambda s: list(map(lambda w: w.text, s)), doc.sents))
    sent_lens = list(map(len, sent_words))
    return t, sent_words, sent_lens


# span_to_head_mapper_spacy_corr = {
#     (k[0], k[1] + 1): (v[0], v[1] + 1) for k, v in span_to_head_mapper.items()
# }
# span_to_head_mapper_strs = {doc[k[0]:k[1] + 1]: doc[v[0]:v[1] + 1] for k, v in span_to_head_mapper.items()}
