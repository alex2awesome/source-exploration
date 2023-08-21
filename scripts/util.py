import spacy
import os
from collections import defaultdict
import jellyfish
from spacy import displacy
from more_itertools import unique_everseen
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
import tqdm
import re
import unidecode
import spacy
import sys

here = os.path.dirname(__file__)
sys.path.insert(0, here)
try:
    from .params import (
        orig_speaking_lexicon,
        full_speaking_lexicon,
        orig_ner_list,
        full_ner_list,
        multiword_phrases_present_tense,
        multiword_phrases_past_tense,
        desired_checklist_of_anonymous_sources,
        desired_checklist_of_documents
    )
except:
    from params import (
        orig_speaking_lexicon,
        full_speaking_lexicon,
        orig_ner_list,
        full_ner_list,
        multiword_phrases_present_tense,
        multiword_phrases_past_tense,
        desired_checklist_of_anonymous_sources,
        desired_checklist_of_documents
    )


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

def clean_multiword_phrases(sent, text_sentence=None):
    # hack to pick up common phrasal signifiers
    for c in multiword_phrases_past_tense:
        if c in text_sentence:
            sent = get_nlp()(text_sentence.replace(c, 'said'))

    for c in multiword_phrases_present_tense:
        if c in text_sentence:
            sent = get_nlp()(text_sentence.replace(c, 'says'))

    return sent, text_sentence


def multiprocess(input_list, func, max_workers=-1):
    """Simple ThreadPoolExecutor Wrapper.

        Input_list: list to be mapped over by the executor.
        func: function to be mapped.

        returns output of the function over the list.

        Code:
            with ThreadPoolExecutor(max_workers=2) as executor:
                for output in executor.map(func, input_list):
                    yield output
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for output in executor.map(func, input_list):
            yield output


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

######## project specific utils
def remove_dup_sentences(text, nlp):
    """Remove duplicate and short sentences."""
    doc = nlp(text)
    unique_sentences = list(unique_everseen(map(lambda x: x.text.strip(), doc.sents)))
    text = ' '.join(unique_sentences)
    return text

def clean(doc):
    doc = unidecode.unidecode(doc)
    # doc = doc.replace('\n', '')
    
    ## replace contractions  
    for to_replace, to_substitute_in in contractions.items():
        doc = doc.replace(to_replace, to_substitute_in)
    
    ## fix spacing 
    doc = doc.replace('`', '\'').replace('LEAD:', '')
    doc = re.sub("\'\s{0,2}\'", "''",  doc)
    doc = doc.replace('\'\'\'', '"').replace('\'\'', '"').replace('.', '. ')
    
    ## fix quote issues 
    doc = re.sub('\s*"\s*', '"', doc)

    for quote_char in ['\'', '"']:
        quote_matches = list(re.finditer(quote_char, doc))
        char_list = list(doc)
        for start_quote, end_quote in list(zip(quote_matches[0::2], quote_matches[1::2])):
            char_list[start_quote.span()[0]] = ' %s' % quote_char
            char_list[end_quote.span()[0]] = '%s ' % quote_char

    doc = ''.join(char_list)
    doc = re.sub('\.\s+', '. ', doc) 
    ## remove duplicate sentences
#     doc = util.remove_dup_sentences(doc, nlp)

    return doc

def is_background_or_quote(text_sentence, speaking_lexicon):
    background_or_quote = 'background sentence'
    if any(list(map(lambda sig: ' %s ' % sig in text_sentence, speaking_lexicon))):
        background_or_quote = 'quote sentence'
    return background_or_quote

def get_quotes_method_1(doc, orig_speaking=True, orig_ner=True, find_anon=True, find_docs=True, return_sents=False):
    if isinstance(doc, list):
        sents = doc
    else:
        sents = doc.sents

    ## extract quotes
    speaking_lexicon = orig_speaking_lexicon if orig_speaking else full_speaking_lexicon
    ner_list = orig_ner_list if orig_ner else full_ner_list
    extra_source_list = (desired_checklist_of_anonymous_sources if find_anon else []) + (desired_checklist_of_documents if find_docs else [])

    entities = defaultdict(lambda: {'background sentence': [], 'quote sentence': []})
    output_sents = []
    for s_idx, sent in enumerate(sents):
        if isinstance(sent, str):
            sent = get_nlp()(sent)

        ## 
        text_sentence = ' '.join([word.text for word in sent]).strip()
        sent, text_sentence = clean_multiword_phrases(sent, text_sentence)

        sources_in_sent = []
        # get person-entities
        for ent in sent.ents:
            if ent.label_ in ner_list:
                background_or_quote = is_background_or_quote(text_sentence, speaking_lexicon)
                entities[ent.text][background_or_quote].append((s_idx, text_sentence))
                sources_in_sent.append({
                    'head': ent.text,
                    'quote_type': background_or_quote
                })

        for anon_source_sig in extra_source_list:
            if anon_source_sig in text_sentence:
                background_or_quote = is_background_or_quote(text_sentence, speaking_lexicon)
                entities[anon_source_sig][background_or_quote].append((s_idx, text_sentence))
                sources_in_sent.append({
                    'head': anon_source_sig,
                    'quote_type': background_or_quote
                })
        output_sents.append({
            'sent': sent,
            'sources': sources_in_sent
        })

    if return_sents:
        return entities, output_sents
    else:
        return entities


_nlp = None
def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load('en_core_web_lg')
    return _nlp


_nlp_coref = None
_greedyness = None
_max_dist = None
def get_coref_nlp(greedyness=0.5, max_dist=50):
    global _nlp_coref
    global _greedyness
    global _max_dist
    if (_nlp_coref is None) or (greedyness != _greedyness) or (max_dist != _max_dist):
        import neuralcoref
        _nlp_coref = spacy.load('en_core_web_lg')
        neuralcoref.add_to_pipe(_nlp_coref, greedyness=greedyness, max_dist=max_dist)
    return _nlp_coref


def get_quotes_method_2(
        doc=None, text=None, cluster=True,
        resolve_coref=False, dedupe_sents=False,
        greedyness=0.5, max_dist=50, orig_ner=True, orig_speaking=True,
        return_sents=False,
):
    if isinstance(doc, list):
        sents = doc
    else:
        sents = doc.sents

    """Get quoted people by finding the nsubj of a 'say', 'said' or 'according to' verb."""
    ner_list = orig_ner_list if orig_ner else full_ner_list
    speaking_lexicon = orig_speaking_lexicon if orig_speaking else full_speaking_lexicon

    if text is not None and doc is None:
        if resolve_coref:
            doc = get_coref_nlp(greedyness=greedyness, max_dist=max_dist)(text)
            text = doc._.coref_resolved
        doc = get_nlp()(text)

    ## extract quotes
    entities = defaultdict(lambda: {'background sentence': [], 'quote sentence': []})
    seen = set()
    ## get quotes
    output_sents = []
    for s_idx, sent in enumerate(sents):
        if isinstance(sent, str):
            sent = get_nlp()(sent)
        #
        text_sentence = ' '.join([word.text for word in sent]).strip()

        # hack to pick up common phrasal signifiers
        sent, text_sentence = clean_multiword_phrases(sent, text_sentence)

        sources_in_sent = []
        ## extract all nsubj of VERB if VERB is 'said', 'says' or 'say'
        nsubjs = []
        for possible_subject in sent:
            if (
                possible_subject.dep_ == 'nsubj' and 
                possible_subject.head.pos_ == 'VERB' and 
                possible_subject.head.text in speaking_lexicon
            ):
                nsubjs.append(possible_subject.text)
                entities[possible_subject.text]['quote sentence'].append((s_idx, text_sentence))
                seen.add(s_idx)
                sources_in_sent.append({
                    'head': possible_subject.text,
                    'quote_type': 'quote'
                })

                
        for noun_phrase in sent.noun_chunks:
            for nsubj in nsubjs:
                if nsubj in noun_phrase.text:
                    entities[noun_phrase.text]['quote sentence'].append((s_idx, text_sentence))
                    seen.add(s_idx)
                    sources_in_sent.append({
                        'head': noun_phrase.text,
                        'quote_type': 'quote'
                    })

        output_sents.append({
            'sent': sent,
            'sources': sources_in_sent
        })

    # get background
    for s_idx, sent in enumerate(sents):
        if isinstance(sent, str):
            sent = get_nlp()(sent)

        if s_idx not in seen:
            # get person-entities
            for ent in sent.ents:
                if ent.label_ in ner_list:
                    entities[ent.text]['background sentence'].append((s_idx, sent.text))

    if cluster:
        entities = cluster_by_name_overlap_jaro(entities)
    if dedupe_sents:
        entities = dedupe_sents_in_entities(entities)
    if return_sents:
        return entities, output_sents
    else:
        return entities


def dedupe_sents_in_entities(quotes):
    deduped_output = {}
    for person, sents in quotes.items():
        person_output = {'background sentence': [], 'quote sentence': []}
        for sent_type in ['background sentence', 'quote sentence']:
            seen = set()
            for idx, s in sents[sent_type]:
                if idx not in seen:
                    seen.add(idx)
                    person_output[sent_type].append((idx, s))
        deduped_output[person] = person_output
    return deduped_output

def cluster_by_name_overlap_jaro(entities, sim=.95):
    """Append clusters of similar names together (fails when a first name is the same for different people.)
        Input:
            * entities: list of NER names extracted from text

        Output:
            * Mapping from full-name to set of all name-variants that appear in text

    """
    name_mapper = defaultdict(set)
    mapped = defaultdict(bool)
    entity_list = list(entities)
    n_ent = len(entity_list)

    clusters = []
    for i in range(n_ent):
        n1 = entity_list[i]

        if not mapped[n1]:
            # new cluster
            n_cluster = [(i, n1)]
            mapped[n1] = True
            #
            for j in range(i, n_ent):
                n2 = entity_list[j]
                if not mapped[n2]:
                    # get similarites
                    name_parts = []
                    for w_i in n1.split():
                        for w_j in n2.split():
                            dist = jellyfish.jaro_winkler(w_i, w_j)
                            name_parts.append(dist)

                    # append to cluster
                    if max(name_parts) > sim:
                        n_cluster.append((j, n2))
                        mapped[n2] = True
            # record
            clusters.append(n_cluster)
    return clusters

from copy import copy
import string
from unidecode import unidecode
def remove_problematic_name_parts(s):
    s = unidecode(s)
    for p in ['Jr', 'Sr', 'III', '\'s']:
        s = s.replace(p, '')
    for p in string.punctuation:
        s = s.replace(p, '')
    return s.strip()

def cluster_by_last_name_equality(entities, sim=.98):
    """Append clusters of people by their last names.
        Input:
            * entities: list of NER names extracted from text

        Output:
            * list of clusters where each cluster is:
                * a list of (idx, full-name) tuples
    """
    mapped = defaultdict(bool)
    entity_list = list(entities)
    n_ent = len(entity_list)
    clusters = []
    for i in range(n_ent):
        n1 = entity_list[i]
        if not mapped[i]:
            # new cluster
            n_cluster = [(i, n1)]
            mapped[i] = True
            temp_n1 = copy(n1)
            temp_n1 = remove_problematic_name_parts(temp_n1)
            for j in range(i, n_ent):
                n2 = entity_list[j]
                if not mapped[j]:
                    temp_n2 = copy(n2)
                    temp_n2 = remove_problematic_name_parts(temp_n2)
                    # append to cluster
                    if jellyfish.jaro_winkler(temp_n1.split()[-1], temp_n2.split()[-1]) > sim:
                        n_cluster.append((j, n2))
                        mapped[j] = True
            # record
            clusters.append(n_cluster)
    return clusters


def get_name_cluster_head_by_length(clusters):
    head_to_cluster = {}
    head_idx_to_cluster_idx = {}
    cluster_to_head = {}
    cluster_idx_to_head = {}
    for c in clusters:
        cluster_head = max(c, key=lambda x: len(x[1]))
        head_to_cluster[cluster_head[1]] = list(map(lambda x: x[1], c))
        head_idx_to_cluster_idx[cluster_head[0]] = list(map(lambda x: x[0], c))
        for c_i in c:
            cluster_to_head[c_i[1]] = cluster_head[1]
            cluster_idx_to_head[c_i[0]] = cluster_head[0]
    return head_to_cluster, head_idx_to_cluster_idx, cluster_to_head, cluster_idx_to_head


def cluster_by_coref(entities):
    pass


def merge_clusters(entities, cluster_mapper):
    # group for output
    entities_clustered = defaultdict(lambda: {'background sentence': [], 'quote sentence': []})
    for c_key, cluster in cluster_mapper.items():
        for c_i in cluster:
            for part in ['background sentence', 'quote sentence']:
                for s_idx, s in entities[c_i][part]:
                    existing_s_ids = set(map(lambda x: x[0], entities_clustered[c_key][part]))
                    if s_idx not in existing_s_ids:
                        entities_clustered[c_key][part].append((s_idx, s))
    
    ## 
    return entities_clustered


contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who has",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}

import matplotlib
color_wheel = list(matplotlib.colors.cnames.values())
hex_to_rgb_tup = lambda h: tuple(int(h.replace('#','')[i:i+2], 16) for i in (0, 2, 4))
make_rgb = lambda h, a: 'rgba' + str(hex_to_rgb_tup(h) + (a,))

def html_replace_list(idx_list, sentence, sent_toks, color):
    for idx in idx_list:
        word = sent_toks[idx]
        sentence = sentence.replace(' %s ' % word, ' <span style="background-color: ' +  color + '">' + word + '</span> ')
    return sentence
