import spacy
import os
from collections import defaultdict
import jellyfish
from spacy import displacy
from  more_itertools import unique_everseen
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


nlp = spacy.load('en_core_web_sm')

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

def preprocess(text):
    """Remove duplicate and short sentences."""
    text = text.replace('\n', ' ').replace("''", '"').strip()
    doc = nlp(text)
    unique_sentences = list(unique_everseen(map(lambda x: x.text.strip(), doc.sents)))
    text = ' '.join(unique_sentences)
    return nlp(text)

def get_quotes_method_1(doc):
    ## extract quotes
    entities = defaultdict(lambda: {'background sentence': [], 'quote sentence': []})

    signifiers = [' say ', ' said ', ' says ', ' according to ', ' described ', ' describes ']
    for s_idx, sent in enumerate(doc.sents):
        ## 
        text_sentence = ' '.join([word.text for word in sent]).strip()

        ## get person-entities
        for ent in sent.ents:
            if ent.label_ == 'PERSON':
                is_quote = False
                ## find quote sentence
                for sig in signifiers:
                    if sig in text_sentence:
                        is_quote = True
                entities[ent.text][['background sentence', 'quote sentence'][is_quote]].append((s_idx, text_sentence))
    
    return cluster_entities(entities)


def get_quotes_method_2(doc):
    """Get quoted people by finding the nsubj of a 'say', 'said' or 'according to' verb."""
    ## extract quotes
    entities = defaultdict(lambda: {'background sentence': [], 'quote sentence': []})

    signifiers = [' say ', ' said ', ' says ', ' according to ']
    seen = set()
    ## get quotes
    for s_idx, sent in enumerate(doc.sents):
        ## 
        text_sentence = ' '.join([word.text for word in sent]).strip()

        ## hack to pick up common phrasal signifiers
        if 'according to' in text_sentence:
            sent = nlp(text_sentence.replace('according to', 'said'))

        ## extract all nsubj of VERB if VERB is 'said', 'says' or 'say'
        nsubjs = []
        for possible_subject in sent:
            if (
                possible_subject.dep_ == 'nsubj' and 
                possible_subject.head.pos_ == 'VERB' and 
                possible_subject.head.text in ('say', 'says', 'said')
            ):
                nsubjs.append(possible_subject.text)
                entities[possible_subject.text]['quote sentence'].append((s_idx, text_sentence))
                
        for noun_phrase in sent.noun_chunks:
            for nsubj in nsubjs:
                if nsubj in noun_phrase.text:
                    entities[noun_phrase.text]['quote sentence'].append((s_idx, text_sentence))
                    seen.add(s_idx)

    ## get background
    for s_idx, sent in enumerate(doc.sents):
        if s_idx not in seen:
            ## get person-entities
            for ent in sent.ents:
                if ent.label_ == 'PERSON':
                    entities[ent.text]['background sentence'].append((s_idx, text_sentence))
    
    return cluster_entities(entities)


def cluster_entities(entities, sim=.95):
    ## append clusters together
    name_mapper = defaultdict(set)
    mapped = defaultdict(bool)

    entity_list = list(entities)
    n_ent = len(entity_list)

    clusters = []
    for i in range(n_ent):
        n1 = entity_list[i]

        if not mapped[n1]:
            ## new cluster
            n_cluster = [n1]
            mapped[n1] = True
            ## 
            for j in range(i, n_ent):
                n2 = entity_list[j]
                if not mapped[n2]:

                    ## get similarites 
                    name_parts = []
                    for w_i in n1.split():
                        for w_j in n2.split():
                            dist = jellyfish.jaro_winkler(w_i, w_j)
                            name_parts.append(dist)

                    ## append to cluster
                    if max(name_parts) > sim:
                        n_cluster.append(n2)
                        mapped[n2] = True
            ## record
            clusters.append(n_cluster)

    cluster_mapper = {}
    for cluster in clusters:
        key = max(cluster, key=lambda x: len(x))
        cluster_mapper[key] = cluster

    ## group for output
    entities_clustered = defaultdict(lambda: {'background sentence': [], 'quote sentence': []})
    for c_key, cluster in cluster_mapper.items():
        for c_i in cluster:
            for part in ['background sentence', 'quote sentence']:
                entities_clustered[c_key][part].extend(entities[c_i][part])
    
    ## 
    return entities_clustered