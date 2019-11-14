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


def cluster_entities_method_1(entities, sim=.95):
    """Append clusters of similar names together
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