import os
from collections import defaultdict
import jellyfish
from spacy import displacy
from more_itertools import unique_everseen
import pandas as pd
import xml.etree.ElementTree as ET


def map_words(input_text, cv, cutoff=None):
    """
        Maps text words to vocab indices, based on CountVectorizer cv's vocabulary.
    """
    # get vocabulary
    vocabulary = None
    if 'vocabulary_' in cv.__dict__:
        vocabulary = cv.vocabulary_
    elif 'vocabulary' in cv.__dict__:
        vocabulary = cv.vocabulary
    else:
        exit("train cv...")

    # transform
    output = []
    for idx, word in enumerate(input_text.split()):
        if word in vocabulary:
            output.append(vocabulary[word])
    return output[:cutoff]


class EntityMention():    
    def __init__(self, name=None, start=None, end=None, head=None, sentence=None):
        self.name=name
        self.start=start
        self.end=end
        self.head=head
        self.sentence=sentence
    
    def key(self):
        return (self.sentence, self.head)
    
    def __eq__(self, other):
        return self.key() == other.key()
    
    def __getitem__(self, item):
        return self.__dict__[item]
    
    def __repr__(self):
        return '<Entity: %s>' % str({'name': self.name, 'start':self.start, 'end': self.end, 'head':self.head, 'sentence': self.sentence})


def extract_pers_ners(ner_tags, words):
    """Extract PERS ners.
            * ner_tags: a list of lists specifying tags
            * ner_words: a list of list of document words
    """
    person_list = []
    
    ### 
    assert(len(ner_tags) == len(words))
    for sentence_idx, (sentence_ner_tags, sentence_words) in enumerate(zip(ner_tags, words)):
        in_person = False
        cur_person = defaultdict(list)
        for word_idx, (tag, word) in enumerate(zip(sentence_ner_tags, sentence_words)):
            if 'PER' in tag:
                cur_person['name'].append(word)
                cur_person['span'].append(word_idx)
                in_person = True
            else:
                if in_person:
                    ent = EntityMention(
                        name=' '.join(cur_person['name']),
                        start=min(cur_person['span']),
                        end=max(cur_person['span']),
                        head=max(cur_person['span']),
                        sentence=sentence_idx
                    )
                    person_list.append(ent)
                    cur_person = defaultdict(list)
                    in_person = False
    ### 
    return person_list



def is_quote(lemmas, pos_tags, deps, entity_mention):
    """For an entity mention, check if the sentence is a quote.
    
    Input:
        * sentence: sentence of the entity mention
        * lemmas: lemmas of the sentence
        * pos_tags: pos tags of the sentence
        * deps: xml dependence parse for the sentence
        * entity_mention: The mention.
    
    Output:
        * speaking_vb_idx
        * speaking_dep_idx
    """
    max_left_vb_positions = 5
    max_right_vb_positions = 5
    speaking_candidate_lemmas = [
        'say', 'said', 'says', 
        'recalls', 'recall', 'recalled',
        'continued', 'continue',
        'added', 'add', 'told',
        'according to' ## need to handle this with the swith hack
    ]
    
    ###     
    sent = entity_mention.sentence
    head = entity_mention.head

    pos = pos_tags[sent]
    lem = lemmas[sent]
    sent_deps = deps[sent]

    ## by verb position
    ### left:
    #### ex: quote-mark QUOTE quote-mark [,] verb [modifier] [determiner] [title] name
    #### ex: quote-mark QUOTE quote-mark [; or ,] [title] name [modifier] verb
    left_located_verb = range(max(0, head - max_left_vb_positions), head)
    ### right: name [, up to 60 characters ,] verb [:|that] quote-mark QUOTE quote-mark
    right_located_verb = range(min(head + 1, len(lem)), min(head + max_left_vb_positions, len(lem)))
    verbs = list(filter(lambda x: 'VB' in pos[x], left_located_verb)) + list(filter(lambda x: 'VB' in pos[x], right_located_verb))
    speaking_vb_idx = list(filter(lambda vb: lem[vb] in speaking_candidate_lemmas, verbs))


    ## by dependency
    subjs = list(filter(lambda x: x.attrib['type'] in ('nsubj', 'dobj'), sent_deps))
    subj_deps = list(filter(lambda x: x.find('dependent').attrib.get('idx', '') == str(head+1), subjs))

    # filter out governors that aren't verbs
    subj_vbs = list(filter(lambda x: 'VB' in pos[int(x.find('governor').attrib['idx']) - 1], subj_deps))
    subj_vbs_idx = list(map(lambda x: int(x.find('governor').attrib['idx']) - 1, subj_vbs))
    speaking_deps = list(filter(lambda x: x.findtext('governor') in speaking_candidate_lemmas, subj_vbs))
    speaking_deps_idx = list(map(lambda x: int(x.find('governor').attrib['idx']) - 1, speaking_deps))
    
    return (speaking_vb_idx, speaking_deps_idx, subj_vbs_idx)

def cluster_entities_method_2(entities, sim=.95):
    """Cluster ALL mentions of named entities together by name-matching, starting with the last name.
    
    Entities: List of Entities with attributes:
        {
            "sentence"
            "name"
            "start": span start
            "end" : span end
            "head"
        }
    """
    name_mapper = defaultdict(set)
    mapped = {} ## name mapper, just so we're not calculating similarity a lot
    seen = defaultdict(bool)  ## unique entity mention mapper
    entity_list = list(entities)
    num_ent = len(entity_list)

    clusters = []
    cluster_idx = 0 
    for i, n1 in enumerate(entity_list):
        name_1 = n1.name
        key_1 = n1.key()

        if name_1 in mapped:
            if not seen[key_1]:
                clusters[mapped[name_1]] += [n1]
                seen[key_1] = True
   
        else:
            ## new cluster
            n_cluster = [n1]

            ## record
            clusters.append(n_cluster)
            mapped[name_1] = cluster_idx
            seen[key_1] = True
            
            ##
            for j in range(i, num_ent):
                n2 = entity_list[j]
                name_2 = n2.name
                key_2 = n2.key()
                
                if name_2 in mapped:
                    if not seen[key_2]:
                        clusters[mapped[name_2]] += [n2]
                        seen[key_2] = True

                else:    
                    ## get similarites 
                    name_parts = []
                    for w_i in name_1.split():
                        for w_j in name_2.split():
                            dist = jellyfish.jaro_winkler(w_i, w_j)
                            name_parts.append(dist)
                            
                    ## append to cluster
                    if max(name_parts) > sim:
                        n_cluster.append(n2)
                        mapped[name_2] = cluster_idx
                        seen[key_2] = True
                        
            cluster_idx += 1

    cluster_mapper = {}
    for cluster in clusters:
        key = max(cluster, key=lambda x: len(x.name))
        cluster_mapper[key.name] = {
            'mentions': cluster,
            'first_mention': min(cluster, key=lambda x: x.key())
        }

    return cluster_mapper


def convert_coref(doc_etree, sentences):
    coref_x = doc_etree.find('document').find('coreference')
    if coref_x is None:
        return []

    entities = []
    for entity_x in coref_x.findall('coreference'):
        mentions = []
        for mention_x in entity_x.findall('mention'):
            m = EntityMention()
            m.sentence = int(mention_x.find('sentence').text) - 1
            m.start = int(mention_x.find('start').text) - 1
            m.end = int(mention_x.find('end').text) - 1
            m.head = int(mention_x.find('head').text) - 1
            m.name = sentences[m.sentence][m.head]
#             m.name = ' '.join(map(lambda idx: sents_list[m.sentence][idx], range(m.start, m.end)))
            mentions.append(m)
        ent = {}
        ent['mentions'] = mentions
        first_mention = min((m.sentence,m.head) for m in mentions)
        ent['first_mention'] = first_mention
        entities.append(ent)
    return entities


def parse_etree(doc_etree):
    sents = doc_etree.find('document').find('sentences').findall('sentence')

    ### sentences
    sents_list = []
    for s in sents:
        tokens = s.find('tokens').findall('token')
        words = list(map(lambda x: x.findtext('word'), tokens))
        sents_list.append(words)
        
    ### NER
    named_ents_list = []
    for s in sents:
        tokens = s.find('tokens').findall('token')
        words = list(map(lambda x: x.findtext('word'), tokens))
        named_ents = list(map(lambda x: x.findtext('NER'), tokens))
        named_ents_list.append(named_ents)
        
    ### lemmas
    lemmas = []
    for sent in sents:
        toks = sent.find('tokens').findall('token')
        sent_lemmas = []
        for tok in toks:
            lemma = tok.findtext('lemma')
            sent_lemmas.append(lemma)
        lemmas.append(sent_lemmas)


    ### pos tags
    pos_tags = []
    for sent in sents:
        toks = sent.find('tokens').findall('token')
        sent_pos = []
        for tok in toks:
            pos = tok.findtext('POS')
            sent_pos.append(pos)
        pos_tags.append(sent_pos)


    ### dependencies
    deps = []
    for sent in sents:
        sent_deps = (sent
            .find('dependencies[@type="enhanced-plus-plus-dependencies"]')
            .findall('dep')
        )
        deps.append(sent_deps)

    return sents_list, named_ents_list, lemmas, pos_tags, deps


def extract_and_merge_ners(named_ents_list, sents_list, stanford_corefs):
    person_list = extract_pers_ners(named_ents_list, sents_list)
    named_entity_clusters = cluster_entities_method_2(person_list)
    named_entity_clusters = merge_coref_and_ner_clusters(named_entity_clusters, person_list, stanford_corefs)
    return named_entity_clusters

def merge_coref_and_ner_clusters(named_entity_clusters, person_list, stanford_corefs):
    """Enhance each NER cluster with coreferents."""

    canonical_mentions = {}
    for head, ent_dict in named_entity_clusters.items():
        for ent in ent_dict['mentions']:
            canonical_mentions[ent.key()] = head

    ## get entities associated with PERSON ner tag
    ner_mentions = []
    # start with corefs
    for entity in stanford_corefs:
        found = False
        for entity_mention in entity['mentions']:
            if entity_mention in person_list:
                found = True
                mention = canonical_mentions[entity_mention.key()]
        
        if found:
            for entity_mention in entity['mentions']:
                named_entity_clusters[mention]['mentions'].append(entity_mention)

    return named_entity_clusters


def extract_quotes_for_each_ner(named_entity_clusters, lemmas, pos_tags, deps):
    """Enhance each NER cluster with quotes."""

    for name, entity_dict in named_entity_clusters.items():
        speaking_vbs = []
        all_vbs = []
        for mention in entity_dict['mentions']:
            speaking_vb_idx, speaking_deps_idx, subj_vbs_idx = is_quote(lemmas, pos_tags, deps, mention)

            ## record
            all_speaking_vb_idx = list(
                # set(speaking_vb_idx) | 
                set(speaking_deps_idx)
            )
            for speaking_vb in all_speaking_vb_idx:
                speaking_vbs.append({'sentence': mention.sentence, 'verb_idx': speaking_vb})

            for vb in subj_vbs_idx:
                all_vbs.append({'sentence': mention.sentence, 'verb_idx': vb})
        
        entity_dict['speaking_vbs'] = speaking_vbs
        entity_dict['all_vbs'] = all_vbs

    return named_entity_clusters


def parse_stanford_and_get_people(stanford_parse):
    """
        Wrapper method for other methods defined in the util. 

        Return sources and usefull stanford parses, given a parse-tree of interest.
    """
    
    fn = os.path.basename(stanford_parse).replace('.txt.xml', '')
    data = open(stanford_parse, encoding='utf-8').read()
    doc_etree = ET.fromstring(data)
    ## parse
    sents_list, named_ents_list, lemmas, pos_tags, deps = parse_etree(doc_etree)
    stanford_corefs = convert_coref(doc_etree=doc_etree, sentences=sents_list)
    ## group named entities
    named_entity_clusters = extract_and_merge_ners(named_ents_list, sents_list, stanford_corefs)
    named_entity_clusters = extract_quotes_for_each_ner(named_entity_clusters, lemmas, pos_tags, deps)
    ## get sources
    sources = dict(filter(lambda x: len(x[1]['speaking_vbs']) > 0, named_entity_clusters.items()))
    ## return 
    return fn, sources, sents_list, lemmas, named_ents_list, deps


def parse_people_and_docs(stanford_parse_file, include_all_mentions=True, include_all_sentences_in_doc=True, use_lemmas=True):
    """Prepare data for topic model. Parse people .

    input:
        * stanford_parse_file: the filename of the stanford parse.
        * include_all_mentions: whether to include all mentions of a source in the text related to a source.
            True: includes all mentions.
            False: includes only speaking sentences and the first mention sentence.
        * include_all_sentences_in_doc: whether to include all sentences in a document in the text.
            True: includes all sentences.
            False: includes only the sentences that are not already allocated to sources.
        * use_lemmas: parse lemmas or full words.
            True: only lemmas.
            False: full words.
    """

    doc_id, sources, sents_list, lemmas, named_ents_list, deps = parse_stanford_and_get_people(stanford_parse_file)
    
    if use_lemmas:
        corpora = lemmas
    else:
        corpora = sents_list

    ## parse sentences for sources
    source_sentences = {}
    included_sent_ids = set()
    if include_all_mentions:
        for name, source_info in sources.items():
            sents = []
            for mention in source_info['mentions']:
                sent_id = mention.sentence
                included_sent_ids.add(sent_id)
                sents.append(' '.join(corpora[sent_id]))
            source_sentences[name] = ' '.join(sents)

    else:
        for name, source_info in sources.items():
            ## get text and id for first mention
            first_mention_id = source_info['first_mention'].sentence
            first_mention = ' '.join(corpora[first_mention_id])
            included_sent_ids.add(first_mention_id)
            ## get text and ids for speaking sentences
            speaking_sents = []
            for speaking_vb in source_info['speaking_vbs']:
                sent_id = speaking_vb['sentence']
                speaking_sents.append(' '.join(corpora[sent_id]))
                included_sent_ids.add(sent_id)
            source_sentences[name] = first_mention + ' ' + ' '.join(speaking_sents)

    ## parse sentences for doc
    if include_all_sentences_in_doc:
        doc_sentences = ' '.join(list(map(lambda x: ' '.join(x), corpora)))
    else:
        doc_sentences = []
        for sent_idx, sent in enumerate(corpora):
            if sent_idx not in included_sent_ids:
                doc_sentences.append(' '.join(sent))
        doc_sentences = ' '.join(doc_sentences)

    return {
        'source_sentences': source_sentences,
        'doc_sentences': doc_sentences,
        'doc_id': doc_id
    } 

def parse_dep(dep):
    """Parse a Stanford dependency XML and return useful parts as a dictionary."""
    return {
        "type": dep.attrib['type'],
        "governor": dep.find('.//governor').text,
        "governor-idx": dep.find('.//governor').attrib['idx'],
        "dependent": dep.find('.//dependent').text,
        "dependent-idx": dep.find('.//dependent').attrib['idx'],
    }