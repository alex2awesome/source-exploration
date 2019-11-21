import numpy as np
import os
import json
from tqdm import tqdm
import pickle
import glob, re

def categorical(k, p):
    return np.random.choice(range(k), p=p)

class BOW_Source_GibbsSampler():
    def __init__(
        self, docs, vocab, use_labels=False,
        num_doc_types=20, num_source_types=20, num_topics=25,
        H_T_prior=10, H_S_prior=10, H_z_prior=1, H_w_prior=1,
    ):
        self.use_labels = use_labels

        # data
        self.docs = docs
        self.vocab = vocab
        self.n_docs = len(docs)
        self.n_sources = sum(list(map(lambda x: len(x['source_map']), docs)))
        self.n_vocab = len(vocab)

        # hyperparameters
        self.num_doc_types = num_doc_types
        self.num_source_types = num_source_types
        self.num_topics = num_topics
        self.doc_types = list(range(self.num_doc_types))
        self.source_types = list(range(self.num_source_types))
        self.topics = list(range(self.num_topics))

        # priors
        self.H_T = H_T_prior
        self.H_S = H_S_prior
        self.H_z = H_z_prior
        self.H_w = H_w_prior

        # counts 
        ## doc-type probability
        #### I(doc d has type t)
        #### count_t_d
        self.doctype_counts = np.zeros(self.num_doc_types)
        self.doc_to_doctype = {}
        self.old_doc_to_doctype = {}

        ## source-type probability
        #### I(doc d has type t, Source S_d has type s)
        #### count_t_s_d_Sd
        self.sourcetype_by_doctype__source_counts = np.zeros((self.num_doc_types, self.num_source_types))
        self.doctype__source_counts = np.zeros(self.num_doc_types)
        self.source_to_source_type = {}
        self.old_source_to_source_type = {}

        ## background word-topic probabilities
        #### I(doc d has type t, word w has topic k)
        #### count_k_d_t_w
        self.doctype__wordtopic_counts = np.zeros(self.num_doc_types)
        self.doctype_by_wordtopic__wordtopic_counts = np.zeros((self.num_doc_types, self.num_topics))
        self.word_to_background_topic = {} ## a word can possibly be both a background word and a source word

        ## source word-topic probabilities
        #### I(source S_d in doc d has type s, word w has topic k)
        #### count_k_d_s_Sd_w
        self.sourcetype__wordtopic_counts = np.zeros(self.num_source_types)
        self.sourcetype_by_wordtopic__wordtopic_counts = np.zeros((self.num_source_types, self.num_topics))
        self.word_to_sourcetopic = {}

        ## word probabilities
        self.vocab_by_wordtopic__word_counts = np.zeros((self.n_vocab, self.num_topics))
        self.wordtopic__word_counts = np.zeros(self.num_topics)


    def initialize(self):
        ## probability vectors
        doctype_prob_vec = np.ones(self.num_doc_types) / self.num_doc_types
        sourcetype_prob_vec = np.ones(self.num_source_types) / self.num_source_types
        z_prob_vec = np.ones(self.num_topics) / self.num_topics

        ## iterate through documents
        for doc_id, doc in tqdm(enumerate(self.docs), total=len(self.docs)):
            ## set doc type
            doctype = categorical(self.num_doc_types, p=doctype_prob_vec)
            self.doctype_counts[doctype] += 1
            self.doctype__wordtopic_counts[doctype] += len(doc['doc_vec'])
            self.doc_to_doctype[doc_id] = doctype
            self.old_doc_to_doctype[doc_id] = doctype

            ## set background word-topics
            for word_id, word in enumerate(doc['doc_vec']):
                doc_word_topic = categorical(self.num_topics, p=z_prob_vec)
                ## counts
                self.doctype_by_wordtopic__wordtopic_counts[doctype, doc_word_topic] += 1
                self.vocab_by_wordtopic__word_counts[word, doc_word_topic] += 1
                self.wordtopic__word_counts[doc_word_topic] += 1
                ## cache
                self.word_to_background_topic[(doc_id, word_id)] = doc_word_topic

            ## iterate through sources
            for source_id, source_text_vec in doc['source_vecs'].items():
                ## set source types
                source_labels = doc.get('source_labels', {})
                if (len(source_labels) > 0) and self.use_labels and (source_id in source_labels):
                    sourcetype = source_labels[source_id]
                else:
                    sourcetype = categorical(self.num_source_types, p=sourcetype_prob_vec)
                ## counts
                self.doctype__source_counts[doctype] += 1
                self.sourcetype_by_doctype__source_counts[doctype, sourcetype] += 1
                self.sourcetype__wordtopic_counts[sourcetype] += len(source_text_vec)
                ## cache
                self.source_to_source_type[(doc_id, source_id)] = sourcetype
                self.old_source_to_source_type[(doc_id, source_id)] = sourcetype

                ## set source word-topics
                for word_id, word in enumerate(source_text_vec):
                    source_wordtopic = categorical(self.num_topics, p=z_prob_vec)
                    ## counts
                    self.sourcetype_by_wordtopic__wordtopic_counts[sourcetype, source_wordtopic] += 1
                    self.vocab_by_wordtopic__word_counts[word, source_wordtopic] += 1
                    self.wordtopic__word_counts[source_wordtopic] += 1
                    ## cache
                    self.word_to_sourcetopic[(doc_id, source_id, word_id)] = source_wordtopic

    ### 
    # sample doc type
    ###

    def doctype_prob(self, proposed_doctype, doc_id):
        ## doc_type factor
        doctype_term = np.log(self.H_T + self.doctype_counts[proposed_doctype])

        ## source_type factor
        sourcetype_term = 0
        log_denom = np.log(self.num_source_types * self.H_S + self.doctype__source_counts[proposed_doctype])
        for source_id in self.docs[doc_id]['source_map'].keys():
            sourcetype = self.source_to_source_type[(doc_id, source_id)]
            num = self.H_S + self.sourcetype_by_doctype__source_counts[proposed_doctype, sourcetype]
            sourcetype_term += np.log(num) - log_denom

        ## background word_topic factor
        background_wordtopic_term = 0
        log_denom = np.log(self.num_topics * self.H_z + self.doctype__wordtopic_counts[proposed_doctype])
        for word_id, word in enumerate(self.docs[doc_id]['doc_vec']):
            if background_wordtopic_term < -100: ## hack... is there any other way to deal with very low probabilities?
                break
            doc_wordtopic = self.word_to_background_topic[(doc_id, word_id)]
            num = self.H_z + self.doctype_by_wordtopic__wordtopic_counts[proposed_doctype, doc_wordtopic]
            background_wordtopic_term += np.log(num) - log_denom

        ## combine and return
        log_prob = doctype_term + sourcetype_term + background_wordtopic_term
        return np.exp(log_prob)

    def propose_new_doctype(self, doc_id):
        doc_prob_vec = np.zeros(self.num_doc_types)
        for t in self.doc_types:
            doc_prob_vec[t] = self.doctype_prob(t, doc_id)
        new_doctype = categorical(self.num_doc_types, p=doc_prob_vec / doc_prob_vec.sum())
        return new_doctype

    def sample_doctype(self):
        ## for each doc, decrement counts and resample 
        for doc_id, doc in tqdm(enumerate(self.docs), total=len(self.docs)):
            old_doctype = self.doc_to_doctype[doc_id]
            ## decrement
            self.doctype_counts[old_doctype] -= 1           
            new_doctype = self.propose_new_doctype(doc_id)
            ## increment
            self.doctype_counts[new_doctype] += 1
            ## cache
            self.doc_to_doctype[doc_id] = new_doctype
            self.old_doc_to_doctype[doc_id] = old_doctype


    ### 
    # Sample Source type
    ###

    def sourcetype_prob(self, proposed_sourcetype, doc_id, source_id):
        """Calculate the probability of a new sourcetype, s, for a given source."""
        doc_type = self.doc_to_doctype[doc_id]

        ## sourcetype factor
        sourcetype_term = np.log(self.H_S + self.sourcetype_by_doctype__source_counts[doc_type, proposed_sourcetype])

        ## source word_topic factor
        source_wordtopic_term = 0
        denom = self.num_topics * self.H_z + self.sourcetype__wordtopic_counts[proposed_sourcetype]
        for word_id, word in enumerate(self.docs[doc_id]['source_vecs'][source_id]):
            if source_wordtopic_term < -80: ## hack... is there any other way to deal with very low probabilities?
                break
            if source_wordtopic_term > 50:
                break
            source_wordtopic = self.word_to_sourcetopic[(doc_id, source_id, word_id)]
            num = self.H_z + self.sourcetype_by_wordtopic__wordtopic_counts[proposed_sourcetype, source_wordtopic]
            source_wordtopic_term += np.log(num) - np.log(denom)

        ## combine and return 
        log_prob = sourcetype_term + source_wordtopic_term
        return np.exp(log_prob)


    def propose_new_sourcetype(self, doc_id, source_id):
        source_prob_vec = np.zeros(self.num_source_types)
        for s in self.source_types:
            source_prob_vec[s] = self.sourcetype_prob(s, doc_id, source_id)

        ### hacks for removing negative probability
        source_prob_vec[np.where(source_prob_vec == np.inf)] = .99   ## another hack
        source_prob_vec[np.where(source_prob_vec == -np.inf)] = .01  ## another hack
        source_prob_vec = source_prob_vec / source_prob_vec.sum()
        source_prob_vec[np.where(source_prob_vec == np.nan)] = 0  ## another hack
        source_prob_vec = source_prob_vec / source_prob_vec.sum()
        ### todo: log what is happening here.
        try:
            sourcetype = categorical(self.num_source_types, p=source_prob_vec)
            return sourcetype
        except:
            print('source sampling error...')
            print(source_prob_vec)
            print()
            return np.random.choice(range(self.num_source_types))

    def sample_source_type(self):
        for doc_id, doc in tqdm(enumerate(self.docs), total=len(self.docs)):
            ## otherwise, sample
            old_doc_type = self.old_doc_to_doctype[doc_id]
            doc_type = self.doc_to_doctype[doc_id]
            for source_id, source_vec in doc['source_vecs'].items():
                if len(doc.get('source_labels', {})) > 0 and self.use_labels and (source_id in doc['source_labels']):
                    continue

                old_source_type = self.source_to_source_type[(doc_id, source_id)]
                ## decrement
                self.sourcetype_by_doctype__source_counts[old_doc_type, old_source_type] -= 1
                self.doctype__source_counts[old_doc_type] -= 1
                ## sample new source type
                new_source_type = self.propose_new_sourcetype(doc_id, source_id)
                ## increment
                self.sourcetype_by_doctype__source_counts[doc_type, new_source_type] += 1
                self.doctype__source_counts[doc_type] += 1
                ## cache
                self.source_to_source_type[(doc_id, source_id)] = new_source_type
                self.old_source_to_source_type[(doc_id, source_id)] = old_source_type


    ###
    # Sample source word topic
    ###

    def source_wordtopic_prob(self, proposed_wordtopic, doc_id, source_id, word_id):
        sourcetype = self.source_to_source_type[(doc_id, source_id)]
        word = self.docs[doc_id]['source_vecs'][source_id][word_id]

        ## source word_topic factor
        source_wordtopic_term = self.H_z + self.sourcetype_by_wordtopic__wordtopic_counts[sourcetype, proposed_wordtopic]

        ## word factor
        denom = self.n_vocab * self.H_w + self.wordtopic__word_counts[proposed_wordtopic]
        num = self.H_w + self.vocab_by_wordtopic__word_counts[word, proposed_wordtopic]
        word_term = (num / denom)

        ## combine and return 
        return source_wordtopic_term * word_term 


    def propose_new_source_wordtopic(self, doc_id, source_id, word_id):
        source_wordtopic_prob_vec = np.zeros(self.num_topics)
        for k in self.topics:
            source_wordtopic_prob_vec[k] = self.source_wordtopic_prob(k, doc_id, source_id, word_id)
        wordtopic = categorical(self.num_topics, p=source_wordtopic_prob_vec / source_wordtopic_prob_vec.sum())
        return wordtopic


    def sample_source_word_topic(self):
        for doc_id, doc in tqdm(enumerate(self.docs), total=len(self.docs)):
            for source_id, source_vec in doc['source_vecs'].items():
                source_type = self.source_to_source_type[(doc_id, source_id)]
                old_source_type = self.old_source_to_source_type[(doc_id, source_id)]
                ## 
                for word_id, word in enumerate(source_vec):
                    old_word_topic = self.word_to_sourcetopic[(doc_id, source_id, word_id)]
                    ## decrement counts
                    self.sourcetype_by_wordtopic__wordtopic_counts[old_source_type, old_word_topic] -= 1
                    self.sourcetype__wordtopic_counts[old_source_type] -= 1
                    self.vocab_by_wordtopic__word_counts[word, old_word_topic] -= 1
                    self.wordtopic__word_counts[old_word_topic] -= 1
                    ## sample new work topic
                    new_word_topic = self.propose_new_source_wordtopic(doc_id, source_id, word_id)
                    # increment counts
                    self.sourcetype_by_wordtopic__wordtopic_counts[source_type, new_word_topic] += 1
                    self.sourcetype__wordtopic_counts[source_type] += 1
                    self.vocab_by_wordtopic__word_counts[word, new_word_topic] += 1
                    self.wordtopic__word_counts[new_word_topic] += 1
                    ## cache
                    self.word_to_sourcetopic[(doc_id, source_id, word_id)] = new_word_topic


    ###
    # Sample background word topic
    ###

    def background_wordtopic_prob(self, proposed_wordtopic, doc_id, word_id):
        doctype = self.doc_to_doctype[doc_id]
        word = self.docs[doc_id]['doc_vec'][word_id]

        ## source word_topic factor
        background_wordtopic_term = self.H_z + self.doctype_by_wordtopic__wordtopic_counts[doctype, proposed_wordtopic]

        ## word factor
        denom = self.n_vocab * self.H_w + self.wordtopic__word_counts[proposed_wordtopic]
        num = self.H_w + self.vocab_by_wordtopic__word_counts[word, proposed_wordtopic]
        word_term = (num / denom)

        ## combine and return 
        return background_wordtopic_term * word_term


    def propose_new_background_wordtopic(self, doc_id, word_id):
        background_wordtopic_prob_vec = np.zeros(self.num_topics)
        for k in self.topics:
            background_wordtopic_prob_vec[k] = self.background_wordtopic_prob(k, doc_id, word_id)
        wordtopic = categorical(self.num_topics, p=background_wordtopic_prob_vec / background_wordtopic_prob_vec.sum())
        return wordtopic


    def sample_background_word_topics(self):
        for doc_id, doc in tqdm(enumerate(self.docs), total=len(self.docs)):
            old_doc_type = self.old_doc_to_doctype[doc_id]
            doc_type = self.doc_to_doctype[doc_id]
            for word_id, word in enumerate(doc['doc_vec']):
                old_word_topic = self.word_to_background_topic[(doc_id, word_id)]
                ## decrement counts
                self.doctype_by_wordtopic__wordtopic_counts[old_doc_type, old_word_topic] -= 1
                self.doctype__wordtopic_counts[old_doc_type] -= 1
                self.vocab_by_wordtopic__word_counts[word, old_word_topic] -= 1
                self.wordtopic__word_counts[old_word_topic] -= 1
                ## sample new word topic
                new_word_topic = self.propose_new_background_wordtopic(doc_id, word_id)
                ## increment counts
                self.doctype_by_wordtopic__wordtopic_counts[doc_type, new_word_topic] += 1
                self.doctype__wordtopic_counts[doc_type] += 1
                self.vocab_by_wordtopic__word_counts[word, new_word_topic] += 1
                self.wordtopic__word_counts[new_word_topic] += 1
                ## cache
                self.word_to_background_topic[(doc_id, word_id)] = new_word_topic

    def sample_pass(self):
        print('sampling doc-type...')
        self.sample_doctype()
        print('sampling source-type...')
        self.sample_source_type()
        print('sampling background word topic...')
        self.sample_background_word_topics()
        print('sampling word-topic...')
        self.sample_source_word_topic()


    def joint_probability(self):
        pass


if __name__=="__main__":
    import argparse; p=argparse.ArgumentParser()
    # model params
    p.add_argument('-i', type=str, help="input directory.")
    p.add_argument('-o', type=str, help="output directory.")
    p.add_argument('-k', type=int, help="num topics.")
    p.add_argument('-p', type=int, help="num personas.")
    p.add_argument('-t', type=int, help="num iterations.")
    p.add_argument('--use-cached', default='store_true', dest='use_cached', help='use intermediate cached file.')
    args = p.parse_args()

    here = os.path.dirname(__file__)
    input_documents_fp = os.path.join(here, args.i, 'doc_source.json')
    with open(input_documents_fp) as f:
        doc_strs =  f.read().split('\n')
        docs = []
        for doc_str in doc_strs:
            if doc_str:
                doc = json.loads(doc_str)
                docs.append(doc)

    vocab_fp = os.path.join(args.i, 'vocab.txt')
    vocab = open(vocab_fp).read().split('\n')

    use_labels = True
    if use_labels:
        roles_fp = os.path.join(args.i, 'roles.txt')
        roles = open(roles_fp).read().split('\n')
        sampler = BOW_Source_GibbsSampler(docs=docs, vocab=vocab, num_source_types=len(roles), use_labels=True)
    else:
        sampler = BOW_Source_GibbsSampler(docs=docs, vocab=vocab, use_labels=False)

    ##
    if not args.use_cached:
        sampler.initialize()
        pickle.dump(sampler, open('trained-sampled-iter-0.pkl', 'wb'))

    else:
        print('loading...')
        files = glob.glob('trained-sampled-iter*')
        max_file = max(files, key=lambda x: int(re.findall('iter-(\d+)', x)[0]))
        sampler = pickle.load(open(max_file, 'rb'))

    for i in tqdm(args.t):
        if i % 10 == 0:
            pickle.dump(sampler, open('trained-sampled-iter-%d.pkl' % i, 'wb'))
        sampler.sample_pass()

    ## done 
    pickle.dump(sampler, open('trained-sampler-with-labels.pkl', 'wb'))
