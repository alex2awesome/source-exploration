import pandas as pd
import jellyfish
from . import utils_params as params


def get_unique_spacy_ents(ent_list):
    seen = set()
    output = []
    for ent in ent_list:
        if str(ent) not in seen:
            output.append(ent)
            seen.add(str(ent))
    return output


def clean_articles(input_str):
    input_str = input_str.lower()
    if input_str.startswith('the '):
        input_str = input_str[len('the '):]
    if input_str.startswith('a '):
        input_str = input_str[len('a '):]
    return input_str.strip()


# 1. extract candidate list from document of NERs and special-tokens
def get_source_candidates(input_doc, nlp=None):
    all_source_candidates = []
    for sent, source_head, sent_idx, _ in input_doc:
        words = list(map(lambda x: {
            'word_idx': x[0],
            'word': x[1].lower(),
            'found': False,
        }, enumerate(sent.split())))

        # A. get named entities
        ents = nlp(sent).ents
        for ent in ents:
            if ent.label_ in params.ner_filter_list:
                all_source_candidates.append({
                    'candidate': str(ent),
                    'start_word': ent.start,
                    'end_word': ent.end,
                    'sent_idx': sent_idx,
                    'type': 'named entity'
                })
                for word_idx in range(ent.start, ent.end):
                    words[word_idx]['found'] = True


        # B. get special strings (e.g. "the reporter")
        special_strs_to_search = (
            params.desired_checklist_of_anonymous_sources +
            params.desired_checklist_of_documents
        )
        for anon_source in special_strs_to_search:
            source_words = anon_source.lower().split()
            matching_start_indices = list(filter(lambda x: x['word'] == source_words[0], words))
            matching_start_indices = list(map(lambda x: x['word_idx'], matching_start_indices))

            for start_idx in matching_start_indices:
                all_matching = True
                for source_word_offset, source_word in enumerate(source_words):
                    if (source_word_offset + start_idx) < len(words):
                        sent_word_cand = words[source_word_offset + start_idx]
                        if (sent_word_cand['word'] != source_word) or (sent_word_cand['found'] == True):
                            all_matching = False
                    else:
                        all_matching = False
                if all_matching:
                    end_idx = start_idx + len(source_words)
                    for word_idx in range(start_idx, end_idx):
                        words[word_idx]['found'] = True
                    all_source_candidates.append({
                            'candidate': anon_source,
                            'start_word': start_idx,
                            'end_word': end_idx,
                            'sent_idx': sent_idx,
                            'type': 'anonymous'
                    })
    return pd.DataFrame(all_source_candidates).drop_duplicates('candidate')

def name_matching_jaro(a, c):
    c_temp, a_temp = clean_articles(c), clean_articles(a)
    if jellyfish.jaro_similarity(c_temp, a_temp) > .9:
        return True
    elif jellyfish.jaro_similarity(c_temp[::-1], a_temp[::-1]) > .9:
        return True
    elif ' %s ' % c_temp in ' %s ' % a_temp:
        return True
    elif ' %s ' % a_temp in ' %s ' % c_temp:
        return True
    else:
        return False
    # a_parts, b_parts = a.split(), b.split()

# 2. reconcile the candidate list with the list of annotations
def reconcile_candidates_and_annotations(source_cand_df, input_doc):
    doc_str = ' '.join(list(map(lambda x: x[0], input_doc)))
    candidate_set = source_cand_df['candidate'].tolist()
    candidate_set = sorted(candidate_set, key=lambda x: -len(x))  # match the longest matches first
    annotated_set = list(filter(lambda x: x != 'None', set(map(lambda x: x[1], input_doc))))
    annotation_to_candidate_mapper = {'None': 'None'}
    for a in annotated_set:
        found = False
        for c in candidate_set:
            if name_matching_jaro(a, c):
                annotation_to_candidate_mapper[a] = c
                found = True
                break
        if not found:
            assert a not in doc_str
            assert a in doc_str
    #         annotation_to_candidate_mapper[a] = a
    return annotation_to_candidate_mapper


# 3. Cache token strings for lookup
def cache_doc_tokens(input_doc, tokenizer, nlp):
    doc_tokens_by_word = []
    doc_tokens_by_sentence = []
    blank_tokens_by_sentence = []
    for sent, _, _, _ in input_doc:
        words = list(map(str, nlp(sent)))
        enc = [tokenizer.encode(x, add_special_tokens=False, add_prefix_space=True) for x in words]
        assert len(enc) == len(words)
        doc_tokens_by_word.append(enc)
        tokenized_sentence = tokenizer.encode(sent)
        doc_tokens_by_sentence.append(tokenized_sentence)
        blank_tokens_by_sentence.append([0] * len(tokenized_sentence))
    doc_tokens = [i for l in doc_tokens_by_sentence for i in l]

    return (
        doc_tokens_by_word,
        doc_tokens_by_sentence,
        blank_tokens_by_sentence,
        doc_tokens
    )


# 4. Get token-type lists for all source-candidates and for all sentences in document
def generate_indicator_lists(blank_tokens_by_sentence, doc_tokens_by_word, source_cand_df, input_doc):
    # A. Generate source indicator output
    source_indicator_output = []
    for cand, start_idx, end_idx, source_sent_idx, _ in source_cand_df.itertuples(index=False):
        source_tokens_by_word = []
        #
        for _, _, sent_idx, _ in input_doc:
            if int(sent_idx) != int(source_sent_idx):
                source_tokens_by_word.append(blank_tokens_by_sentence[int(sent_idx)])
            else:
                sent_toks = doc_tokens_by_word[int(sent_idx)]
                source_toks = []
                for w_idx in range(0, start_idx):
                    source_toks.append([0] * len(sent_toks[w_idx]))
                for w_idx in range(start_idx, end_idx):
                    source_toks.append([1] * len(sent_toks[w_idx]))
                for w_idx in range(end_idx, len(sent_toks)):
                    source_toks.append([0] * len(sent_toks[w_idx]))
                #
                source_toks = [i for s in source_toks for i in s]
                source_toks = [0] + source_toks + [0]
                source_tokens_by_word.append(source_toks)
        source_tokens_doc_level = [i for s in source_tokens_by_word for i in s]
        source_indicator_output.append(source_tokens_doc_level)

    # B. Generate sentence indicator output
    sentence_indicator_output = []
    for t_idx in range(len(blank_tokens_by_sentence)):
        one_sent_indicator = []
        for s_idx, b_sent in enumerate(blank_tokens_by_sentence):
            if s_idx == t_idx:
                one_sent_indicator.append([1] * len(b_sent))
            else:
                one_sent_indicator.append([0] * len(b_sent))

        one_sent_indicator = [i for s in one_sent_indicator for i in s]
        sentence_indicator_output.append(one_sent_indicator)

    return source_indicator_output, sentence_indicator_output


# 5. prepare lookup table
def augment_lookup_table_with_none(source_candidates_df, source_indicator_output):
    source_candidates_df = pd.concat([
        source_candidates_df,
        pd.Series({
            'candidate': 'None',
            'start_word': 0,
            'end_word': 0,
            'sent_idx': 0,
            'type': 'none',
            'source_tokenized': [0] * len(source_indicator_output[0])

        }).to_frame().T
    ])
    source_candidates_df = source_candidates_df.set_index('candidate')
    return source_candidates_df


# 7. Generate actual training data.
def generate_training_data(input_doc, annot_to_cand_mapper, source_cand_df, sentence_indicator_output, doc_tokens):
    training_data = []
    true_pairs = set()
    candidate_set = source_cand_df.index.tolist()

    # generate true-labeled data from the annotated document
    for _, annotated_source, sent_idx, _ in input_doc:
        candidate = annot_to_cand_mapper[annotated_source]
        source_ind_tokens = source_cand_df.loc[candidate]['source_tokenized']
        sentence_ind_tokens = sentence_indicator_output[int(sent_idx)]
        sentence_indicator_output.append({
            'source_ind_tokens': source_ind_tokens,
            'sentence_ind_tokens': sentence_ind_tokens,
            'doc_tokens': doc_tokens,
            'label': True
        })
        true_pairs.add((candidate, sent_idx))

    # generate false-labeled data from all other sentence/candidate pairs
    for c in candidate_set:
        for _, _, sent_idx, _ in input_doc:
            if (c, sent_idx) not in true_pairs:
                source_ind_tokens = source_cand_df.loc[c]['source_tokenized']
                sentence_ind_tokens = sentence_indicator_output[int(sent_idx)]
                sentence_indicator_output.append({
                    'source_ind_tokens': source_ind_tokens,
                    'sentence_ind_tokens': sentence_ind_tokens,
                    'doc_tokens': doc_tokens,
                    'label': False
                })

    # output
    return training_data






#                 p.EntityMention(
#                     name=str(ent),
#                     start=ent.start,
#                     end=ent.end,
#                     sentence=sent_idx,
#                     type='named entity'
#                 )

#                     p.EntityMention(
#                         name=anon_source,
#                         start=start_idx,
#                         end=end_idx,
#                         sentence = sent_idx,
#                         type='anonymous'
#                     )

