from util import clean_multiword_phrases, get_nlp
from params import speaking_lexicon, ner_list


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
        sent, text_sentence = clean_multiword_phrases(sent, text_sentence)

        ## extract all nsubj of VERB if VERB is 'said', 'says' or 'say'
        for possible_subject in sent:
            if (
                    possible_subject.dep_ == 'nsubj' and
                    possible_subject.head.pos_ == 'VERB' and
                    possible_subject.head.text in speaking_lexicon
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
    # 1. extract all quote sentences
    in_quote = False
    word_idx_in_quote = []
    for word_idx, word in enumerate(doc):
        if unidecode(word.text) == '"':
            in_quote = bool((in_quote + 1) % 2)
        if in_quote:
            word_idx_in_quote.append(word_idx)

    # 2. get sent idx of quote sentences
    quote_sent = list(map(lambda x: doc[x].sent, word_idx_in_quote))
    quote_sent = list(unique_everseen(quote_sent))
    quote_sent_idxs = list(map(lambda x: get_sent_idx_from_sent(x, doc), quote_sent))

    # 3.
    new_quote_sent_chunks = deepcopy(already_extracted_quote_chunks)
    if not isinstance(new_quote_sent_chunks, list):
        new_quote_sent_chunks = new_quote_sent_chunks.tolist()
    seen_set = flatten_list_of_lists(new_quote_sent_chunks)
    seen_set = set(map(lambda x: x[0], seen_set))
    adjacent_candidates = list(set(quote_sent_idxs) - seen_set)

    # 4. First, attribute candidates to the source preceeding them, if any.
    for i in adjacent_candidates:
        for q_idx, q in enumerate(new_quote_sent_chunks):
            for (q_sent_idx, q_span) in q:
                if i == q_sent_idx + 1:
                    new_quote_sent_chunks[q_idx].append((i, copy(q_span)))

    # 5. Next, attribute candidates to the source succeeding them, if any.
    for i in adjacent_candidates:
        for q_idx, q in enumerate(new_quote_sent_chunks):
            for (q_sent_idx, q_span) in q:
                if i == q_sent_idx - 1:
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
            if ent.label_ in ner_list:
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
