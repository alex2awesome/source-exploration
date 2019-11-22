import pandas as pd
import glob, os, sys
from tqdm import tqdm
import parsing_util
from sklearn.feature_extraction.text import CountVectorizer
import json, pickle
from collections import defaultdict

def parse_sources_from_texts(
        stanford_input_dir,
        output_dir,
        include_all_source_mentions=False,
        include_all_sentences_in_doc=False
):
    """
    Takes as input the directory of stanford CoreNLP parses and finds sources. Writes to the output directory and returns the flatlist.

    :param stanford_input_dir: directory containing the Stanford XML parses.
    :param output_dir: directory to write the sources/document that was parsed.
    :param include_all_source_mentions:
    :param include_all_sentences_in_doc:
    :return:


    Format of json output for each doc:

    """
    stanford_parses = glob.glob(os.path.join(stanford_input_dir, '*', '*'))
    parsed_texts = []
    for xml_file in tqdm(stanford_parses):
        try:
            ## parse
            people_and_doc = parsing_util.parse_people_and_docs(
                xml_file,
                include_all_mentions=include_all_source_mentions,
                include_all_sentences_in_doc=include_all_sentences_in_doc
            )
            ## filter
            if len(people_and_doc['source_sentences']) > 0:
                doc_id = people_and_doc['doc_id']
                parsed_texts.append(people_and_doc)
                ## maintain recursive structure
                folder_id = os.path.basename(os.path.dirname(xml_file))
                outpath = os.path.join(output_dir, folder_id)
                if not os.path.exists(outpath):
                    os.makedirs(outpath)
                ## cache
                json.dump(people_and_doc, open(os.path.join(outpath, doc_id + '.json'), 'w'))
        except:
            continue

    ## return
    return parsed_texts

### build count vectors
def build_count_vectors(parsed_texts, source_min_df=.001, source_max_df=.5, doc_min_df=.01, doc_max_df=.5):
    """
    Takes a list of parsed people/docs and constructs a combined vocabulary. Uses different cutoffs for source vocabulary vs. document vocabulary.

    :param parsed_texts:
    :param source_min_df:
    :param source_max_df:
    :param doc_min_df:
    :param doc_max_df:
    :return:
    """
    source_sentences = []
    doc_sentences = []
    for text in parsed_texts:
        for name, source_text in text['source_sentences'].items():
            source_sentences.append(source_text)
        doc_sentences.append(text['doc_sentences'])

    doc_cv = CountVectorizer(min_df=doc_min_df, max_df=doc_max_df, stop_words='english')
    source_cv = CountVectorizer(min_df=source_min_df, max_df=source_max_df, stop_words='english')
    ###
    doc_cv.fit(doc_sentences)
    source_cv.fit(source_sentences)
    ###
    combined_vocab = pd.concat([
        pd.Series(source_cv.vocabulary_).reset_index()['index'],
        pd.Series(doc_cv.vocabulary_).reset_index()['index']
    ]).drop_duplicates().reset_index(drop=True).reset_index().set_index('index').iloc[:, 0].to_dict()

    full_cv = CountVectorizer(vocabulary=combined_vocab)
    return full_cv

###
# read in roles
def read_roles(parsed_texts, marked_file_dir, role_outfile):
    """
    Reads in labels applied by humans and maps them to doc_id/source_id for lookup later in processing.

    :param parsed_texts:
    :param marked_file_glob:
    :param role_outfile:
    :return:
    """
    ## read in all tagged files
    tagged_files = glob.glob(os.path.join(marked_file_dir, '*'))
    tags = []
    for f in tagged_files:
        tags.extend(json.load(open(f)))

    ## load them into a dataframe and parse out nonlegit sources.
    tags_df = pd.DataFrame(tags)
    legit_tagged_sources = (
        tags_df
            .groupby(['doc_id', 'person'])[['question_class', 'label']]
            .apply(lambda df: dict(df.itertuples(index=False)))
            .apply(pd.Series)
            .fillna('')
            .loc[lambda df: df['error'] == 'legit']
    )
    legit_tagged_sources = (
        legit_tagged_sources
            .reset_index()
            .assign(person=lambda df: df['person'].str.replace('-', ' '))
            .set_index(['doc_id', 'person'])
    )
    role = (
        legit_tagged_sources
            [list(filter(lambda x: '-role' in x, legit_tagged_sources.columns))]
            .apply(lambda x: x[x != ''][0], axis=1)
    )
    affil = legit_tagged_sources['affiliation']
    legit_tags = affil + '-' + role

    ## cache role mapping
    with open(role_outfile, 'w') as f:
        for tag in legit_tags.unique():
            f.write(tag)
            f.write('\n')

    label2l_id_map = {v: k for k,v in enumerate(legit_tags.unique())}

    ## map document list to dict, {doc_id: data_chunk}
    doc_idx_to_chunks = {}
    for text_json in parsed_texts:
        doc_idx_to_chunks[text_json['doc_id']] = text_json

    ## map labels to doc_id/source_id
    s_id2label = defaultdict(dict)
    for doc_id, person, role in legit_tags.reset_index().itertuples(index=False):
        s_id2source = doc_idx_to_chunks.get(doc_id, {}).get('source_map', {})
        source2s_id = {v:k for k,v in s_id2source.items()}
        if person in source2s_id:
            s_id = source2s_id[person]
            s_id2label[doc_id][s_id] = label2l_id_map[role]
    return s_id2label

def format_and_dump_text(
        parsed_texts, s_id2label, cv,
        doc_cutoff=200, source_cutoff=100,
        doc_source_output='doc_source.json',
        vocab_source_output='vocab.txt',
        convert_words_to_idx=True
    ):
    """
    Takes in processed steps and dumps them.

    ## processed data
    :param parsed_texts:
    :param s_id2label:
    :param cv:
    ## cutoffs
    :param doc_cutoff:
    :param source_cutoff:
    ## output files
    :param doc_source_output: single output file for all doc jsons. Line delimited, so each line is one JSON.
    :param vocab_source_output:
    :return:

    Format of JSON output for each doc:


    """
    text_output = []
    for doc_num, text in enumerate(parsed_texts):
        doc_chunk = {}
        doc_id = text['doc_id']
        if convert_words_to_idx:
            doc_chunk['doc_vec'] = parsing_util.map_words(text['doc_sentences'], cutoff=doc_cutoff, cv=cv)
            if len(doc_chunk['doc_vec']) < 4:
                continue
        else:
            doc_chunk['doc_vec'] = text['doc_sentences']

        doc_chunk['doc_id'] = doc_id

        ## configure sources
        source_map = {}
        source_vecs = {}
        for source_num, (name, source_text) in enumerate(text['source_sentences'].items()):
            source_id = 'S_%s_%s' % (doc_num, source_num)
            source_map[source_id] = name
            if convert_words_to_idx:
                source_vecs[source_id] = parsing_util.map_words(source_text, cutoff=source_cutoff, cv=cv)
            else:
                source_vecs[source_id] = source_text

        ## store source information in the document.
        doc_chunk['source_map'] = source_map
        doc_chunk['source_vecs'] = source_vecs
        doc_chunk['source_labels'] = {}
        ##
        if doc_id in s_id2label:
            doc_chunk['source_labels'] = s_id2label[doc_id]
        ##
        text_output.append(doc_chunk)

    ## write document chunks
    with open(doc_source_output, 'w') as f:
        for doc_chunk in text_output:
            f.write(json.dumps(doc_chunk))
            f.write('\n')

    ## write vocabulary
    with open(vocab_source_output, 'w') as f:
        for word in pd.Series(cv.vocabulary).index:
            f.write(word)
            f.write('\n')


if __name__=="__main__":
    import argparse; p=argparse.ArgumentParser()
    # model params
    p.add_argument('-i', type=str, help="input directory.")
    p.add_argument('-o', type=str, help="output directory.")
    p.add_argument('--use-full-paths', dest='full_paths', action='store_true', default=False, help="Whether to use relative paths or full paths for I/O.")
    p.add_argument('--use-prev-vocab', dest='use_cached_cv', action='store_true', default=False, help="Use a previously pickled CountVectorizer. Use if intermediate processing.")
    p.add_argument('--use-labels', dest='use_labels', action='store_true', default=False, help="Whether to include hand-labels in a semi-supervised manner.")
    p.add_argument('--full-source-text', dest='full_source_text', action='store_true', default=False, help="True -- include the full text of each speaker. False -- store only first sentence and quote.")
    p.add_argument('--full-doc-text', dest='full_doc_text', action='store_true', default=False, help="True -- include the full text of each document. False -- exclue text associated with speakers.")
    p.add_argument('--map-text', dest='map_text', action='store_true', default=False, help="Map text to the indexes.")
    args = p.parse_args()

    here = os.path.dirname(__file__)
    if args.full_paths:
        here = ''
    source_data_dir = os.path.join(here, args.i)
    output_data_dir = os.path.join(here, args.o)
    ##
    stanford_dir = os.path.join(source_data_dir, 'stanford-parses')
    processed_text_dir = os.path.join(source_data_dir, 'sources-and-docs-for-tm')
    label_dir = os.path.join(source_data_dir, 'labels')
    ##
    role_outfile = os.path.join(output_data_dir, 'roles.txt')
    vocab_outfile = os.path.join(output_data_dir, 'vocab.txt')
    doc_outfile = os.path.join(output_data_dir, 'doc_source.json')
    cv_outfile = os.path.join(output_data_dir, 'cv.pkl')

    ## handle dir structure

    ## check dirs
    if args.use_labels:
        check_dirs = [stanford_dir, label_dir]
    else:
        check_dirs = [stanford_dir]
    for dir_path in check_dirs:
        if not os.path.exists(dir_path):
            sys.exit('Required dir: %s not found.' % dir_path)
    ## make dirs
    for dir_path in [processed_text_dir, output_data_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    ## run parse

    ## parse stanford and extract sources
    print('parsing stanford...')
    parsed_texts = parse_sources_from_texts(
        stanford_input_dir=stanford_dir,
        output_dir=processed_text_dir,
        include_all_source_mentions=args.full_source_text,
        include_all_sentences_in_doc=args.full_doc_text
    )

    ## build vocabulary
    print('building vocab...')
    cv = None
    if args.use_cached_cv:
        cv = pickle.load(open(cv_outfile, 'rb'))
    cv = build_count_vectors(parsed_texts=parsed_texts)
    if not args.use_cached_cv:
        pickle.dump(cv, open(cv_outfile, 'wb'))

    ## read labels
    print('reading labels...')
    s_id2label = {}
    if args.use_labels:
        s_id2label = read_roles(parsed_texts=parsed_texts, marked_file_dir=label_dir, role_outfile=role_outfile)

    ## final formatting
    print('writing...')
    format_and_dump_text(
        parsed_texts=parsed_texts,
        s_id2label=s_id2label,
        cv=cv,
        doc_source_output=doc_outfile,
        vocab_source_output=vocab_outfile,
        convert_words_to_idx=not args.map_text
    )