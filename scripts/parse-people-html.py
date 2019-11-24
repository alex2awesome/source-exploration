from collections import defaultdict
from tqdm import tqdm
import xml.etree.ElementTree as ET
import os, glob, copy, json, re
import parsing_util
import numpy as np
import pickle
import pandas as pd
from more_itertools import unique_everseen

def full_groupby(l, key=lambda x: x):
    d = defaultdict(list)
    for item in l:
        d[key(item)].append(item)
    return d.items()

def parse_stanford(stanford_files):
    """
    Process Stanford XMLs and identify where the speaking verbs are.

    Return source_sentences, which is a list of dictionaries, each of which is:
    {
        "source_sentences": sentences per source, with source and speaking verb tagged with <SOURCE> and <VERB>.
        "doc_sentences": list of lists, for all sentences of the document. Use this to construct summary.
        "doc_id": fn,
    }
    :param stanford_files: List of filenames where the stanford files live.
    :return: source_summary_output, source_sentences_output
    """
    source_summary_output = []
    source_sentences_output = []
    for idx, stanford_parse in tqdm(enumerate(stanford_files), total=len(stanford_files)):
        ## filename
        fn = os.path.basename(stanford_parse).replace('.txt.xml', '')
        data = open(stanford_parse, encoding='utf-8').read()
        doc_etree = ET.fromstring(data)
        ## parse
        sents_list, named_ents_list, lemmas, pos_tags, deps = parsing_util.parse_etree(doc_etree)
        stanford_corefs = parsing_util.convert_coref(doc_etree=doc_etree, sentences=sents_list)
        ## group
        named_entity_clusters = parsing_util.extract_and_merge_ners(named_ents_list, sents_list, stanford_corefs)
        named_entity_clusters = parsing_util.extract_quotes_for_each_ner(named_entity_clusters, lemmas, pos_tags, deps)

        ## get source sentences
        long_named_entities = list(filter(lambda x: len(x[1]['speaking_vbs']) > 0, named_entity_clusters.items()))
        source_sentences = defaultdict(lambda: defaultdict(list))
        for source_name, mentions in long_named_entities:
            for sentence_idx, sentence_mentions in full_groupby(mentions['mentions'], key=lambda x: x.sentence):
                sentence = copy.copy(sents_list[sentence_idx])
                sentence_mentions = list(sentence_mentions)
                ## tag source heads with <SOURCE> tags.
                for m in sentence_mentions:
                    sentence = (
                        sentence[:m.head]
                        + ['<SOURCE>' + sentence[m.head] + '</SOURCE>']
                        + sentence[m.head + 1:]
                    )

                ## tag the speaking with <VERB> tags.
                speaking_verbs_in_sentence = list(filter(lambda x: x['sentence'] == sentence_idx, mentions['speaking_vbs']))
                for vb in speaking_verbs_in_sentence:
                    sentence = (
                        sentence[:vb['verb_idx']]
                        + ['<VERB>' + sentence[vb['verb_idx']] + '</VERB>']
                        + sentence[vb['verb_idx'] + 1:]
                    )

                ## clean up for double-mentions
                sentence = ' '.join(sentence).replace('<SOURCE><SOURCE>', '<SOURCE>')
                sentence = sentence.replace('</SOURCE></SOURCE>', '</SOURCE>')
                ## cache
                source_sentences[source_name]['sentences'].append(sentence)
                source_sentences[source_name]['sentence_ids'].append(sentence_idx)

        source_sentences_output.append({
            "source_sentences": source_sentences,
            "doc_sentences": list(sents_list),
            "doc_id": fn,
        })

        ## record
        summ = list(map(lambda x: {
            'fn': fn,
            'name': x[0],
            'num_quotes': len(x[1]['speaking_vbs']),
            'num_mentions': len(x[1]['mentions'])
        }, named_entity_clusters.items()))
        source_summary_output.append(summ)

    return source_sentences_output, source_summary_output

def attach_labels(source_sentences, doc_source, sampler, roles):
    """Take HTML output and attach the labels given by the topic model."""

    ## construct mappers
    docid2idx_map = list(map(lambda x: x['doc_id'], doc_source))
    doc_to_source_map = {}
    for doc in sampler.docs:
        if doc['doc_id'] not in doc_to_source_map:
            doc_to_source_map[doc['doc_id']] = doc['source_map']

    source_to_sourcetype_df = (pd.Series(sampler.source_to_source_type)
        .apply(lambda x: roles[x])
        .reset_index()
        .rename(columns={'level_0': 'doc_idx', 'level_1': 'source_id', 0: 'source_role'})
        .assign(doc_id=lambda df: df['doc_idx'].apply(lambda x: docid2idx_map[x]))
        .assign(source_name=lambda df: df.apply(lambda x: doc_to_source_map[x['doc_id']].get(x['source_id']), axis=1))
        .loc[lambda df: df['source_name'].notnull()]
    )

    ## map
    for datapoint in source_sentences:
        doc_id = datapoint['doc_id']
        for source_name in datapoint['source_sentences'].keys():
            source_df = (source_to_sourcetype_df
                .loc[lambda df: df['doc_id'] == doc_id]
                .loc[lambda df: df['source_name'] == source_name]
            )
            source_role = ''
            if len(source_df) > 0:
                source_role = source_df.iloc[0]['source_role']

            datapoint['source_sentences'][source_name]['label'] = source_role

    return source_sentences

def format_output(source_sentences, attach_labels=False):
    """
    Take the parsed texts and format them for HTML representation.

    :param source_sentences: output of `parse_stanford`
    :return:
    """

    ## apply labels
    html_output = []
    for idx, datapoint in enumerate(source_sentences):
        doc_id = datapoint['doc_id']

        ## construct lead sentences of article.
        doc_sents = list(map(lambda x: ' '.join(x).strip(), datapoint['doc_sentences']))
        doc_sents = list(unique_everseen(doc_sents))
        doc_sents = ' '.join(doc_sents[:2]) + '...'

        for source_name, source_data in datapoint['source_sentences'].items():
            sentences_str = list(map(lambda x: x.strip(), source_data['sentences']))
            sentences_str = list(filter(lambda x: x[0] < 5 or '<VERB>' in x[1], enumerate(sentences_str)))
            sentences_str = list(map(lambda x: x[1], sentences_str))
            first_sentences = ' ... '.join(sentences_str)

            ## replace person
            first_sentences = (first_sentences
                   .replace('<SOURCE>', '<span style="background-color: #FFFF00">')
                   .replace('</SOURCE>', '</span>')
            )
            ## replace verb
            first_sentences = (first_sentences
                   .replace('<VERB>', '<span style="background-color: #ADD8E6">')
                   .replace('</VERB>', '</span>')
                   )

            html = (
                '<h3>source: ' + source_name + '</h3>' +
                '<b>article\'s lead paragraph:</b><br>' + doc_sents +
                '<br><br><b>source\'s sentences:</b><br>' + first_sentences
            )

            if attach_labels:
                label_str = source_data['label']
                html += (
                    '<br><br><h4><b>model-applied label:</b> <span style="background-color: #F9C1AF">'
                    + label_str + '</span></h4>'
                )

            html_output.append({
                "doc_id": doc_id,
                "person": source_name,
                'sentence_ids': sorted(source_data['sentence_ids']),
                "html": html
            })

    return html_output


if __name__ == '__main__':
    import argparse; p=argparse.ArgumentParser()
    # model params
    p.add_argument('-i', type=str, help="input directory.")
    p.add_argument('-o', type=str, default='', help="output directory.")
    p.add_argument('-m', type=str, default='', help="model dir.")
    p.add_argument('--use-full-paths', dest='full_paths', action='store_true', default=False, help="Whether to use relative paths or full paths for I/O.")
    p.add_argument('--full-source-text', dest='full_source_text', action='store_true', default=False, help="True -- include the full text of each speaker. False -- store only first sentence and quote.")
    p.add_argument('--full-doc-text', dest='full_doc_text', action='store_true', default=False, help="True -- include the full text of each document. False -- exclue text associated with speakers.")
    p.add_argument('--attach-labels', dest='attach_labels', action='store_true', default=False, help='Attach labels from model.')
    p.add_argument('--convert-to-html', dest='make_html', action='store_true', default=False, help='Whether to convert to HTML.')
    args = p.parse_args()

    here = os.path.dirname(__file__)
    if args.full_paths:
        here = ''
    source_data_dir = os.path.join(here, args.i)
    if not args.o:
        args.o = args.i
    output_data_dir = os.path.join(here, args.o)
    if args.attach_labels:
        model_dir = os.path.join(here, args.m)
        import sys
        sys.path.append(model_dir)
        from sampler import BOW_Source_GibbsSampler

    ##
    stanford_dir = os.path.join(source_data_dir, 'stanford-parses')
    processed_text_dir = os.path.join(output_data_dir, 'html-for-sources')
    # processed_text_dir = os.path.join(output_data_dir, 'model-labeled-sources')

    if not os.path.exists(processed_text_dir):
        os.makedirs(processed_text_dir)

    doc_outfile = os.path.join(processed_text_dir, 'doc_html.json')
    ## parse stanford and extract sources

    print('parsing stanford...')
    stanford_files = glob.glob(os.path.join(stanford_dir, '*', '*'))
    parsed_texts, summaries = parse_stanford(stanford_files=stanford_files)

    if args.attach_labels:
        print('attaching labels...')
        ## read sampler
        model_files = glob.glob(os.path.join(model_dir, 'trained-sampled-iter*'))
        model_file = max(model_files, key=lambda x: int(re.findall('iter-(\d+)', x)[0]))
        print('loading model file...')
        sampler = pickle.load(open(model_file, 'rb'))
        print('done loading model file...')
        ## read roles
        roles = open(os.path.join(model_dir, 'input_data', 'roles.txt')).read().split('\n')
        roles[-1] = 'other'
        ## read input docs to get id mapping
        doc_source = sampler.docs ## might want to change this to the model files aren't so big...
        ##
        parsed_texts = attach_labels(source_sentences=parsed_texts, sampler=sampler, roles=roles, doc_source=doc_source)

    print('writing...')
    if args.make_html:
        html_output = format_output(parsed_texts, attach_labels=args.attach_labels)
        with open(doc_outfile, 'w') as f:
            for html in html_output:
                f.write(json.dumps(html))
                f.write('\n')

    else:
        with open(doc_outfile, 'w') as f:
            for doc in parsed_texts:
                f.write(json.dumps(doc))
                f.write('\n')