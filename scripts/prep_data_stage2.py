import re
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import glob
CLEANR = re.compile('<.*?>')


def cleanhtml(raw_html):
    cleantext = re.sub(CLEANR, '', raw_html)
    return cleantext


def get_combined_df(annotated_fn, input_fn):
    json_dat = json.load(open(annotated_fn))['data']
    if isinstance(json_dat, dict) and 'row_data' in json_dat:
        json_dat = json_dat['row_data']

    #
    annot_df = pd.DataFrame(json_dat)
    annot_df = annot_df.applymap(lambda x: x['field_value'] if isinstance(x, dict) else x)

    #
    input_dat = json.load(open(input_fn))['html_data']
    input_df = pd.DataFrame(input_dat)

    annot_df_with_input = (
        input_df[['sent', 'sent_idx']]
            .merge(annot_df[['row_idx', 'head', 'quote_type', 'source_type']], left_on='sent_idx', right_on='row_idx')
            .drop(['row_idx', ], axis=1)
    )
    return annot_df_with_input


def split_into_train_test(doc_df):
    """
        Split the document set into training/test sets.
        Append other useful information (`entry_id`, `sent`, `head`) as columns.
    """
    entry_ids = doc_df['doc_id'].unique().tolist()
    train_files, test_files = train_test_split(entry_ids)
    split_df = pd.concat([
        pd.Series(train_files).to_frame('file_id').assign(split='train'),
        pd.Series(test_files).to_frame('file_id').assign(split='test')
    ])
    return (
        doc_df
            .merge(split_df, left_on='doc_id', right_on='file_id')
            .assign(entry_id=lambda df: '/' + df['split'] + '/' + df['doc_id'])
            .assign(sent=lambda df: df['sent'].apply(cleanhtml))
            .assign(head=lambda df: df.apply(lambda x: x['head'] if x['quote_type'] not in ['BACKGROUND', 'NARRATIVE'] else '', axis=1))
            .assign(head=lambda df: df['head'].fillna('None').apply(lambda x: {'':'None'}.get(x, x)))
            [['sent', 'head', 'sent_idx', 'entry_id']]
    )


def get_all_annot_input_file_pairs(annot_fps, checked_fps, input_fps):
    """
    Reads in and matches annotated and input files to produce the desired output.
    Also checks if we have checked these files.

    Params:
    * `annot_fps`: a list of filepaths for all annotated files we have on our fp.
    * `checked_fps`: a list of filepaths for all annotated files we checked, on our fp.
    * `input_fps:` the filepaths for all the input files we did annotation on.
    """
    all_sources = []

    # get annotated/input filepath mapper
    annotated_fns = list(map(lambda x: x.split('/')[-1].replace('to-annotate', 'annotated'), input_fps))
    annot_input_mapper = dict(zip(annotated_fns, input_fps))

    # get annotated/checked filepath mapper
    annotated_fns = list(map(lambda x: x.split('/')[-1].replace('checked', 'annotated'), checked_fps))
    annot_checked_mapper = dict(zip(annotated_fns, checked_fps))

    # match files and combine them
    for annot_fp in annot_fps:
        doc_id = re.search('\d+', annot_fp.split('/')[-1])[0]
        annot_fn = annot_fp.split('/')[-1]
        input_fp = annot_input_mapper[annot_fn]
        annot_fp = annot_checked_mapper.get(annot_fn, annot_fp)

        # combine
        annot_df_w_input = get_combined_df(annot_fp, input_fp)
        annot_df_w_input['doc_id'] = doc_id
        all_sources.append(annot_df_w_input)
    return all_sources


if __name__ == "__main__":
    input_files = glob.glob('../app/data/input_data/*/*')
    checked_files = glob.glob('../app/data/checked_data_affil-role/*/*')
    alex_annotated_files = glob.glob('../app/data/output_data_affil-role/*/*')
    james_annotated_files = glob.glob('../app/data/output_data_affil-role_james/*')

    # dedup
    alex_annotated_set = set(map(lambda x: x.split('/')[-1], alex_annotated_files))
    james_annotated_files = list(filter(lambda x: x.split('/')[-1] not in alex_annotated_set, james_annotated_files))

    all_sources = get_all_annot_input_file_pairs(
        alex_annotated_files + james_annotated_files, checked_files, input_files
    )

    all_doc_df = pd.concat(all_sources)
    all_doc_df = split_into_train_test(all_doc_df)

    out_file = '../models_neural/quote_attribution/data/our-annotated-data__stage-2.tsv'
    all_doc_df.to_csv(out_file, sep='\t', index=False, header=False)
