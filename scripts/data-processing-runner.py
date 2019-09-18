from tqdm import tqdm as tqdm
import pandas as pd 
import util
import spacy
import pickle
import argparse 
import logging

nlp = spacy.load('en_core_web_lg')

def process_one_body(row, nlp=nlp):
    """Take one body and extract the people with their descriptions."""
    article_id, body = row
    output = []
    doc = util.preprocess(body, nlp)
    entities = util.get_quotes_method_1(doc)
    if len(entities)> 0:
        # quoted_entities = dict(filter(lambda x: len(x[1]['quote sentence']) > 0, entities_clustered.items()))
        entity_df = pd.DataFrame.from_dict(entities, orient='index')
        quoted_ent_df = entity_df.loc[lambda df: df['quote sentence'].str.len() > 0]
        quoted_ent_df['article_id'] = article_id
        output.append(quoted_ent_df)
    return output

if __name__=='__main__':
    ## argparse
    parser = argparse.ArgumentParser(description='Parse out the data.')
    parser.add_argument('--start', type=int, help='start row')
    parser.add_argument('--end',  type=int, help='end row')
    args = parser.parse_args()

    # ## logger
    # log = logging.getLogger(__name__)
    # fh = logging.FileHandler('logs/process-logger__start-%d_end-%d.log' % (args.start, args.end))
    # fh.setLevel(logging.DEBUG)
    # log.setLevel(logging.INFO)
    # log.addHandler(TqdmLoggingHandler())


    ## i/o paths
    path_to_data_input = '../data/a1_df.csv'
    path_to_data_output = '../data/2019-09-16__parse-df-method-1__start-%d__end-%d.pkl'

    ## load data
    a1_df = pd.read_csv(
        path_to_data_input,
        index_col=0,
        header=-1,
        squeeze=True,
        skiprows=args.start,
        nrows=(args.end - args.start)
    )

    ## iterate
    quoted_dfs_method_1 = []
    for output in tqdm(util.multiprocess(a1_df.iteritems(), process_one_body, max_workers=8), total=len(a1_df)):
        if len(output)> 0:
            quoted_dfs_method_1.extend(output)

    ## concat and output
    all_quotes_df = pd.concat(quoted_dfs_method_1)
    pickle.dump(
        all_quotes_df,
        open(path_to_data_output % (args.start, args.end), 'wb')
    )