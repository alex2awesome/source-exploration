import sqlite3
import pandas as pd
import sys
sys.path.insert(0, '../scripts/')
from scripts import rules_method_num_3
import pickle
from tqdm.auto import tqdm
import numpy as np


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_row', type=int)
    parser.add_argument('--n_rows', type=int)
    parser.add_argument('--use_csv', action='store_true')
    parser.add_argument('--output_dir', type=str, default='output')
    args = parser.parse_args()

    ##
    if not args.use_csv:
        db_path = '../data/newssniffer-nytimes.db'
        conn = sqlite3.connect(db_path)
        df = pd.read_sql('''
            SELECT entry_id, version, summary FROM 
            entryversion 
            WHERE num_versions < 40
            AND LENGTH(summary) < 9000
            LIMIT %s
            OFFSET %s
        ''' % (args.n_rows, args.start_row), con=conn)
        df['label'] = np.nan
        df = df.set_index('entry_id')
        df = df[['version', 'summary', 'label']]
        df['summary'] = df['summary'].str.replace('</p><p>', ' ')
    else:
        db_path = '../resources/data/nytimes-articles-to-extract-sources.csv'
        df = pd.read_csv(db_path, index_col=0)
        df = df.iloc[args.start_row: args.start_row + args.n_rows]
        df['version'] = 0
        df = df[['version', 'text', 'y_pred']]

    ##
    output = []
    for entry_id, (version, summary, label) in tqdm(df.iterrows(), total=len(df)):
        # try:
        quote_idxes, sent_words, _ = rules_method_num_3.perform_quote_extraction_and_clustering(summary)
        output.append({
            'entry_id': entry_id,
            'version': version,
            'quote_idxes': quote_idxes,
            'sent_parse': sent_words,
            'label': label
        })
        # except:
            # pass
    s, e = args.start_row, args.start_row + args.n_rows
    with open('%s/output_chunk__start-%s_end-%s.pkl' % (args.output_dir, s, e), 'wb') as f:
        pickle.dump(output, f)