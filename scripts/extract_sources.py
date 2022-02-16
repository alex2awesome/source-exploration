import sqlite3
import pandas as pd
import sys
sys.path.insert(0, '../scripts/')
import coref_resolution_util
import pickle


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_row', type=int)
    parser.add_argument('--n_rows', type=int)
    args = parser.parse_args()

    db_path = '../data/newssniffer-nytimes.db'
    conn = sqlite3.connect(db_path)
    df = pd.read_sql('''
        SELECT entry_id, version, summary FROM 
        entryversion 
        LIMIT %s
        OFFSET %s
    ''' % (args.n_rows, args.start_row), con=conn)
    output = []
    for idx, (entry_id, version, summary) in df.iterrows():
        summary = summary.replace('</p><p>', ' ')
        quote_idxes, sent_words, _ = coref_resolution_util.perform_quote_extraction_and_clustering(summary)
        output.append({
            'entry_id': entry_id,
            'version': version,
            'quote_idxes': quote_idxes,
            'sent_parse': sent_words
        })
    s, e = args.start_row, args.start_row + args.n_rows
    with open('output/output_chunk__start-%s_end-%s.pkl' % (s, e), 'wb') as f:
        pickle.dump(output, f)