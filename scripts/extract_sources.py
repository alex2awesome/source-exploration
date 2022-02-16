import sqlite3
import pandas as pd
import sys
sys.path.insert(0, '../scripts/')
import coref_resolution_util

db_path = '../data/newssniffer-nytimes.db'
conn = sqlite3.connect(db_path)
df = pd.read_sql('SELECT * from entryversion limit 5', con=conn)
t = df['summary'][0].replace('</p><p>', ' ')

##
quote_idxes, sent_words, sent_lens = coref_resolution_util.perform_quote_extraction_and_clustering(t)