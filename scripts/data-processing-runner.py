from tqdm import tqdm_notebook as tqdm
import pandas as pd 
import util
import pickle

## load data
path_to_data = '../data/a1_df.csv'
a1_df = pd.read_csv(path_to_data, nrows=100, index_col=0, header=-1, squeeze=True)

def process_one_body(article_id, body):
	"""Take one body and extract the people with their descriptions."""
	output = []
    doc = util.preprocess(body)
    entities = util.get_quotes_method_1(doc)
    if len(entities)> 0:
        # quoted_entities = dict(filter(lambda x: len(x[1]['quote sentence']) > 0, entities_clustered.items()))
        entity_df = pd.DataFrame.from_dict(entities, orient='index')
        quoted_ent_df = entity_df.loc[lambda df: df['quote sentence'].str.len() > 0]
        quoted_ent_df['article_id'] = article_id
        output.append(quoted_ent_df)
    return output

## iterate
quoted_dfs_method_1 = []
for output in tqdm(util.multiprocess(a1_df.iteritems(), process_one_body)):
    if len(output)> 0:
        quoted_dfs_method_1.extend(output)

## concat
all_quotes_df = pd.concat(quoted_dfs_method_1)
pickle.dump(all_quotes_df, open('../data/2019-09-16__parse-df-method-1.pkl', 'wb'))