import pandas as pd 
import spacy
import os

def preprocess(text, nlp):
    """Remove duplicate and short sentences."""
    text = text.replace('\n', ' ').replace("''", '"').strip()
    doc = nlp(text)
    unique_sentences = list(unique_everseen(map(lambda x: x.text.strip(), doc.sents)))
    text = ' '.join(unique_sentences)
    return nlp(text)

if __name__=='__main__':
    import argparse
    nlp = spacy.load('en_core_web_lg')

    ## argparse
    parser = argparse.ArgumentParser(description='Parse out the data.')
    parser.add_argument('--start', type=int, help='start row')
    parser.add_argument('--end', type=int, help='end row')
    parser.add_argument('--batch', type=int, help="batch num.")
    args = parser.parse_args()

    ## handle I/O
    data_dir = '../../../data'
    path_to_data_input = os.path.join(data_dir, 'a1_df.csv')
    path_to_data_output = os.path.join(data_dir, 'news-article-flatlist', 'raw', args.batch)

    if not os.path.exists(path_to_data_output):
        os.mkdir(path_to_data_output)

    ## load data
    a1_df = pd.read_csv(
        path_to_data_input,
        index_col=0,
        header=-1,
        squeeze=True,
        skiprows=args.start,
        nrows=(args.end - args.start)
    )

    ## process and write
    for idx, text in a1_df.iteritems():
        processed_text = util.preprocess(text, nlp=nlp)
        with open(os.path.join(output_dir, idx + '.txt'), 'w') as f:
            f.write(processed_text.text)
