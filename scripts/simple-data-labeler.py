import pickle
import pandas as pd 
import os
all_quotes_df = pickle.load(open('../data/2019-09-16__parse-df-method-1.pkl', 'rb'))

## get the sentence with the full name 
usable_people = (
    all_quotes_df
        .reset_index()
        .loc[lambda df: df['index'].str.split(' ').str.len() > 1]
)

## get the main sentences
main_sentences = (
    usable_people
        .apply(lambda x: (
            x['index'],
            list(filter(lambda y: x['index'] in y[1], x['background sentence'] + x['quote sentence'])),
            x['background sentence'] + x['quote sentence']
        ), axis=1)
)

## apply labels
labels = []
for sentences in main_sentences:
    person, first_sents, sents = sentences 
    first_sentences = ' '.join(list(map(lambda x: x[1], first_sents)))
    next_sentences = ' '.join(list(map(lambda x: x[1], sents[:3]))) + '...'
    label = input('person: ' + person + '\n\nmain sentence:\n' + first_sentences + '\n\nothers\n' + next_sentences)
    labels.append(label)
    os.system('cls' if os.name == 'nt' else 'clear')