from flask import Flask, render_template, request
from collections import defaultdict
import re
import itertools
import random
import os
import json

app = Flask(__name__, static_url_path='/static')
basedir = os.path.abspath(os.path.dirname(__file__))


def get_client():
    from google.cloud import datastore
    try:
        return datastore.Client()
    except:
        import os
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:/Users/alexa/google-cloud/usc-research-c087445cf499.json'
        return datastore.Client()


@app.route('/')
def hello():
    return "Hello World!"

def get_user(email, client):
    """Get user. Create new entity if doesn't exist."""
    from google.cloud import datastore
    user_key = client.key('user', email)
    user = client.get(user_key)
    if user:
        return user
    e = datastore.Entity(key=user_key)
    e.update({'total_tasks': 0})
    client.put(e)
    return e

@app.route('/get_user_stats', methods=['POST'])
def get_user_stats():
    request_data = request.get_json()
    client = get_client()
    user_email = request_data.get('user_email', '')
    user = get_user(user_email, client)
    return str(user['total_tasks'])


###
# annotation
###

@app.route('/render_annotation_experiment')
def render_annotation():
    client = get_client()
    query = client.query(kind='source-annotation-unmarked')
    query.add_filter('done', '=', False)
    results = list(query.fetch(limit = 10))
    num_sources = len(results)
    num_docs = len(set(map(lambda x: x['doc_id'], results)))
    return render_template(
        'task-annotation-slim.html',
        sources_count=num_sources,
        paper_count=num_docs,
        input=results
    )

@app.route('/post_annotation_experiment', methods=['POST'])
def post():
    from google.cloud import datastore
    client = get_client()
    output_data = request.get_json()
    ##
    crowd_data = output_data['data']
    output_dict = defaultdict(list)
    for answer in crowd_data:
        doc_id = answer['doc_id']
        person = answer['person'].replace('|||', ' ')
        output_dict['%s-%s' % (doc_id, person)].append(answer)

    for key, val in output_dict.items():
        ## new
        marked_k = client.key('source-annotation-marked', key)
        marked_e = datastore.Entity(marked_k, exclude_from_indexes=['data',])
        marked_e.update({'data': val})
        client.put(marked_e)
        ## update old
        unmarked_k = client.key('source-annotation-unmarked', key)
        unmarked_e = client.get(unmarked_k)
        if unmarked_e:
            unmarked_e['done'] = True
            client.put(unmarked_e)

    return "success"


###
# validation
###

@app.route('/render_validation_experiment')
def render_validation():
    if False:
        ## fetch data
        client = get_client()
        query = client.query(kind='source-validation-unscored')
        query.add_filter('finished', '=', False)
        results = list(query.fetch(limit=3))
    else:
        results = []
        with open('../data/news-article-flatlist/html-for-sources/doc_html.json') as f:
            for line in f.readlines():
                results.append(json.loads(line.strip()))
        import numpy as np
        results = np.random.choice(results, 3).tolist()

    num_sources = len(results)
    num_docs = len(set(map(lambda x: x['doc_id'], results)))
    return render_template('task-validation.html', sources_count=num_sources, paper_count=num_docs, input=results)

@app.route('/post_validation_experiment', methods=['POST'])
def post_validation():
    output_data = request.get_json()
    ##
    crowd_data = output_data['data']
    if False:
        client = get_client()
        query = client.query(kind='source-validation-unscored')

    else:
        json.dump(crowd_data, open('data/batch-marked-t.json', 'w'))
    return "success"

import time
import glob, os
import sys
@app.route('/render_table', methods=['GET'])
def make_table_html():
    import pandas as pd
    import os

    task = request.args.get('task', 'full')
    shuffle = request.args.get('shuffle', False)
    practice = request.args.get('practice', False)
    annotator = request.args.get('annotator', 'alex')

    input_data_filepattern = os.path.join(basedir, 'data', 'input_data', '*', '*')
    to_annotate = glob.glob(input_data_filepattern)
    to_annotate = list(map(lambda x: (x, re.findall('to-annotate-\d+', x)[0]), to_annotate))
    to_annotate = sorted(map(lambda x: (x[0], x[1].replace('to-annotate', 'annotated')), to_annotate))

    output_data_filepattern = os.path.join(basedir, 'data', 'output_data_%s' % task, '*', '*')
    annotated = glob.glob(output_data_filepattern)
    annotated_files = set(map(lambda x: re.findall('annotated-\d+', x)[0], annotated))

    # filter down to remaining files
    if practice:
        to_annotate = list(filter(lambda x: x[1] in annotated_files, to_annotate))
    else:
        to_annotate = list(filter(lambda x: x[1] not in annotated_files, to_annotate))

    # shuffle the files
    if shuffle:
        random.shuffle(to_annotate)

    # filter to annotator-assigned files
    annotator_assignments = None
    annotator_assignment_fn = 'data/2022-07-08__annotator-assignments.csv.gz'
    if os.path.exists(annotator_assignment_fn):
        annotator_assignments = pd.read_csv(annotator_assignment_fn)
        annotator_assignments = (
            annotator_assignments
                .loc[lambda df: df['annotator'] == annotator]['file_id']
                .tolist()
        )
        to_annotate = list(filter(lambda x: x[0].split('-')[-1] in annotator_assignments, to_annotate))

    input = {}
    if len(to_annotate) > 0:
        while 'html_data' not in input:
            fname, file_id = to_annotate[0]
            with open(fname) as f:
                input = json.load(f)
            to_annotate = list(filter(lambda x: x[0] != fname, to_annotate))
            random.shuffle(to_annotate)

            if task == 'diversity':
                annotated_sources = list(filter(lambda x: file_id in x, annotated))
                annotated_sources = list(map(lambda x: int(re.findall('source-id-(\d+)', x)[1]), annotated_sources))
                html_data = input.pop('html_data', [])
                html_data = sorted(html_data, key=lambda x: x['source_idx'] if isinstance(x['source_idx'], int) else -1)
                groups = {}
                for k, g in itertools.groupby(html_data, key=lambda x: x['source_idx']):
                    groups[k] = list(g)
                keys = list(groups.keys())
                keys = sorted(filter(lambda x: x not in annotated_sources and x != 'null', keys))
                if len(keys) > 0:
                    input['html_data'] = groups[keys[0]]
                    file_id = '%s_%s' % (file_id, 'source-id-%s' % keys[0])

        output_fn = fname.replace('input_data', 'output_data_%s' % task)
        output_dir = os.path.dirname(output_fn)
        output_fn = os.path.join(output_dir, file_id + '.json')
        if sys.platform.startswith('win'):
            output_fn = output_fn.replace('\\', '\\\\')

        return render_template(
            'table-annotation-slim-%s.html' % task,
            data=input['html_data'],
            file_id=file_id.split('-')[1],
            entry_id=input['entry_id'],
            version=input['version'],
            label=input['label'],
            annotator=annotator,
            url=input['url'],
            headline=input['headline'],
            published_date=input['published_date'],
            do_mturk=False,
            start_time=time.time(),
            output_fname=output_fn
        )
    else:
        return "No more data."


def get_annotated_files(task):
    annotated_filepattern = os.path.join(basedir, 'data/output_data_%s/*/*' % task)
    annotated = glob.glob(annotated_filepattern)

    checked_filepattern = os.path.join(basedir, 'data/checked_data_%s/*/*' % task)
    checked = glob.glob(checked_filepattern)
    return annotated, checked


@app.route('/check_table', methods=['GET'])
def check_table():
    task = request.args.get('task', 'full')
    shuffle = request.args.get('shuffle', False)

    to_check, checked = get_annotated_files(task)
    to_check = list(map(lambda x: (x, re.findall('annotated-\d+', x)[0]), to_check))
    to_check = sorted(map(lambda x: (x[0], x[1].replace('annotated', 'checked')), to_check))
    checked_files = set(map(lambda x: re.findall('checked-\d+', x)[0], checked))
    to_check = list(filter(lambda x: x[1] not in checked_files, to_check))

    if len(to_check) > 0:
        fname, file_id = to_check[0]
        output_fn = fname.replace('output_data', 'checked_data')
        output_dir = os.path.dirname(output_fn)
        output_fn = os.path.join(output_dir, file_id + '.json')

        with open(fname) as f:
            data_to_check = json.load(f)
            data_to_check = data_to_check['data']
            if isinstance(data_to_check, dict):
                data_to_check = data_to_check['row_data']

        orig_input_fname = fname.replace('output_data_%s' % task, 'input_data').replace('annotated', 'to-annotate')
        with open(orig_input_fname) as f:
            orig_input_data = json.load(f)

        return render_template(
            'check-%s.html' % task,
            annotated_data=data_to_check,
            orig_input_data=orig_input_data['html_data'],
            entry_id=orig_input_data['entry_id'],
            version=orig_input_data['version'],
            label=orig_input_data['label'],
            url=orig_input_data['url'],
            headline=orig_input_data['headline'],
            published_date=orig_input_data['published_date'],
            do_mturk=False,
            submit=True,
            start_time=time.time(),
            output_fname=output_fn
        )
    else:
        return 'No more data.'


@app.route('/check_table_no_submit')
def check_specific_file():
    task = request.args.get('task', 'full')
    file_id = request.args.get('file_id')
    annotated, checked = get_annotated_files(task)

    if file_id is not None:
        checked_w_file_id = list(map(lambda x: (x, re.findall('\d+', x.split(os.sep)[-1])[0]), checked))
        if file_id in list(map(lambda x: x[1], checked_w_file_id)):
            to_display = list(filter(lambda x: file_id == x[1], checked_w_file_id))[0][0]
        else:
            annotated_w_file_id = list(map(lambda x: (x, re.findall('\d+', x.split(os.sep)[-1])[0]), annotated))
            if file_id in list(map(lambda x: x[1], annotated_w_file_id)):
                to_display = list(filter(lambda x: file_id == x[1], annotated_w_file_id))[0][0]
            else:
                return 'Failed to find file by file_id.'

        with open(to_display) as f:
            data = json.load(f)
            data = data['data']
            if isinstance(data, dict):
                data = data['row_data']

        orig_input_fname = (
            to_display
                .replace('output_data_%s' % task, 'input_data')
                .replace('checked_data_%s' % task, 'input_data')
                .replace('annotated', 'to-annotate')
                .replace('checked', 'to-annotate')
        )
        with open(orig_input_fname) as f:
            orig_input_data = json.load(f)

        return render_template(
            'check-%s.html' % task,
            annotated_data=data,
            orig_input_data=orig_input_data['html_data'],
            entry_id=orig_input_data['entry_id'],
            version=orig_input_data['version'],
            label=orig_input_data['label'],
            url=orig_input_data['url'],
            headline=orig_input_data['headline'],
            published_date=orig_input_data['published_date'],
            do_mturk=False,
            submit=False,
            start_time=time.time(),
        )

    else:
        return 'File incorrectly specified, no file_id provided.'




@app.route('/post_table', methods=['POST'])
def post_table_html():
    output_data = request.get_json()
    output_data['end_time'] = time.time()
    output_fname = output_data['output_fname']
    output_dir = os.path.dirname(output_fname)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_fname, 'w') as f:
        json.dump(output_data, f)
    return 'success'

if __name__ == '__main__':
    app.run(debug=True, port=5001)