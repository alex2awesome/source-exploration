from flask import Flask, render_template, request

import json

app = Flask(__name__)


@app.route('/')
def hello():
    return "Hello World!"

@app.route('/render')
def render():
	input_json = json.load(open('data/batch-unmarked-4.json'))
	num_sources = len(input_json)
	num_docs = len(set(map(lambda x: x['doc_id'], input_json)))
	return render_template('task.html', sources_count=num_sources, paper_count=num_docs, input=input_json)

@app.route('/post', methods=['POST'])
def post():
	output_data = request.get_json()
	## 
	crowd_data = output_data['data']
	json.dump(crowd_data, open('data/batch-marked-4.json', 'w'))

	return "success"


if __name__ == '__main__':
    app.run(debug=True, port=5001)