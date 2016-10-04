__author__ = 'Atanas'

from flask import Flask, jsonify
from flask import request
from lda import lda_process, data_preparation

app = Flask(__name__)


@app.route('/')
def hello():
    return "Hello"

@app.route('/lda', methods=['POST'])
def lda():
    content = request.get_json(silent=True)
    documents = content.get("documents")
    num_topics = content.get("num_topics")
    passes = content.get("passes")
    num_words = content.get("num_words")
    normalization_type = content.get("normalization_type")
    return jsonify(lda_process(documents, num_topics, passes, num_words, normalization_type))


@app.route('/preprocess', methods=['POST'])
def data_preprocess():
    content = request.get_json(silent=True)
    documents = content.get("documents")
    type = content.get("type")
    return jsonify(data_preparation(documents, type))
    #return str(data_preparation(documents, type))


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


if __name__ == "__main__":
    app.run()