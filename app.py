import os
import pathlib
import subprocess
import waitress
from functools import partial
from typing import List
from flask import abort, Flask, jsonify, request
from . import PYTHON, ROOT
from segmentation.ncrffpp import NCRFpp

RUN_GENERATION = partial(
    '{python} generate.py '
    '--data {root}/data/chukchi_chars/ '
    '--checkpoint {root}/data/chukchi_model.pt'
    '--input {tokens}'
    '--cuda --words {words}'.format,
    python=PYTHON,
    root=ROOT,
)

app = Flask(__name__)


def tokenize(input_text: str) -> List[str]:
    ncrffpp = NCRFpp('', '', '', '')
    return ncrffpp.decode('')


def get_prediction(input_text: str) -> str:
    tokens = tokenize(input_text)
    generate_proc = subprocess.run(
        RUN_GENERATION(tokens=",".join(tokens), words=5),
        shell=True, stdout=subprocess.PIPE,
    )
    result = generate_proc.stdout.decode('utf-8')
    return result


@app.route('/health')
def health():
    return jsonify({'success': 'story'})


@app.route('/get_suggestions', methods=['POST'])
def get_suggestions():
    input_text = request.json.get('text')
    if not input_text:
        abort(405, 'В запросе нет поля `text` или оно пустое')
    result = get_prediction(input_text)
    return jsonify({'suggestions': [result]})


waitress.serve(app, listen='127.0.0.1:5000')
