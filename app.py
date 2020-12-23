import os
import pathlib
import subprocess
import waitress
from functools import partial
from time import time_ns
from typing import List
from flask import abort, Flask, jsonify, request
from segmentation.ncrffpp import NCRFpp

HOST = '127.0.0.1:5000'
PYTHON = os.environ.get('WEB_DEVELOPMENT_PROJECT_PYTHON') or '/usr/bin/python3'
_ROOT = pathlib.Path(__file__).parent.resolve()
ROOT = str(_ROOT)
UPLOADS = _ROOT / 'uploads'
if not UPLOADS.exists():
    UPLOADS.mkdir()


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
    path = UPLOADS / str(time_ns())
    with open(path, 'w') as f:
        f.write(input_text)
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


if __name__ == '__main__':
    waitress.serve(app, listen=HOST)