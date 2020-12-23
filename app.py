import os
import pathlib
import subprocess
import waitress
from functools import partial
from time import time_ns
from typing import List
from flask import abort, Flask, jsonify, request
from segmentation.ncrffpp import NCRFpp
import re

HOST = '127.0.0.1:5000'
PYTHON = os.environ.get('WEB_DEVELOPMENT_PROJECT_PYTHON') or '/usr/bin/python3'
_ROOT = pathlib.Path(__file__).parent.resolve()
ROOT = str(_ROOT)
UPLOADS = _ROOT / 'models' / 'ncrfpp' / 'corpus_home'
if not UPLOADS.exists():
    UPLOADS.mkdir()


import time

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

myNCRFpp = NCRFpp("models/ncrfpp/corpus_home", "ru_standard_v4", "models/ncrfpp/results", 10)

def tokenize(input_text: str) -> List[str]:
    curr_time = time.time_ns()
    file_name = str(curr_time)
    path = UPLOADS / file_name
    with open(path, 'w') as f:
        f.write(input_text)

    raw_file_name = f"raw_{curr_time}"
    myNCRFpp.make_raw(file_name, raw_file_name)
    decode_file_path = f"results_{curr_time}"
    decode_config_path = f"config_{curr_time}"
    myNCRFpp.load_model("model.571.model", "model.dset", decode_file_path, decode_config_path, raw_file_name)
    myNCRFpp.decode(PYTHON, ROOT, decode_config_path)
    res_file_name = f"res_{curr_time}"
    myNCRFpp.convert_bmes_to_words(decode_file_path, res_file_name)
    results = myNCRFpp.convert_words_to_strings(res_file_name)
    return re.split(r"(>|\s)", results[0])


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
    # waitress.serve(app, listen=HOST)
    print(tokenize("Амаравкэваратэн таа’койӈын."))
