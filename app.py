import logging
import os
import pathlib
import re
import subprocess
import tempfile
import waitress
from functools import partial
from time import time_ns
from typing import List
from flask import abort, Flask, jsonify, Response, request
from flask_cors import CORS
from segmentation.ncrffpp import NCRFpp

HOST = '192.168.1.64:5000'
PYTHON = os.environ.get('WEB_DEVELOPMENT_PROJECT_PYTHON') or '/usr/bin/python3'
_ROOT = pathlib.Path(__file__).parent.resolve()
ROOT = str(_ROOT)
UPLOADS = _ROOT / 'models' / 'ncrfpp' / 'corpus_home'
if not UPLOADS.exists():
    UPLOADS.mkdir()


RUN_GENERATION = partial(
    '{python} {root}/language_modelling/awdlstmlm/generate.py '
    '--data {root}/models/awdlstm/chukchi_segments/ '
    '--checkpoint {root}/models/awdlstm/chukchi_model.pt '
    '--input {tokens} '
    '--outf {outf} '
    '--cuda --words {words}'.format,
    python=PYTHON,
    root=ROOT,
)

app = Flask(__name__)
CORS(app)
myNCRFpp = NCRFpp(_ROOT / "models/ncrfpp/corpus_home", "ru_standard_v4", "models/ncrfpp/results", 10)


def tokenize(input_text: str) -> List[str]:
    curr_time = time_ns()
    file_name = str(curr_time)
    path = UPLOADS / file_name
    with open(path, 'w') as f:
        f.write(input_text)

    raw_file_name = f"raw_{curr_time}"
    myNCRFpp.make_raw(file_name, raw_file_name)
    decode_file_path = f"results_{curr_time}"
    decode_config_path = f"config_{curr_time}"
    myNCRFpp.load_model("model.571.model", "model.dset", decode_file_path,
                        decode_config_path, raw_file_name)
    myNCRFpp.decode(PYTHON, ROOT, decode_config_path)
    res_file_name = f"res_{curr_time}"
    myNCRFpp.convert_bmes_to_words(decode_file_path, res_file_name)
    results = myNCRFpp.convert_words_to_strings(file_name, res_file_name)
    myNCRFpp.delete_corpus_files(file_name, raw_file_name, res_file_name)
    myNCRFpp.delete_results_files(decode_config_path, decode_file_path)
    return re.split(r"[ >]", results[0])


def get_prediction(input_text: str) -> str:
    tokens = tokenize(input_text)
    tmp = tempfile.NamedTemporaryFile()
    p = subprocess.run(
        RUN_GENERATION(tokens=",".join(tokens), words=5, outf=tmp.name),
        shell=True, capture_output=True,
    )
    if p.returncode != 0:
        raise Exception('Генерация не отработала')
    with open(tmp.name) as f:
        result = f.read()
    return result


@app.after_request
def log_response(response: Response) -> Response:
    logger = logging.getLogger('waitress')
    info = ' {host} - {method} - {status}'.format(
        host=request.host, method=request.method, status=response.status)
    logger.info(info)
    return response


@app.route('/health')
def health():
    return jsonify({'success': 'story'})


@app.route('/get_suggestions', methods=['POST'])
def get_suggestions():
    input_text = request.json.get('text')
    if not input_text:
        abort(405, 'В запросе нет поля `text` или оно пустое')
    result = get_prediction(input_text)
    return jsonify({'suggestions': result.split(',')})


if __name__ == '__main__':
    logging.getLogger('waitress').setLevel(logging.INFO)
    waitress.serve(app, listen=HOST, threads=1)
