import os
import pathlib

PYTHON = os.environ.get('WEB_DEVELOPMENT_PROJECT_PYTHON') or '/usr/bin/python3'
ROOT = pathlib.Path(__file__).parent.resolve()
