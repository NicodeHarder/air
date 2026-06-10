#!/bin/bash
# Create the project virtual environment and install dependencies. Python 3.10/3.11.
set -e
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate
echo
echo "Done. Next: cp .vscode/.env.example .vscode/.env  (fill in HF_TOKEN / RESULTS_DIR), then 'source env/bin/activate'."
