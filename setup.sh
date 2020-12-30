#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Create virtualenv
ENV_DIR=${SCRIPT_DIR}/venv
python3 -m venv ${ENV_DIR}
source ${ENV_DIR}/bin/activate

# Install python dependencies
pip install -r "${SCRIPT_DIR}"/requirements.txt

# Install non-python dependencies
git clone https://github.com/ufoym/imbalanced-dataset-sampler.git
cd imbalanced-dataset-sampler
python setup.py install

sudo apt install graphviz