#!/usr/bin/env bash

set -e -x

python3 -m black --check --diff dizertatie/ tests/
python3 -m coverage run -m unittest -v