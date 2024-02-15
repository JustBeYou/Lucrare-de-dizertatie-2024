#!/usr/bin/env bash

set -e -x

python3 -m black --check dizertatie/ tests/
python3 -m isort --check --diff dizertatie/ tests/
python3 -m pylint dizertatie/ tests/
python3 -m mypy dizertatie/ tests/
python3 -m coverage run -m unittest -v