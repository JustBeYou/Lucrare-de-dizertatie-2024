#!/usr/bin/env bash

set -e -x

python3 -m isort dizertatie/ tests/
python3 -m black dizertatie/ tests/
