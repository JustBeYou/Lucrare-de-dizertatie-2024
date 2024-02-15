#!/usr/bin/env bash

conda env export > environment.yml
conda env export --from-history > environment-explicit.yml