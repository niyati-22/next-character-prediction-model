#!/usr/bin/env bash
set -e
set -v
python src/modularizedLSTM.py test --work_dir work --test_data $1 --test_output $2
