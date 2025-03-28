#!/usr/bin/env bash
set -x
set -e

rm -rf submit submit.zip
mkdir -p submit

# submit team.txt
printf "Sai Sunku,sais14\nNiyati Trivedi,ntriv04\nHarshitha Rebala,lhrebala" > submit/team.txt

# train model
python src/modularizedRNN.py train --work_dir work

# make predictions on example data submit it in pred.txt
python src/modularizedRNN.py test --work_dir work --test_data example/input.txt --test_output submit/pred.txt

# submit docker file
cp Dockerfile submit/Dockerfile

# submit source code
cp -r src submit/src

# submit checkpoints
cp -r work submit/work

# make zip file
zip -r submit.zip submit
