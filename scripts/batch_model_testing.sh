#!/bin/bash
INPUTMODEL=$1"gen_f*"
OUTPUT=$2
mkdir -p $2

for f in $INPUTMODEL
do
	echo "Processing $f model..."
	python batch_model_test.py -g 0 --load_gen_f_model $f -e $2 -o $(basename $f) --recurrent 2
done
