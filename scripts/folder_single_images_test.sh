#!/bin/bash
INPUTJPG=$1"/*.jpg"
INPUTPNG=$1"/*.png"
OUTPUT=$2
MODEL=$3
mkdir -p $2

for f in $INPUTJPG
do
  echo "Processing $f file..."
	python single_image_test.py -g 0 --load_gen_model $3 -i $f -o $2"/"$(basename $f) -r 2
done

for f in $INPUTPNG
do
  echo "Processing $f file..."
	python single_image_test.py -g 0 --load_gen_model $3 -i $f -o $2"/"$(basename $f) -r 2
done
