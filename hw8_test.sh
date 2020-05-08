#!/bin/bash

if [ $# -ne 2 ]; then
  echo -e "usage:\tbash hw8_test.sh [input_directory] [output_path]"
  exit
fi

python3 hw8_test.py $1 $2 model_full.pth
