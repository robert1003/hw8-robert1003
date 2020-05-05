#!/bin/bash

if [ $# -ne 1 ]; then
  echo -e "usage:\tbash hw8_test.sh [input_directory]"
  exit
fi

python3 hw8_test.py $1 model_full.pth
