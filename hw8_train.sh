#!/bin/bash

if [ $# -ne 1 ]; then
  echo -e "usage:\tbash hw8_best.sh [input_directory]"
  exit
fi

python3 hw8_train.py $1 linear model_linear.pth log_linear.log
