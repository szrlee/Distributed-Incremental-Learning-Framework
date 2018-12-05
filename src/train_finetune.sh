#!/usr/bin/env sh
now=$(date +"%Y%m%d_%H%M%S")
python main.py --experiment=voc --approach=fine_tuning --epochs 20  \
2>&1|tee ../res/log/voc-fine_tuning-$now.log 

