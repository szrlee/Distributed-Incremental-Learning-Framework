#!/usr/bin/env sh
now=$(date +"%Y%m%d_%H%M%S")
python main.py --experiment=voc --approach=joint_train --epochs 30  \
2>&1|tee ../res/log/voc-joint_train-$now.log 

