#!/usr/bin/env sh
now=$(date +"%Y%m%d_%H%M%S")
python main.py --experiment=voc --approach=joint_train $3 $4 $5 $6 \
2>&1|tee ../res/log/Photo-$1$3-$now.log 

