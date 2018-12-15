#!/usr/bin/env sh
network='res18'
now=$(date +"%Y%m%d_%H%M%S")
echo $network
echo $now
python main.py --experiment=voc --approach=joint_train --epochs 15 --network=$network --time=$now --pretrain  \
2>&1|tee ../res/log/voc-joint_train-$network-$now.log 

