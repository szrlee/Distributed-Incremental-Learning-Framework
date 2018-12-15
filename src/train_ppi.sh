#!/usr/bin/env sh
appr=ppi
network='res18'
now=$(date +"%Y%m%d_%H%M%S")
echo $network
echo $now
python main.py --experiment=voc --approach=$appr --epochs 15 --network=$network --time=$now --pretrain  \
2>&1|tee ../res/log/voc-$appr-$network-$now.log 

