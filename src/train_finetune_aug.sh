#!/usr/bin/env sh
network='res18'
now=$(date +"%Y%m%d_%H%M%S")
appr='finetune_aug'
echo $network
echo $now
python main.py --experiment=voc --approach=$appr --epochs 15 --time=$now --network=$network --pretrain  \
2>&1|tee ../res/log/voc-$appr-$network-$now.log 
