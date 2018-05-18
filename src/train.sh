#!/usr/bin/env sh
now=$(date +"%Y%m%d_%H%M%S")
MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 GLOG_vmodule=MemcachedClient=-1 srun --mpi=pmi2 --gres=gpu:8 -n8 --ntasks-per-node=8  --job-name=Inc --partition=VIBackEnd1 \
python main.py --experiment=Photo --approach=$1 \
2>&1|tee ../res/log/Photo-$1-$now.log &
