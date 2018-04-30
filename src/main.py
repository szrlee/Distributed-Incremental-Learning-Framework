import multiprocessing as mp
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)

import sys,os,argparse,time
import numpy as np
import torch

import utils
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from distributed_utils import dist_init, average_gradients, DistModule

# Arguments
parser=argparse.ArgumentParser(description='Continual Learning Framework')
parser.add_argument('--experiment', default='', type=str, required=True, choices=['CelebA'], help='(default=%(default)s)')
parser.add_argument('--approach', default='', type=str, required=True, choices=['lwf', 'joint_train', 'fine_tuning'], help='(default=%(default)s)')
parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
parser.add_argument('--lr', default=0.01, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--epochs', default=7, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--batch_size', default=64, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--momentum', default=0.9, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--weight_decay', default=0.0005, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--print_freq', default=50, type=int, required=False, help='(default=%(default)d)')

########################################################################################################################

# Args -- Experiment
from dataloaders import celeba as generator

# Args -- Approach
from approaches import lwf
from approaches import joint_train
from approaches import fine_tuning

# Args -- Network
from networks import resnet as network

########################################################################################################################
def main():
    global args
    args = parser.parse_args()

    if args.approach == 'lwf':
        approach = lwf
    elif args.approach == 'joint_train':
        approach = joint_train
    elif args.approach == 'fine_tuning':
        approach = fine_tuning
    else:
        approach = None

    rank, world_size = dist_init('27777')

    if rank == 0:
        print('=' * 100)
        print('Arguments = ')
        for arg in vars(args):
            print('\t' + arg + ':', getattr(args, arg))
        print('=' * 100)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
    else: print('[CUDA unavailable]'); sys.exit()

    # Generate Tasks
    args.batch_size = args.batch_size // world_size
    Tasks = generator.GetTasks(args.approach, args.batch_size, world_size)
    # Network
    net = network.resnet50(pretrained=True).cuda()
    net = DistModule(net)
    # Approach
    Appr = approach.Approach(net, args)


    # Solve tasks incrementally
    for t in range(len(Tasks)):
        task = Tasks[t]

        if rank == 0:
            print('*'*100)
            print()
            print('Task {:d}: {:d} classes ({:s})'.format(t, task['class_num'], task['description']))
            print()
            print('*'*100)

        Appr.solve(t, Tasks)

        if rank == 0:
            print('*'*100)
            print('Task {:d}: {:d} classes Finished.'.format(t, task['class_num']))
            print('*'*100)

if __name__ == '__main__':
    main()