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
parser.add_argument('--experiment', default='', type=str, required=True, choices=['CelebA', 'Photo'], help='(default=%(default)s)')
parser.add_argument('--approach', default='', type=str, required=True, \
    choices=['lwf', 'joint_train', 'fine_tuning', 'gem', 'ewc', 'gmas'], help='(default=%(default)s)')
parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
parser.add_argument('--lr', default=0.01, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--epochs', default=1, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--batch_size', default=512, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--momentum', default=0.9, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--weight_decay', default=0.0005, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--print_freq', default=50, type=int, required=False, help='(default=%(default)d)')

parser.add_argument('--memory_size', default=None, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--memory_mini_batch_size', default=None, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--margin', default=1.0, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--gem_mem_strategy', default=None, type=str, required=False, choices=['direct_load', 'fixed_gpu_cache'], help='(default=%(default)s)')

########################################################################################################################

# Args -- Experiment
from dataloaders import celeba
from dataloaders import photo

# Args -- Approach
from approaches import lwf
from approaches import joint_train
from approaches import fine_tuning
from approaches import gem
# Args -- Network
from networks import mobilenet as network

########################################################################################################################
def main():
    global args
    args = parser.parse_args()

    if args.experiment == 'CelebA':
        generator = celeba
    elif args.experiment == 'Photo':
        generator = photo
    else:
        generator = None

    # TODO model arguments module should be more easy to write and read
    if args.approach == 'lwf':
        approach = lwf
        assert(args.memory_size is None)
        assert(args.memory_mini_batch_size is None)
    elif args.approach == 'joint_train':
        approach = joint_train
        assert(args.memory_size is None)
        assert(args.memory_mini_batch_size is None)
    elif args.approach == 'fine_tuning':
        approach = fine_tuning
        assert(args.memory_size is None)
        assert(args.memory_mini_batch_size is None)
    elif args.approach == 'gem':
        approach = gem
        assert(args.memory_size is not None)
        assert(args.memory_mini_batch_size is None)
    else:
        approach = None

    rank, world_size = dist_init('27771')

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
    Tasks = generator.GetTasks(args.approach, args.batch_size, world_size, \
        memory_size=args.memory_size, memory_mini_batch_size=args.memory_mini_batch_size)
    # Network
    net = network.mobilenet(pretrained=True).cuda()
    net = DistModule(net)
    # Approach
    Appr = approach.Approach(net, args, Tasks)

    # Solve tasks incrementally
    for t in range(len(Tasks)):
        task = Tasks[t]

        if rank == 0:
            print('*'*100)
            print()
            print('Task {:d}: {:d} classes ({:s})'.format(t, task['class_num'], task['description']))
            print()
            print('*'*100)

        Appr.solve(t)

        if rank == 0:
            print('*'*100)
            print('Task {:d}: {:d} classes Finished.'.format(t, task['class_num']))
            print('*'*100)

if __name__ == '__main__':
    main()
