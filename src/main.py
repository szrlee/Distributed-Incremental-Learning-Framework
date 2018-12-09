import sys,os,argparse,time
import numpy as np
import torch

import utils
from torch.utils.data import DataLoader

# Arguments
parser=argparse.ArgumentParser(description='Continual Learning Framework')
parser.add_argument('--experiment', default='', type=str, required=True, choices=['voc'], help='(default=%(default)s)')
parser.add_argument('--approach', default='', type=str, required=True, \
    choices=['joint_train', 'fine_tune', 'ml_lwf', 'gpm', 'ml_lwf_gpm', 'ppi'], help='(default=%(default)s)')
parser.add_argument('--save_dir', default='../res/model/', type=str, required=False, help='(default=%(default)s)')
parser.add_argument('--time', default='now', type=str, required=True, help='(default=%(default)s)')

parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')

# Network
parser.add_argument('--network', default='res18', type=str, required=False, choices=['res18', 'res34', 'res50', 'res101'], help='(default=%(default)s)')
parser.add_argument('--pretrain', required=False, action='store_true', help='(default=%(default)f)')

parser.add_argument('--lr', default=0.1, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--epochs', default=5, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--batch_size', default=512, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--momentum', default=0.9, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--weight_decay', default=0.0005, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--print_freq', default=5, type=int, required=False, help='(default=%(default)d)')

parser.add_argument('--memory_size', default=None, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--memory_mini_batch_size', default=None, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--margin', default=1.0, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--gem_mem_strategy', default=None, type=str, required=False, choices=['direct_load', 'fixed_gpu_cache'], help='(default=%(default)s)')

########################################################################################################################

# Args -- Experiment
from dataloaders import voc

# Args -- Approach
from approaches import joint_train
from approaches import fine_tune
from approaches import ml_lwf
from approaches import gpm
from approaches import ml_lwf_gpm

from approaches import ppi


# Args -- Network
from networks import resnet

########################################################################################################################
def main():
    global args
    args = parser.parse_args()

    # Dataset
    generator = voc

    # TODO model arguments module should be more easy to write and read
    if args.approach == 'joint_train':
        approach = joint_train
        assert(args.memory_size is None)
        assert(args.memory_mini_batch_size is None)
    elif args.approach == 'fine_tune':
        approach = fine_tune
        assert(args.memory_size is None)
        assert(args.memory_mini_batch_size is None)
    elif args.approach == 'ml_lwf':
        approach = ml_lwf
        assert(args.memory_size is None)
        assert(args.memory_mini_batch_size is None)
    elif args.approach == 'gpm':
        approach = gpm
        assert(args.memory_size is None)
        assert(args.memory_mini_batch_size is None)
    elif args.approach == 'ml_lwf_gpm':
        approach = ml_lwf_gpm
        assert(args.memory_size is None)
        assert(args.memory_mini_batch_size is None)
    elif args.approach == 'ppi':
        approach = ml_lwf_gpm
        assert(args.memory_size is None)
        assert(args.memory_mini_batch_size is None)
    else:
        approach = None

    # print all arguments
    print('=' * 100)
    print('Arguments = ')
    for arg in vars(args):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 100)

    # manually set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
    else: print('[CUDA unavailable]'); sys.exit()

    ### generate dataloader and network
    # Generate Tasks
    Tasks = generator.GetTasks(args.approach, args.batch_size, \
        memory_size=args.memory_size, memory_mini_batch_size=args.memory_mini_batch_size)
    # Network
    net = {'res18': resnet.resnet18,
        'res34': resnet.resnet34,
        'res50': resnet.resnet50,
        'res101': resnet.resnet101}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # net = resnet.resnet18(pretrained=args.pretrain).to(device)
    net = net[args.network](pretrained=args.pretrain).to(device)
    net = torch.nn.DataParallel(net)

    # Approach
    Appr = approach.Approach(net, args, Tasks)

    # Solve tasks incrementally
    for t in range(len(Tasks)):
        # get dict for one single task
        task = Tasks[t]

        print('*'*100)
        print()
        print('Task {:d}: {:d} classes ({:s})'.format(t, task['class_num'], task['description']))
        print()
        print('*'*100)
        # solve current task t
        Appr.solve(t)

        print('*'*100)
        print('Task {:d}: {:d} classes Finished.'.format(t, task['class_num']))
        print('*'*100)

if __name__ == '__main__':
    main()
