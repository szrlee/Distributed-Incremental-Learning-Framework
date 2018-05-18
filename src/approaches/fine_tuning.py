import multiprocessing as mp
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)
import sys,time
import numpy as np
import torch
from copy import deepcopy
import torch.backends.cudnn as cudnn
import utils

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from distributed_utils import dist_init, average_gradients, DistModule

class Approach(object):
    """ Class implementing the Learning Without Forgetting approach described in https://arxiv.org/abs/1606.09282 """

    def __init__(self, model, args, Tasks):
        self.model = model

        self.Tasks = Tasks
        cudnn.benchmark = True
        self.epochs = args.epochs
        self.lr = args.lr
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.print_freq = args.print_freq
        self.criterion = torch.nn.BCELoss().cuda()
        self.optimizer = torch.optim.SGD(model.parameters(), self.lr,
                                momentum=self.momentum,
                                weight_decay=self.weight_decay)

        return

    def solve(self, t):
        task = self.Tasks[t]
        train_loader = task['train_loader']
        val_loader = task['test_loader']
        class_num = task['class_num']
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr,
                                momentum=self.momentum,
                                weight_decay=self.weight_decay)
        criterion = self.criterion

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        best_accu = 0

        train_sampler = task['train_sampler']
        testsampler = task['test_sampler']

        for epoch in range(self.epochs):
            train_sampler.set_epoch(epoch)
            self.adjust_learning_rate(self.optimizer, epoch)
            # train for one epoch
            self.train(t, train_loader, self.model, self.optimizer, epoch)
            # evaluate on validation set
            accu = self.validate(t, self.model, epoch)

            # remember best prec@1 and save checkpoint
            if accu > best_accu:
                best_accu = accu

        # if rank == 0:
        #     print('Best accuracy: ', best_accu)

        return best_accu


    def train(self, t, train_loader, model, optimizer, epoch):
        """Train for one epoch on the training set"""
        batch_time = AverageMeter()
        losses = AverageMeter()
        accuracy = AverageMeter()
        # switch to train mode
        model.train()

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        end = time.time()
        batch_cnt = int(len(train_loader))
        for i, (input, target) in enumerate(train_loader):
            target = target.cuda(async=True)
            input = input.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            output = torch.nn.functional.sigmoid(output)

            loss = self.criterion(output[:,self.Tasks[t]['train_subset']], target_var) / world_size

            # measure accuracy and record loss
            (accu, accus) = self.cleba_accuracy(t, output.data, target, 'train')

            reduced_loss = loss.data.clone()
            reduced_accu = accu.clone() / world_size

            dist.all_reduce_multigpu([reduced_loss])
            dist.all_reduce_multigpu([reduced_accu])

            losses.update(reduced_loss[0], input.size(0))
            accuracy.update(reduced_accu[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            average_gradients(model)
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0 and rank == 0: 
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})'.format(
                          epoch, i, batch_cnt, batch_time=batch_time,
                          loss=losses, accuracy=accuracy))

    def validate(self, t, model, epoch):
        """Perform validation on the validation set"""
        tol_accu = 0.0
        tol_loss = 0.0
        tol_class = 0
        # switch to evaluate mode
        model.eval()

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        tol_tasks = len(self.Tasks)
        cur_class = 0
        for cur_t in range(tol_tasks):
            losses = AverageMeter()
            accuracy = AverageMeter()
            class_num = self.Tasks[cur_t]['class_num']
            accuracys = []
            for cl in range(class_num):
                accuracys.append(AverageMeter())
            test_loader = self.Tasks[cur_t]['test_loader']
            for i, (input, target) in enumerate(test_loader):
                target = target.cuda(async=True)
                input = input.cuda()
                input_var = torch.autograd.Variable(input, volatile=True)
                target_var = torch.autograd.Variable(target, volatile=True)

                # compute output
                output = model(input_var)
                output = torch.nn.functional.sigmoid(output)
                loss = self.criterion(output[:,self.Tasks[cur_t]['subset']], target_var) / world_size

                # measure accuracy and record loss
                (accu, accus) = self.cleba_accuracy(cur_t, output.data, target)
                
                reduced_loss = loss.data.clone()
                reduced_accu = accu.clone() / world_size

                reduced_accus = accus.clone() / world_size

                dist.all_reduce_multigpu([reduced_loss])
                dist.all_reduce_multigpu([reduced_accu])
                for cl in range(class_num):
                    dist.all_reduce_multigpu([reduced_accus[cl:cl+1]])

                losses.update(reduced_loss[0], input.size(0))
                accuracy.update(reduced_accu[0], input.size(0))
                for cl in range(class_num):
                    accuracys[cl].update(reduced_accus[cl], input.size(0))

                # if i % self.print_freq == 0 and rank == 0: 
                # print('Epoch: [{0}][{1}/{2}]\t'
                #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                #       'Accu {accuracy.val:.3f} ({accuracy.avg:.3f})'.format(
                #           epoch, i, len(test_loader), batch_time=batch_time,
                #           loss=losses, accuracy=accuracy))

            if rank == 0:
                for cl in range(class_num):
                    print('Accu @ Class{:2d} = {:.3f}'.format(cur_class, accuracys[cl].avg))
                    cur_class = cur_class + 1
                print(' *{:s} Task {:d}: Accuracy {accuracy.avg:.3f} Loss {loss.avg:.4f}'.format('**' if t==cur_t else '', cur_t, accuracy=accuracy, loss=losses))
                tol_accu = tol_accu + accuracy.avg
                tol_loss = tol_loss + losses.avg

        if rank == 0:
            print(' * Total: Accuracy {:3f} Loss {:4f}'.format(tol_accu/tol_tasks, tol_loss/tol_tasks))
        return tol_accu / tol_tasks

    def adjust_learning_rate(self, optimizer, epoch):
        lr = self.lr * (0.1 ** (epoch // 5)) * (0.1 ** (epoch // 7))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def cleba_accuracy(self, t, output, target, stat='test'):
        batch_size = target.size(0)
        attr_num = target.size(1)

        if stat == 'train':
            output = output.cpu().numpy()[:,self.Tasks[t]['train_subset']]
        else:
            output = output.cpu().numpy()[:,self.Tasks[t]['subset']]
        output = np.where(output > 0.5, 1, 0)
        pred = torch.from_numpy(output).long().cuda()
        target = target.long()
        correct = pred.eq(target).float()

        #if stat == 'test':
        #    print(batch_size, pred.nonzero().size(0), target.nonzero().size(0))

        accu = correct.sum(0).mul_(100.0/batch_size)
        ave_accu = accu.sum(0).div_(attr_num)
        return ave_accu, accu


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count