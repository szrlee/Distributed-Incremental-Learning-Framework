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

import quadprog

######################################################################
# Auxiliary functions useful for GEM's inner optimization.

def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose())
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))

########################################################################
# Model class description
class Approach(object):
    """ Class implementing the Gradient Episodic Memory approach described
        in https://arxiv.org/abs/1706.08840
    """

    def __init__(self, model, args):
        self.model = model

        cudnn.benchmark = True
        self.epochs = args.epochs
        self.lr = args.lr
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.print_freq = args.print_freq
        self.criterion = torch.nn.BCELoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr,
                                momentum=self.momentum,
                                weight_decay=self.weight_decay)
        
        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.model.parameters():
            self.grad_dims.append(param.data.numel())
        # currently manually set number of tasks as 4
        # TODO auto-maintain the number of total tasks
        self.grads = torch.Tensor(sum(self.grad_dims), 4).cuda()
        self.margin = args.margin # regularization parameter
        self.solved_tasks = []

        # process
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        return

    def solve(self, t, Tasks):
        task = Tasks[t]
        best_accu = 0
        train_sampler = task['train_sampler']
        train_loader = task['train_loader']
        # reinit learning rate for optimizer for each task
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        # cycle epoch training
        for epoch in range(self.epochs):
            train_sampler.set_epoch(epoch)
            self.adjust_learning_rate(self.optimizer, epoch)
            # train for one epoch
            self.train(t, train_loader, epoch, Tasks)
            # evaluate on validation set
            accu = self.validate(t, epoch, Tasks)
            # remember best prec@1 and save checkpoint
            if accu > best_accu:
                best_accu = accu

        # if rank == 0:
        #     print('Best accuracy: ', best_accu)

        # finish solving current task and
        # then append it to solved_tasks
        if t not in self.solved_tasks:
            self.solved_tasks.append(t)
        
        return best_accu

    def compute_pre_param(self, t, memory_loader, epoch, Tasks):
        # if self.rank == 0:
        #     print("== BEGIN: compute grad for pre observed tasks: {task}".format(task=t))
        # end = time.time()
        self.optimizer.zero_grad()
        mem_batch_cnt = int(len(memory_loader))
        for input, target in memory_loader:
            target = target.cuda(async=True)
            input = input.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = self.model(input_var)
            output = torch.nn.functional.sigmoid(output)
            # compute loss divided by world_size and mem_batch_cnt
            loss = self.criterion(output[:,Tasks[t]['subset']], target_var) / (self.world_size*mem_batch_cnt)
            # compute gradient for each batch of memory and accumulate
            loss.backward()
        
        average_gradients(self.model)
        # if self.rank == 0:
        #     print("== END: compute grad for pre observed task: {task} | TIME: {time} ".\
        #         format(task=t, time=(time.time()-end)) )
        return self.model.parameters

    def train(self, t, train_loader, epoch, Tasks):
        """Train for one epoch on the training set"""
        batch_time = AverageMeter()
        losses = AverageMeter()
        accuracy = AverageMeter()
        # switch to train mode
        self.model.train()

        end = time.time()
        batch_cnt = int(len(train_loader))
        for i, (input, target) in enumerate(train_loader):
            # ================================================================= #
            # compute grad for data at previous tasks
            if len(self.solved_tasks) > 0:
                if self.rank == 0:
                    print("====== compute grad for pre observed tasks: {tasks}".format(tasks=self.solved_tasks))
                # compute grad for pre observed tasks
                for pre_t in self.solved_tasks:
                    ## smaple few examples from previous tasks
                    # memory_sampler = Tasks[pre_t]['memory_sampler']
                    # memory_sampler.set_epoch(epoch) # random or fix sample?
                    memory_loader = Tasks[pre_t]['memory_loader']
                    ## compute gradient for few samples in previous tasks
                    if self.rank == 0:
                        print("== BEGIN: compute grad for pre observed tasks: {task}".format(task=pre_t))
                    end_pre = time.time()
                    #
                    pre_param = self.compute_pre_param(pre_t, memory_loader, epoch, Tasks)
                    #
                    if self.rank == 0:
                        print("== END: compute grad for pre observed task: {task} | TIME: {time} ".\
                            format(task=pre_t, time=(time.time()-end_pre)) )
                    ## copy previous grad to tensor
                    store_grad(pre_param, self.grads, self.grad_dims, pre_t)
            # ================================================================= #
            # compute grad for data at current task
            target = target.cuda(async=True)
            input = input.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = self.model(input_var)
            output = torch.nn.functional.sigmoid(output)

            loss = self.criterion(output[:,Tasks[t]['subset']], target_var) / self.world_size

            # compute gradient within constraints and backprop errors
            self.optimizer.zero_grad()
            loss.backward()
            average_gradients(self.model)
            # ================================================================== #
            # check grad and get new grad 
            if len(self.solved_tasks) > 0:
                if self.rank == 0:
                    print("== BEGIN: check constraints; if violate, get surrogate grad.")
                end_opt = time.time()
                ## copy gradient for data at current task to a tensor and clear grad
                store_grad(self.model.parameters, self.grads, self.grad_dims, t)
                ## check if current step gradient violate constraints
                indx = torch.cuda.LongTensor(self.solved_tasks)
                dotp = torch.mm(self.grads[:, t].unsqueeze(0),
                            self.grads.index_select(1, indx))
                if (dotp < 0).sum() != 0:
                    violate_constr = True
                else:
                    violate_constr = False
                ## use convex quadratic prorgamming to get surrogate grad
                if violate_constr:
                    # if violate, use quadprog to get new grad
                    self.optimizer.zero_grad()
                    project2cone2(self.grads[:, t].unsqueeze(1),
                              self.grads.index_select(1, indx), self.margin)
                    ## copy surrogate grad back to model gradient parameters
                    overwrite_grad(self.model.parameters, self.grads[:, t],
                               self.grad_dims)
                if self.rank == 0:
                    print("== END: violate constraints? : {vio_constr} | TIME: {time}".\
                        format(vio_constr=violate_constr, time=(time.time()-end_opt))
                        )
            # ================================================================= #
            # then do SGD step
            self.optimizer.step()

            # measure accuracy and record loss
            accu, _ = self.cleba_accuracy(t, output.data, target, Tasks)

            reduced_loss = loss.data.clone()
            reduced_accu = accu.clone() / self.world_size

            dist.all_reduce_multigpu([reduced_loss])
            dist.all_reduce_multigpu([reduced_accu])

            losses.update(reduced_loss[0], input.size(0))
            accuracy.update(reduced_accu[0], input.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            if i % self.print_freq == 0 and self.rank == 0: 
                print('Training Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})'.format(
                          epoch, i, batch_cnt, batch_time=batch_time,
                          loss=losses, accuracy=accuracy))

            end = time.time()

    def validate(self, t, epoch, Tasks):
        """Perform validation on the validation set"""
        tol_accu = 0.0
        tol_loss = 0.0
        tol_class = 0
        # switch to evaluate mode
        self.model.eval()
        batch_time = AverageMeter()

        tol_tasks = len(Tasks)
        cur_class = 0
        
        # Begin
        end = time.time()

        for cur_t in range(tol_tasks):
            losses = AverageMeter()
            accuracy = AverageMeter()
            class_num = Tasks[cur_t]['class_num']
            accuracys = []
            for cl in range(class_num):
                accuracys.append(AverageMeter())
            test_loader = Tasks[cur_t]['test_loader']
            for i, (input, target) in enumerate(test_loader):
                target = target.cuda(async=True)
                input = input.cuda()
                input_var = torch.autograd.Variable(input, volatile=True)
                target_var = torch.autograd.Variable(target, volatile=True)

                # compute output
                output = self.model(input_var)
                output = torch.nn.functional.sigmoid(output)
                loss = self.criterion(output[:,Tasks[cur_t]['subset']], target_var) / self.world_size

                # measure accuracy and record loss
                (accu, accus) = self.cleba_accuracy(cur_t, output.data, target, Tasks)
                
                reduced_loss = loss.data.clone()
                reduced_accu = accu.clone() / self.world_size

                reduced_accus = accus.clone() / self.world_size

                dist.all_reduce_multigpu([reduced_loss])
                dist.all_reduce_multigpu([reduced_accu])
                for cl in range(class_num):
                    dist.all_reduce_multigpu([reduced_accus[cl:cl+1]])

                losses.update(reduced_loss[0], input.size(0))
                accuracy.update(reduced_accu[0], input.size(0))
                for cl in range(class_num):
                    accuracys[cl].update(reduced_accus[cl], input.size(0))
                
                # measure elapsed time
                batch_time.update(time.time() - end)
                if i % self.print_freq == 0 and self.rank == 0: 
                    print('Testing Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accu {accuracy.val:.3f} ({accuracy.avg:.3f})'.format(
                        epoch, i, len(test_loader), batch_time=batch_time,
                        loss=losses, accuracy=accuracy))
                end = time.time()


            if self.rank == 0:
                print('=' * 100)
                for cl in range(class_num):
                    print('Accu @ Class{:2d} = {:.3f}'.format(cur_class, accuracys[cl].avg))
                    cur_class = cur_class + 1
                print(' *{:s} Task {:d}: Accuracy {accuracy.avg:.3f} Loss {loss.avg:.4f}'\
                    .format('**' if t==cur_t else '', cur_t, accuracy=accuracy, loss=losses))
                tol_accu = tol_accu + accuracy.avg
                tol_loss = tol_loss + losses.avg

        if self.rank == 0:
            print(' * Total: Accuracy {:3f} Loss {:4f}'.format(tol_accu/tol_tasks, tol_loss/tol_tasks))
        return tol_accu / tol_tasks

    def adjust_learning_rate(self, optimizer, epoch):
        lr = self.lr * (0.1 ** (epoch // 5)) * (0.1 ** (epoch // 7))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def cleba_accuracy(self, t, output, target, Tasks):
        batch_size = target.size(0)
        attr_num = target.size(1)

        output = output.cpu().numpy()[:,Tasks[t]['subset']]
        output = np.where(output > 0.5, 1, 0)
        pred = torch.from_numpy(output).long().cuda()
        target = target.long()
        correct = pred.eq(target).float()

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
