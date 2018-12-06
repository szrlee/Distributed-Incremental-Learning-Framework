import sys,time
import numpy as np
import torch
from copy import deepcopy
import torch.backends.cudnn as cudnn
import utils
from meter.apmeter import APMeter

from torch.utils.data import DataLoader

class Approach(object):
    """ Fine Tuning """

    def __init__(self, model, args, Tasks):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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

    def solve(self, t):
        task = self.Tasks[t]
        train_loader = task['train_loader']
        val_loader = task['test_loader']
        class_num = task['class_num']
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr,
                                momentum=self.momentum,
                                weight_decay=self.weight_decay)
        criterion = self.criterion

        best_accu = 0
        print("evaluate the initialization model")
        accu = self.validate(-1, self.model, -1)
        print('=' * 100)

        for epoch in range(self.epochs):
            self.adjust_learning_rate(self.optimizer, epoch)
            # train for one epoch
            self.train(t, train_loader, self.model, self.optimizer, epoch)
            # evaluate on validation set
            accu = self.validate(t, self.model, epoch)

            # remember best prec@1 and save checkpoint
            if accu > best_accu:
                best_accu = accu

        print('Best accuracy: ', best_accu)

        return best_accu


    def train(self, t, train_loader, model, optimizer, epoch):
        """Train for one epoch on the training set"""
        batch_time = AverageMeter()
        losses = AverageMeter()
        accuracy = AverageMeter()
        # switch to train mode
        model.train()

        end = time.time()
        batch_cnt = int(len(train_loader))
        for i, (input, target) in enumerate(train_loader):
            target = target.to(self.device)
            input = input.to(self.device)

            # compute output
            output = model(input)
            output = torch.sigmoid(output)

            loss = self.criterion(output[:,self.Tasks[t]['train_subset']], target)

            # measure accuracy and record loss
            (accu, accus) = self.cleba_accuracy(t, output.data, target, 'train')

            reduced_loss = loss.data.clone()
            reduced_accu = accu.clone()
            losses.update(reduced_loss.item(), input.size(0))
            accuracy.update(reduced_accu.item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {accuracy.val:.3f} ({accuracy.avg:.3f})'.format(
                          epoch, i, len(train_loader), batch_time=batch_time,
                          loss=losses, accuracy=accuracy))

    def validate(self, t, model, epoch):
        """Perform validation on the validation set"""
        tol_accu = 0.0
        tol_loss = 0.0
        tol_ap = 0.0
        tol_class = 0
        # switch to evaluate mode
        model.eval()

        tol_tasks = len(self.Tasks)
        cur_class = 0
        for cur_t in range(tol_tasks):
            losses = AverageMeter()
            accuracy = AverageMeter()
            APs = APMeter()
            class_num = self.Tasks[cur_t]['class_num']
            accuracys = []
            for cl in range(class_num):
                accuracys.append(AverageMeter())
            test_loader = self.Tasks[cur_t]['test_loader']
            for i, (input, target) in enumerate(test_loader):
                target = target.to(self.device)
                input = input.to(self.device)
                with torch.no_grad():

                    # compute output
                    output = model(input)
                    output = torch.sigmoid(output)
                    loss = self.criterion(output[:,self.Tasks[cur_t]['test_subset']], target)

                    # measure accuracy and record loss
                    (accu, accus) = self.cleba_accuracy(cur_t, output.data, target)
                    APs.add(output[:,self.Tasks[cur_t]['test_subset']], target)

                    reduced_loss = loss.data.clone()
                    reduced_accu = accu.clone()

                    reduced_accus = accus.clone()

                    losses.update(reduced_loss.item(), input.size(0))
                    accuracy.update(reduced_accu.item(), input.size(0))
                    for cl in range(class_num):
                        accuracys[cl].update(reduced_accus[cl], input.size(0))

                    # if i % self.print_freq == 0 and rank == 0: 
                    # print('Epoch: [{0}][{1}/{2}]\t'
                    #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    #       'Accu {accuracy.val:.3f} ({accuracy.avg:.3f})'.format(
                    #           epoch, i, len(test_loader), batch_time=batch_time,
                    #           loss=losses, accuracy=accuracy))

            ap = APs.value() * 100.0
            for cl in range(class_num):
                print(f'* AP * & Accu @ Class{cur_class:2d} = * {ap[cl]:.3f} * & {accuracys[cl].avg:.3f} ')
                cur_class = cur_class + 1
            print(' *{:s} Task {:d}: Accuracy {accuracy.avg:.3f} Loss {loss.avg:.4f}'.format('**' if t==cur_t else '', cur_t, accuracy=accuracy, loss=losses))
            print(' *{:s} Task {:d}: mAP {mAP:.3f}'.format('**' if t==cur_t else '', cur_t, mAP=ap.mean().item()))

            tol_accu = tol_accu + accuracy.avg
            tol_loss = tol_loss + losses.avg
            tol_ap = tol_ap + ap.sum().item()

        # TODO: wrong average
        print(' * Total: mAP {:3f} Accuracy {:3f} Loss {:4f}'.format(tol_ap/20, tol_accu/tol_tasks, tol_loss/tol_tasks))
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
            output = output.cpu().numpy()[:,self.Tasks[t]['test_subset']]
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