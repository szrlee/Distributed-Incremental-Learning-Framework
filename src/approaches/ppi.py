import sys,time
import numpy as np
import torch
from copy import deepcopy
import torch.backends.cudnn as cudnn
import utils
from meter.apmeter import APMeter

from torch.utils.data import DataLoader

######################################################################
# Auxiliary functions useful for GEM's inner optimization.
import quadprog

def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters iterator
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp:
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters iterator
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp:
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
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
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))

########################################################################
# Model class description
class Approach(object):
    """ Gradient Pseudo Memory """

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

        self.save_seen_mAP = [args.save_dir+'/'+args.network+'_'+args.approach+'_TASK'+str(t)+'_bestmAPseenTasks_'+args.time+'.pt' for t in range(len(Tasks))]
        self.save_mAP_t = [args.save_dir+'/'+args.network+'_'+args.approach+'_TASK'+str(t)+'_bestmAPcurTask_'+args.time+'.pt' for t in range(len(Tasks))]

        # allocate temporary synaptic memory
        ignored_params_id_list = list(map(id, self.model.module.newfc.parameters()))
        # self.base_params = filter(lambda p: 'newfc' not in p[0], self.model.named_parameters())
        self.base_params = [p for p in self.model.module.parameters() if id(p) not in ignored_params_id_list]

        self.grad_dims = []
        for param in self.base_params:
            if param.requires_grad:
                self.grad_dims.append(param.data.numel())

        # auto-maintain the number of total tasks
        self.total_tasks = len(Tasks)
        self.grads = torch.Tensor(sum(self.grad_dims), self.total_tasks).cuda()
        self.margin = args.margin # regularization parameter
        self.solved_tasks = []
        # for prefetch memory for cache
        self.cur_t = -1

        # alternate updating
        self.n_sub_iter = 3

    def solve(self, t):
        self.cur_t = t
        # load best model in previous task (Start from the second task)
        if t>0:
            print(f"loading best model in previous task {t-1}")
            checkpoint = torch.load(self.save_seen_mAP[t-1])
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
            print(f"loading completed!")

        # prepare task specific object
        task = self.Tasks[t]
        train_loader = task['train_loader']

        self.optimizer = torch.optim.SGD(self.base_params, self.lr,
                                momentum=self.momentum,
                                weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[8], gamma=0.1)
        
        self.optim_fc = torch.optim.SGD(self.model.module.newfc.parameters(), self.lr,
                                momentum=self.momentum,
                                weight_decay=self.weight_decay)
        self.sche_fc = torch.optim.lr_scheduler.MultiStepLR(self.optim_fc, milestones=[8], gamma=0.1)

        print("Evaluate the Initialization model")
        self.validate(t=t-1, epoch=-1)
        print('=' * 100)

        # cycle epoch training
        best_mAP_t  = 0
        ep_best_mAP_t = 0
        best_seen_mAP  = 0
        ep_best_seen_mAP = 0
        for epoch in range(self.epochs):
            self.scheduler.step()
            self.sche_fc.step()

            # train for one epoch
            print("===Start Training")
            self.train(t, train_loader, epoch)
            # evaluate on validation set
            print("===Start Evaluating")
            accu, mean_ap, seen_mAP, mAP_t = self.validate(t, epoch)

            ## remember best mAP on cur task and save checkpoint
            if mAP_t > best_mAP_t:
                best_mAP_t = mAP_t
                ep_best_mAP_t = epoch
                # save best
                torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.module.state_dict()
                }, self.save_mAP_t[t])

            ## remember best mAP on seen tasks and save checkpoint
            if seen_mAP > best_seen_mAP:
                best_seen_mAP = seen_mAP
                ep_best_seen_mAP = epoch
                # save best
                torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.module.state_dict()
                }, self.save_seen_mAP[t])

        print(f'Best mean AP for Curr Task=={t} at epoch {ep_best_mAP_t}: {best_mAP_t}')
        print(f'Best mean AP for Seen Task<={t} at epoch {ep_best_seen_mAP}: {best_seen_mAP}')

        # finish solving current task and
        # then append it to solved_tasks
        if t not in self.solved_tasks:
            self.solved_tasks.append(t)
        
    def compute_pre_param(self, t, output, target, epoch):
        self.optimizer.zero_grad()
        self.optim_fc.zero_grad()
        subset = self.Tasks[t]['test_subset']

        # compute loss
        loss = self.criterion(output[:,subset], target[:,subset])
        # compute gradient
        loss.backward(retain_graph=True)
        return self.base_params

    def update_task_param(self, output, target, epoch, iter, sub_iter):
        self.optimizer.zero_grad()
        self.optim_fc.zero_grad()
        # subset = np.empty(0, dtype=int)
        # # freeze head
        # for param in self.base_params:
        #     param.requires_grad = False
        # compute loss
        subset = self.Tasks[self.cur_t]['test_subset']
        loss = self.criterion(output[:,subset], target[:,subset])
        ## added with prev task loss
        for pre_t in self.solved_tasks:
            subset = self.Tasks[pre_t]['test_subset']
            loss += self.criterion(output[:,subset], target[:,subset])

        # compute gradient and retain computation graph
        loss.backward(retain_graph=True)
        # fc layer update
        self.optim_fc.step()
        # # unfreeze head
        # for param in self.base_params:
        #     param.requires_grad = True
        return self.base_params

    def train(self, t, train_loader, epoch):
        """Train for one epoch on the training set"""
        assert(t==self.cur_t)
        batch_time = AverageMeter()
        losses = AverageMeter()
        accuracy = AverageMeter()
        # switch to train mode
        self.model.train()
        count_vio = 0
        end = time.time()
        for i, (input, target) in enumerate(train_loader):

            target = target.to(self.device)
            input = input.to(self.device)

            # ================================================================= #
            # subiteration for task spec param
            utils.freeze_param(self.base_params)
            for sub_i in range(self.n_sub_iter):
                # compute output
                output = self.model(input)
                output = torch.sigmoid(output)
                # update task specific param
                self.update_task_param(output, target, epoch, i, sub_i)
            utils.unfreeze_param(self.base_params)
            # ================================================================= #

            # ================================================================= #
            # compute output
            output = self.model(input)
            output = torch.sigmoid(output)
            # ================================================================= #
            # compute grad for previous tasks
            if len(self.solved_tasks) > 0:
                # compute grad for pre observed tasks
                for pre_t in self.solved_tasks:
                    ## compute gradient for few samples in previous tasks
                    pre_param = self.compute_pre_param(pre_t, output, target, epoch)
                    ## store prev grad to tensor
                    store_grad(pre_param, self.grads, self.grad_dims, pre_t)
            # ================================================================= #
            # compute grad for current task
            subset = self.Tasks[t]['test_subset']
            loss = self.criterion(output[:,subset], target[:,subset])
            # compute gradient within constraints and backprop errors
            self.optimizer.zero_grad()
            self.optim_fc.zero_grad()
            loss.backward()

            # ================================================================== #
            # check grad and get new grad 
            if len(self.solved_tasks) > 0:
                # print("== BEGIN: check constraints; if violate, get surrogate grad.")
                end_opt = time.time()
                ## copy gradient for data at current task to a tensor and clear grad
                store_grad(self.base_params, self.grads, self.grad_dims, t)
                ## check if current step gradient violate constraints
                indx = torch.cuda.LongTensor(self.solved_tasks)
                dotp = torch.mm(self.grads[:, t].unsqueeze(0),
                            self.grads.index_select(1, indx))
                violate_constr = ((dotp < 0).sum() != 0)

                ## use convex quadratic prorgamming to get surrogate grad
                if violate_constr:
                    # count violating times
                    count_vio += 1
                    # if violate, use quadprog to get new grad
                    self.optimizer.zero_grad()
                    self.optim_fc.zero_grad()
                    project2cone2(self.grads[:, t].unsqueeze(1),
                              self.grads.index_select(1, indx), self.margin)
                    ## copy surrogate grad back to model gradient parameters
                    overwrite_grad(self.base_params, self.grads[:, t],
                               self.grad_dims)
                # print(f"== END: violate constraints? : {violate_constr} | TIME: {time.time()-end_opt}")
            # ================================================================= #
            # then do SGD step
            self.optimizer.step()

            # measure accuracy and record loss
            accu, _ = self.cleba_accuracy(t, output.data, target, 'train')

            reduced_loss = loss.data.clone()
            reduced_accu = accu.clone()
            losses.update(reduced_loss.item(), input.size(0))
            accuracy.update(reduced_accu.item(), input.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % self.print_freq == 0 or (i+1) == len(train_loader):
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {accuracy.val:.3f} ({accuracy.avg:.3f})\t'
                      '#Proj [{count_vio}/{batch_cnt}]'.format(
                          epoch, i+1, len(train_loader), batch_time=batch_time,
                          loss=losses, accuracy=accuracy,
                          count_vio=count_vio, batch_cnt=len(train_loader)))

    def validate(self, t, epoch):
        """Perform validation on the validation set"""
        # switch to evaluate mode
        self.model.eval()

        tol_tasks = len(self.Tasks)
        cur_class = 0
        losses = AverageMeter()
        accuracy = AverageMeter()
        APs = APMeter()
        # Sum over self.Tasks[cur_t]['class_num']
        class_num = 20
        accuracys = []
        
        for cl in range(class_num):
            accuracys.append(AverageMeter())
        test_loader = self.Tasks[0]['test_loader']
        for i, (input, target) in enumerate(test_loader):
            target = target.to(self.device)
            input = input.to(self.device)
            with torch.no_grad():
                # compute output
                output = self.model(input)
                output = torch.sigmoid(output)
                loss = self.criterion(output, target)
                # measure accuracy and record loss
                (accu, accus) = self.cleba_accuracy(0, output.data, target)
                APs.add(output, target)
                # average inside batch
                reduced_loss = loss.data.clone()
                reduced_accu = accu.clone()
                reduced_accus = accus.clone()
                # average with other batch
                losses.update(reduced_loss.item(), input.size(0))
                accuracy.update(reduced_accu.item(), input.size(0))
                for cl in range(class_num):
                    accuracys[cl].update(reduced_accus[cl], input.size(0))
        
        seen_subset = np.empty(0, dtype=int)
        mAP_t = 0.0
        for cur_t in range(tol_tasks):
            subset = self.Tasks[cur_t]['test_subset']
            ap = APs.value() * 100.0
            print(' '*10+'|   AP   |  Accu  |')
            for cl in subset:
                print(f'Class{cl:2d} = | {ap[cl]:6.3f} | {accuracys[cl].avg:6.3f} |')
            # print(' *{:s} Task {:d}: Accuracy {accuracy.avg:.3f} Loss {loss.avg:.4f}'.format('**' if t==cur_t else '', cur_t, accuracy=accuracy, loss=losses))
            mAP_cur_t = ap[subset].mean().item()
            print(' *{:s} Task {:d}: mAP {mAP:.3f}'.format('**' if t==cur_t else '', cur_t, mAP=mAP_cur_t))
            if cur_t <= t:
                seen_subset = np.unique(np.concatenate((seen_subset, subset)))
            if cur_t == t:
                mAP_t = mAP_cur_t
            
        mean_ap = ap.mean().item()
        print('===Total: mAP {:3f} Accuracy {:3f} Loss {:4f}'.format(mean_ap, accuracy.avg, losses.avg))
        seen_mAP = ap[seen_subset].mean().item()
        print(f'===Seen Labels {seen_subset}: mean_AP {seen_mAP:3f}')
        print()
        return accuracy.avg, mean_ap, seen_mAP, mAP_t

    def cleba_accuracy(self, t, output, target, stat='test'):
        batch_size = target.size(0)
        attr_num = target.size(1)

        if stat == 'train':
            output = output.cpu().numpy()[:,self.Tasks[t]['train_subset']]
        elif stat == 'test':
            output = output.cpu().numpy()
        else:
            raise NotImplementedError
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