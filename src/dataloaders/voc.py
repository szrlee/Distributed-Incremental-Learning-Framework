import os,sys
import numpy as np
import torch
import utils
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import logging
from PIL import Image

import torch.distributed as dist
from torch.utils.data import DataLoader

def default_loader(path):
    return Image.open(path).convert('RGB')

class MultiLabelDataset(data.Dataset):
    def __init__(self, root, label, transform = None, subset = [], loader = default_loader):
        images = [] 
        labels = open(label).readlines()
        for line in labels:
            items = line.split() 
            img_name = items.pop(0)
            items = np.array(items)[subset]
            images.append((img_name, tuple([int(v) for v in items])))

        self.root = root
        self.images = images
        self.transform = transform
        self.loader = loader
        self.initialized = False

    def __getitem__(self, index):

        img_name, label = self.images[index]
        img_name = os.path.join(self.root, img_name + '.jpg')

        img = self.loader(img_name)

        if self.transform is not None:
            img = self.transform(img)
        return img, torch.Tensor(label)

    def __len__(self):
        return len(self.images)


def GetTasks(approach, batch_size, memory_size=None, memory_mini_batch_size=None):
    """Get Tasks (a list of dict) that yield sampler and dataloader for training, testing and memory dataset.

    Arguments:
        approach (string) : The name of current method for incremental learning
        batch_size (int) : Size of mini-batch
        memory_size (optional) : Size of memory sampled from previous tasks (for approaces need memory)
        memory_mini_batch_size (optional) : Size of batch size for memory in the approachs (EWC, MAS)
            for specific gradient-related computing
        
    """

    # # Data Transforms
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                              std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.Resize(size = (224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        ])
    transform_test = transforms.Compose([
        transforms.Resize(size = (224,224)),
        transforms.ToTensor()
        ])

    # Construct dataloader for every task
    Tasks = []
    fullset = np.arange(20)
    task1_set = np.arange(7)
    task2_set = np.arange(7,14)
    task3_set = np.arange(14,20)

    print('Dataloader Process: BEGIN')

    # manualy set tasks specific number
    access_prev_label =  ['fine_tune_aug_label', 'gpm', \
                               'ml_lwf_aug_label', 'ml_lwf_aug_label_gpm']
    no_access_prev_label = ['fine_tuning', 'ml_lwf', 'ml_lwf_gpm']
    if approach == 'joint_train':
        total_t = 1
        test_subsets = [fullset]
        train_subsets = [fullset]
    elif approach in access_prev_label:
        ## access to label in previous tasks
        total_t = 3
        test_subsets = [task1_set, task2_set, task3_set]
        # for training: union current subset with all previous
        task2_set = np.unique(np.concatenate((task1_set, task2_set)))
        task3_set = np.unique(np.concatenate((task2_set, task3_set)))
        train_subsets = [task1_set, task2_set, task3_set]
    elif approach in no_access_prev_label:
        ## no access to label in previous tasks
        total_t = 3
        train_subsets = [task1_set, task2_set, task3_set]
        test_subsets = [task1_set, task2_set, task3_set]
    else:
        raise NotImplementedError

    class_nums = [len(subset_i) for subset_i in test_subsets]

    for t in range(total_t): # default order of tasks 0, 1, 2, 3, ...
        print(f'generating dataloader for task {t}')
        test_subset = test_subsets[t]
        train_subset = train_subsets[t]

        if approach == 'joint_train':
            train_file='/home/ubuntu/ml-voc/ImageSets/train_all.txt'
        else:
            train_file='/home/ubuntu/ml-voc/ImageSets/task'+str(t)+'_train.txt'

        print(f"filename of training data in Task {t}: {train_file}")
        train_dataset = MultiLabelDataset(root='/home/ubuntu/ml-voc/JPEGImages/',
                                    label=train_file,
                                    transform=transform_train,
                                    subset = train_subset)
        test_dataset = MultiLabelDataset(root='/home/ubuntu/ml-voc/JPEGImages/',
                                    label='/home/ubuntu/ml-voc/ImageSets/test_all.txt', 
                                    transform=transform_test,
                                    subset = test_subset)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                                num_workers=2, pin_memory=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                                num_workers=2, pin_memory=False)
        task = {}
        task['train_dataset'] = train_dataset
        task['test_dataset'] = test_dataset
        task['train_loader'] = train_loader
        task['test_loader'] = test_loader
        task['description'] = 'VOC Task #' + str(t)
        task['class_num'] = class_nums[t]
        task['train_subset'] = train_subset
        task['test_subset'] = test_subset

        Tasks.append(task)

    print('Dataloader Process: DONE')

    return Tasks
