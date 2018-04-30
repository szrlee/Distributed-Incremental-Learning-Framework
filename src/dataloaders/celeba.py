import os,sys
import numpy as np
import torch
import utils
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
from PIL import Image

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from distributed_utils import dist_init, average_gradients, DistModule

def default_loader(path):
    return Image.open(path).convert('RGB')
class MultiLabelDataset(data.Dataset):
    def __init__(self, root, label, transform = None, subset = [], loader = default_loader):
        images = [] 
        labels = open(label).readlines()
        for line in labels:
            line = line.replace('-1', '0')
            items = line.split() 
            img_name = items.pop(0)
            items = np.array(items)[subset]
            if os.path.isfile(os.path.join(root, img_name)):
                images.append((img_name, tuple([int(v) for v in items])))
            else:
                print(os.path.join(root, img_name) + 'Not Found.')
        self.root = root
        self.images = images
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_name, label = self.images[index]
        img = self.loader(os.path.join(self.root, img_name))
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.Tensor(label)

    def __len__(self):
        return len(self.images)


def GetTasks(approach, batch_size, gpus):
    # Data Transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.Resize(size = (224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
        ])
    transform_test = transforms.Compose([
        transforms.Resize(size = (224,224)),
        transforms.ToTensor(),
        normalize
        ])

    # Construct dataloader for every task
    Tasks = []
    fullset = np.arange(40)
    for t in range(4):
        test_subset = fullset[t*10:t*10+10]
        if approach == 'joint_train':
            train_subset = fullset[0:t*10+10]
        else :
            train_subset = test_subset
        train_dataset = MultiLabelDataset(root='/mnt/lustre17/tangchufeng/sensedata/CelebA/Img/img_align_celeba/',
                                    label='/mnt/lustre17/tangchufeng/sensedata/CelebA/Anno/CelebA/train.list', transform=transform_train,
                                    subset = train_subset)
        test_dataset = MultiLabelDataset(root='/mnt/lustre17/tangchufeng/sensedata/CelebA/Img/img_align_celeba/',
                                    label='/mnt/lustre17/tangchufeng/sensedata/CelebA/Anno/CelebA/test.list', transform=transform_test,
                                    subset = test_subset)
        
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2*gpus, pin_memory=True, sampler = train_sampler)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2*gpus, pin_memory=True, sampler = test_sampler)
        task = {}
        task['train_loader'] = train_loader
        task['test_loader'] = test_loader
        task['train_sampler'] = train_sampler
        task['test_sampler'] = test_sampler
        task['train_subset'] = train_subset
        task['subset'] = test_subset
        task['description'] = 'CelebA Task #' + str(t)
        task['class_num'] = 10
        Tasks.append(task)

    if dist.get_rank() == 0:
        print('Data Processing ...')

    return Tasks