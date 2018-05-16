import os,sys
import numpy as np
import torch
import utils
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import mc
import io
from PIL import Image

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from distributed_utils import dist_init, average_gradients, DistModule, DistributedMemorySampler


def default_loader(path):
    return Image.open(path).convert('RGB')

def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    with Image.open(buff) as img:
        img = img.convert('RGB')
    return img

class MultiLabelDataset(data.Dataset):
    def __init__(self, root, label, transform = None, subset = [], loader = pil_loader):
        images = [] 
        labels = open(label).readlines()
        for line in labels:
            line = line.replace('-1', '0')
            items = line.split() 
            img_name = items.pop(0)
            items = np.array(items)[subset]
            # if os.path.isfile(os.path.join(root, img_name)):
            images.append((img_name, tuple([int(v) for v in items])))
            # else:
                # print(os.path.join(root, img_name) + 'Not Found.')
        self.root = root
        self.images = images
        self.transform = transform
        self.loader = loader
        self.initialized = False

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def __getitem__(self, index):
        self._init_memcached()
        img_name, label = self.images[index]
        img_name = os.path.join(self.root, img_name)
        value = mc.pyvector()
        self.mclient.Get(img_name, value)
        value_str = mc.ConvertBuffer(value)
        img = pil_loader(value_str)

        if self.transform is not None:
            img = self.transform(img)
        return img, torch.Tensor(label)

    def __len__(self):
        return len(self.images)


def GetTasks(approach, batch_size, gpus, memory_size=None, memory_mini_batch_size=None):
    """Get Tasks (a list of dict) that yield sampler and dataloader for training, testing and memory dataset.

    Arguments:
        approach (string) : The name of current method for incremental learning
        batch_size (int) : Size of mini-batch
        gpus (int) : The number of using gpus
        memory_size (optional) : Size of memory sampled from previous tasks (for approaces need memory)
        memory_mini_batch_size (optional) : Size of batch size for memory in the approachs (EWC, MAS)
            for specific gradient-related computing
        
    """
    # processing information
    rank = dist.get_rank()

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

    print('Dataloader Process: BEGIN at rank:{rank}'.format(rank=rank))

    for t in range(4):
        print('generating dataloader for task {t:2d} at rank:{rank:2d}'.format(t=t, rank=rank))
        test_subset = fullset[t*10:t*10+10]
        if approach == 'joint_train':
            train_subset = fullset[0:t*10+10]
        else :
            train_subset = test_subset
        train_dataset = MultiLabelDataset(root='/mnt/lustre17/tangchufeng/sensedata/CelebA/Img/img_align_celeba/',
                                    label='/mnt/lustre17/tangchufeng/sensedata/CelebA/Anno/CelebA/train.list',
                                    transform=transform_train,
                                    subset = train_subset)
        test_dataset = MultiLabelDataset(root='/mnt/lustre17/tangchufeng/sensedata/CelebA/Img/img_align_celeba/',
                                    label='/mnt/lustre17/tangchufeng/sensedata/CelebA/Anno/CelebA/test.list', 
                                    transform=transform_test,
                                    subset = test_subset)
        
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, 
                                                num_workers=2, pin_memory=True, sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                                num_workers=2, pin_memory=True, sampler=test_sampler)
        task = {}
        task['train_dataset'] = train_dataset
        task['test_dataset'] = test_dataset
        task['train_loader'] = train_loader
        task['test_loader'] = test_loader
        task['train_sampler'] = train_sampler
        task['test_sampler'] = test_sampler
        task['train_subset'] = train_subset
        task['subset'] = test_subset
        task['description'] = 'CelebA Task #' + str(t)
        task['class_num'] = 10
        # for those Approaches that need memory
        if memory_size is not None:
            memory_sampler = DistributedMemorySampler(train_dataset, sample_size=memory_size)
            memory_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                                    num_workers=2, pin_memory=False, sampler=memory_sampler,
                                                    drop_last=True)
            task['memory_sampler'] = memory_sampler
            task['memory_loader']  = memory_loader
            # if define memory_mini_batch_size (useful for the importance based approaches e.g. EWC MAS)
            if memory_mini_batch_size is not None:
                memory_mini_batch_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=memory_mini_batch_size,
                                                                    shuffle=False, num_workers=2, pin_memory=False,
                                                                    sampler=memory_sampler, drop_last=True)
                task['memory_mini_batch_loader'] = memory_mini_batch_loader
        # append current task (dict) to Task (list)
        Tasks.append(task)

    print('Dataloader Process: DONE at rank:{rank}'.format(rank=rank))

    return Tasks
