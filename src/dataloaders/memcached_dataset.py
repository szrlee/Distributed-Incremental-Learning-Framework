import mc
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import io
from PIL import Image

def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    with Image.open(buff) as img:
        img = img.convert('RGB')
    return img
 
class McDataset(Dataset):
    def __init__(self, root_dir, meta_file, transform=None, n_class=20):
        self.root_dir = root_dir
        self.transform = transform
        with open(meta_file) as f:
            lines = f.readlines()
        print("building dataset from %s"%meta_file)
        self.num = len(lines)
        self.metas = []
        for line in lines:
            dataline = line.strip().split()
            # path, cls = line.rstrip().split()
            path = dataline[0]
            t_num = len(dataline) - 1
            assert t_num % 2 == 0
            t_cls = dataline[1:]
            t_cls = t_cls[::2]
            # print(t_cls)
            # b_cls = [int(0)] * n_class
            # for index in t_cls:
                # b_cls[int(index)] = int(1)
                
            self.metas.append((path, t_cls))
        print("read meta done")
 
    def __len__(self):
        return self.num
 
    def __getitem__(self, idx):
        filename = self.root_dir + '/' + self.metas[idx][0]
        t_cls = self.metas[idx][1]
        b_cls = [int(0)] * 4193
        for index in t_cls:
            b_cls[int(index)] = int(1)
        b_cls = torch.FloatTensor(b_cls)
        
        # for id, v in enumerate(t_cls):
            # print('label{}: {}'.format(id, v))
        # for id, v in enumerate(b_cls):
            # print('binary{}: {}'.format(id, v))
        ## memcached
        server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
        client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
        mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
        value = mc.pyvector()
        mclient.Get(filename, value)
        value_str = mc.ConvertBuffer(value)
        img = pil_loader(value_str)
        #img = np.zeros((350, 350, 3), dtype=np.uint8)
        #img = Image.fromarray(img)
        #cls = 0
        
        ## transform
        if self.transform is not None:
            img = self.transform(img)
        return img, b_cls
