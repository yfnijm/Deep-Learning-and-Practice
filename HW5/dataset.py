import torch
import os
import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob

default_transform = transforms.Compose([
    transforms.ToTensor(),
    ])

class bair_robot_pushing_dataset(Dataset):
    def __init__(self, args, mode='train', transform=default_transform):
        assert mode == 'train' or mode == 'test' or mode == 'validate'
        
        self.ordered = False
        if mode == 'test':
            self.ordered = True 
            
        self.seed_is_set = False
        #self.data_dir = os.path.join(data_root, mode)
        self.data_dir = os.path.join('/DATA/2022DLP/LAB5/data/processed_data', mode)
        self.filenames = glob.glob(os.path.join(self.data_dir, '*'))
        
        more_filenames = []
        for name in self.filenames:
            tmp = glob.glob(os.path.join(name, '*'))
            for tmp2 in tmp:
                more_filenames.append(tmp2)
        
        self.filenames = more_filenames
        self.seq_len = args.n_past + args.n_future
        self.d = 0
        
        self.cur_dirs = ''
        self.transform = transform
        #raise NotImplementedError
        
    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
            
    def __len__(self):
        return len(self.filenames)
        #raise NotImplementedError
       
    def get_seq(self):
        if self.ordered:
            d = self.filenames[self.d]
            if self.d == len(self.filenames) - 1:
                self.d = 0
            else:
                self.d+=1
        else:
            d = self.filenames[np.random.randint(len(self.filenames))]
        self.cur_dirs = d
        
        image_seq = []
        for i in range(30):
            fname = '%s/%d.png' % (d, i)
            im = Image.open(fname)
            im = self.transform(im).reshape((1, 3, 64, 64))
            image_seq.append(im)
        image_seq = torch.from_numpy(np.concatenate(image_seq))
        
        return image_seq
        #raise NotImplementedError
        
    def get_csv(self):
        cond_seq = []
        with open(os.path.join(f'{self.cur_dirs}/actions.csv'), newline='') as csvfile:
            rows = csv.reader(csvfile)
            actions = []
            for row in rows:
                actions.append(row)
        
        with open(os.path.join(f'{self.cur_dirs}/endeffector_positions.csv'), newline='') as csvfile:
            rows = csv.reader(csvfile)
            endeffector_positions = []
            for row in rows:
                endeffector_positions.append(row)
                
        for i in range(30):
            concat = actions[i]
            concat.extend(endeffector_positions[i])
            cond_seq.append(concat)
        cond_seq = torch.Tensor(np.array(cond_seq, dtype=float))
        return cond_seq
        #raise NotImplementedError

    def __getitem__(self, index):
        self.set_seed(index)
        seq = self.get_seq()
        cond = self.get_csv()
        return seq, cond   