#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import dataloader
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from tabulate import tabulate

device = torch.device("cuda:0")


# In[6]:


class DeepConvNet(nn.Module):
    def __init__(self, activate):
        super(DeepConvNet, self).__init__()
        self.C = 2
        self.T = 750
        self.N = 2
        
        #Reshape
        self.conv1 = nn.Conv2d(1, 25, kernel_size=(1, 5))
        self.conv2 = nn.Conv2d(25, 25, kernel_size=(self.C, 1))
        self.batchnorm3 = nn.BatchNorm2d(25, eps=1e-05, momentum=0.1)
        self.elu4 = activate
        self.maxpool5 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout6 = nn.Dropout(p=0.5)
        self.conv7 = nn.Conv2d(25, 50, kernel_size=(1, 5))
        self.batchnorm8 = nn.BatchNorm2d(50, eps=1e-05, momentum=0.1)
        self.elu9 = activate
        self.maxpool10 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout11 = nn.Dropout(p=0.5)
        self.conv12 = nn.Conv2d(50, 100, kernel_size=(1, 5))
        self.batchnorm13 = nn.BatchNorm2d(100, eps=1e-05, momentum=0.1)
        self.elu14 = activate
        self.maxpool15 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout16 = nn.Dropout(p=0.5)
        self.conv17 = nn.Conv2d(100, 200, kernel_size=(1, 5))
        self.batchnorm18 = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1)
        self.elu19 = activate
        self.maxpool20 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout21 = nn.Dropout(p=0.5)
        self.linear22 = nn.Linear(in_features=8600, out_features=2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batchnorm3(x)
        x = self.elu4(x)
        x = self.maxpool5(x)
        x = self.dropout6(x)
        x = self.conv7(x)
        x = self.batchnorm8(x)
        x = self.elu9(x)
        x = self.maxpool10(x)
        x = self.dropout11(x)
        x = self.conv12(x)
        x = self.batchnorm13(x)
        x = self.elu14(x)
        x = self.maxpool15(x)
        x = self.dropout16(x)
        x = self.conv17(x)
        x = self.batchnorm18(x)
        x = self.elu19(x)
        x = self.maxpool20(x)
        x = self.dropout21(x)
        
        x = x.view(x.shape[0], -1)
        x = self.linear22(x)
        return x


# In[7]:


class EEGNet(nn.Module):
    def __init__(self, activate):
        super(EEGNet, self).__init__()
        
        #Firstconv
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        #DepthwiseConv
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act2 = activate #nn.ELU(alpha=1.0)
        self.avgpool2 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0)
        self.dropout2 = nn.Dropout(p=0.25)
        
        #SperableConv
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act3 = activate #nn.ELU(alpha=1.0)
        self.avgpool3 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0)
        self.dropout3 = nn.Dropout(p=0.25)
        
        #Classify
        self.linear4 = nn.Linear(in_features=736, out_features=2, bias=True)

    def forward(self, x):
        #Firstconv
        x = self.conv1(x)
        x = self.batchnorm1(x)
        
        #DepthwiseConv
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.act2(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        
        #SperableConv
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.act3(x)
        x = self.avgpool3(x)
        x = self.dropout3(x)
        
        #Classify
        x = x.view(x.shape[0], -1)
        x = self.linear4(x)
        
        return x


# In[11]:


criterion = nn.CrossEntropyLoss()
train_data, train_label, test_data, test_label = dataloader.read_bci_data()
dataset_train = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_label))
loader_train = DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=4)
dataset_test = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_label))
loader_test = DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=4)

def train(net, optimizer):
    net.train()
    count = 0
    
    for data in loader_train:
        inputs = data[0].to(device, dtype=torch.float)
        labels = data[1].to(device, dtype=torch.long)
        optimizer.zero_grad()
        
        output = net(inputs)
        loss = criterion(output, labels)
        loss.backward()
        
        count += (output.max(dim=1)[1] == labels).sum().item()
        optimizer.step()
    acc = 100 * (count / len(dataset_train))
    return net, acc

def test(net, optimizer):
    net.eval()
    count = 0
    
    for data in loader_test:
        inputs = data[0].to(device, dtype=torch.float)
        labels = data[1].to(device, dtype=torch.long)
        optimizer.zero_grad()
        
        output = net(inputs)
        loss = criterion(output, labels)
        loss.backward()
        
        count += (output.max(dim=1)[1] == labels).sum().item()
        optimizer.step()
    acc = 100 * (count / len(dataset_test))
    return net, acc

def run(net, name):
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    best = 0
    index = []
    arr_train = []
    arr_test = []
    for epoch in range(300):
        net, tmp_train = train(net, optimizer)
        net, tmp_test = test(net, optimizer)
        index.append(epoch)
        arr_train.append(tmp_train)
        arr_test.append(tmp_test)
        if(tmp_test > best):
            best = tmp_test
        if(epoch % 100 == 0):
            print(name, 'in epoch:', epoch, tmp_test)
    #plt.plot(index, arr_train, label=(name + ' train'))
    plt.plot(index, arr_test, label=(name))
    torch.save(net.state_dict(), './' + name + '.model')
    return best


net1 = EEGNet(nn.ReLU()).to(device)
net2 = EEGNet(nn.LeakyReLU()).to(device)
net3 = EEGNet(nn.ELU(alpha=1.0)).to(device)

net4 = DeepConvNet(nn.ReLU()).to(device)
net5 = DeepConvNet(nn.LeakyReLU()).to(device)
net6 = DeepConvNet(nn.ELU(alpha=1.0)).to(device)

table = [[]]
table.append(['', 'ReLU', 'Leaky ReLU', 'ELU'])
table.append(['EEGNET', str(run(net1, 'EEGNET ReLU')) + '%', str(run(net2, 'EEGNET Leaky ReLU')) + '%', str(run(net3, 'EEGNET ELU')) + '%'])
table.append(['DeepConvNet', str(run(net4, 'DeepConvNet ReLU')) + '%', str(run(net5, 'DeepConvNet Leaky ReLU')) + '%', str(run(net6, 'DeepConvNet ELU')) + '%'])
print(tabulate(table))

plt.rcParams["figure.figsize"] = (8, 12)
plt.legend()
plt.show()        


# In[ ]:




