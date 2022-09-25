#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dataloader 
import torch 
import torchvision.models as models
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.optim as optim
import numpy as np
from time import sleep
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay


# In[2]:


torch.cuda.set_device(1)
print(torch.cuda.is_available())
device=torch.device('cuda')


# In[3]:


dataset_train = dataloader.RetinopathyLoader(root = 'data', mode = 'train')
loader_train = DataLoader(dataset=dataset_train, batch_size=32, shuffle=True, num_workers=32)

dataset_test = dataloader.RetinopathyLoader(root = 'data', mode = 'test')
loader_test = DataLoader(dataset=dataset_test, batch_size=16, shuffle=False, num_workers=4)


# In[ ]:


def normal_and_plot(confusion_matrix):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
    for j in range(confusion_matrix.shape[1]):
        total_case = 0
        for i in range(confusion_matrix.shape[0]):
            total_case += confusion_matrix[j, i]
        for i in range(confusion_matrix.shape[0]):
            ax.text(i, j, '{:.2f}'.format(confusion_matrix[j, i] / total_case), va='center', ha='center')
    plt.show()
    return fig

criterion = nn.CrossEntropyLoss()
def train():
    model.train()
    count = 0
    mat = np.zeros((5, 5))
    for inputs, labels in loader_train:
        
        inputs,label= inputs.to(device), labels.to(device,dtype=torch.long)
        output = model(inputs.cuda())
        count += (output.max(dim=1)[1] == labels.cuda()).sum().item()
        
        optimizer.zero_grad()
        loss = criterion(output.cuda(), labels.cuda())
        loss.backward()
        optimizer.step()
        for i in range(len(labels)):
            mat[int(labels[i])][int(output.max(dim=1)[1][i])]+=1
    acc = 100. * count / len(loader_train.dataset)
    print(mat)
    normal_and_plot(mat)
    print(count, len(loader_train.dataset))
    return acc

def test():
    model.eval()
    count = 0
    mat = np.zeros((5, 5))
    for inputs, labels in loader_test:
        inputs,label= inputs.to(device), labels.to(device,dtype=torch.long)
        output = model(inputs.cuda())
        count += (output.max(dim=1)[1] == labels.cuda()).sum().item()
        for i in range(len(labels)):
            mat[int(labels[i])][int(output.max(dim=1)[1][i])]+=1
    acc = 100. * count / len(loader_test.dataset)
    print(mat)
    normal_and_plot(mat)
    print(count, len(loader_test.dataset))
    return acc


def run(name):
    #net.load_state_dict(torch.load('./' + name + '.model'))
    best = 0
    index = []
    arr_train = []
    arr_test = []
    for epoch in range(5):
        print(name, "epoch: ", epoch)
        acc_train = train()
        print('acc_train =', str(acc_train))
        acc_test = test()
        print('acc_test =', str(acc_test))
        index.append(epoch)
        arr_train.append(acc_train)
        arr_test.append(acc_test)
        if(acc_test > 82):
            torch.save(model.state_dict(), './' + name + '_82' + '.model')
        if(acc_test > 83):
            torch.save(model.state_dict(), './' + name + '_83' + '.model')
    plt.plot(index, arr_train, label=(name + ' train'))
    plt.plot(index, arr_test, label=(name + 'test'))
    plt.show()
    torch.save(model.state_dict(), './' + name + '.model')
    return best

model = models.resnet50(pretrained=True)
model.fc = nn.Linear(in_features=model.fc.in_features, out_features=5)
model.cuda()
#print(model)
optimizer = optim.SGD(model.parameters(), lr=1e-3,momentum=0.9,weight_decay=5e-4)
run("resnet50_pretrained")



model = models.resnet50(pretrained=False)
model.fc = nn.Linear(in_features=model.fc.in_features, out_features=5)
model.cuda()
#print(model)
optimizer = optim.SGD(model.parameters(), lr=1e-3,momentum=0.9,weight_decay=5e-4)
run("resnet50_non-pretrained")


model = models.resnet18(pretrained=True)
model.fc = nn.Linear(in_features=model.fc.in_features, out_features=5)
model.cuda()
#print(model)
optimizer = optim.SGD(model.parameters(), lr=1e-3,momentum=0.9,weight_decay=5e-4)
run("resnet18_pretrained")


model = models.resnet18(pretrained=False)
model.fc = nn.Linear(in_features=model.fc.in_features, out_features=5)
model.cuda()
#print(model)
optimizer = optim.SGD(model.parameters(), lr=1e-3,momentum=0.9,weight_decay=5e-4)
run("resnet18_non-pretrained")


# In[ ]:




