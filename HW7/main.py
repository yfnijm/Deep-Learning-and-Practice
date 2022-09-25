import torch
import torch.nn as nn
import copy
import os
import numpy as np
import copy
import math
import json
import random
from dataloader import dataLoader
from torch.utils.data import DataLoader
from model import Generator, Discriminator, weights_init
from torchvision.utils import save_image
from evaluator import evaluation_model
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

if __name__=='__main__':
    nz = 100
    nc = 200
    ngf = 64
    ndf = 64
    image_shape = (64, 64, 3)
    epochs = 5
    lr = 0.0002
    beta1 = 0.5
    b_size = 16
    
    manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    dataset_train = dataLoader(root='/DATA/2022DLP/LAB7/', mode='train.json')
    loader_train = DataLoader(dataset_train, batch_size=b_size, shuffle=True, num_workers=4)

    netG = Generator(nz, nc).to(device)
    weights_init(netG)
    netD = Discriminator(image_shape).to(device)
    weights_init(netD)

    #Loading model
    #netG.load_state_dict(torch.load('models/best.pth'))
    
    criterion  = nn.BCELoss()
    with open('objects.json', 'r') as file:
        total_label = json.load(file)
        
    with open('test.json','r') as file:
        tmp = json.load(file)
    
    label_test = torch.zeros(len(tmp),len(total_label), device=device)
    for i in range(len(tmp)):
        for j in tmp[i]:
            label_test[i,int(total_label[j])] = 1.
            
        
    with open('new_test.json','r') as file:
        tmp2 = json.load(file)
        
    new_label_test = torch.zeros(len(tmp2),len(total_label), device=device)
    for i in range(len(tmp2)):
        for j in tmp2[i]:
            new_label_test[i,int(total_label[j])] = 1.
    #print(label_test)
    #print(new_label_test)
    optimG = torch.optim.Adam(netG.parameters(),lr,betas=(beta1, 0.99))
    optimD = torch.optim.Adam(netD.parameters(),lr,betas=(beta1, 0.99))

    scores = []
    new_scores = []
    G_losses = []
    D_losses = []
    best_score = 0
    for epoch in range(epochs):
        G_total_loss=0
        D_total_loss=0
        
        netG.train()
        netD.train()
        for i, (real_img, condition) in enumerate(loader_train):
            real_img = real_img.to(device)
            condition = condition.to(device)
            netG.train()
            netD.train()
            
            b_size = len(real_img)
            real_label = torch.full((b_size,), 1, dtype=torch.float, device = device) - torch.randn(b_size, device = device) * 0.05
            fake_label = torch.full((b_size,), 0, dtype=torch.float, device = device) + torch.randn(b_size, device = device) * 0.05
            
            optimD.zero_grad()
            pred = netD(real_img, condition)
            loss_real = criterion(pred, real_label)
            z = torch.randn(b_size, nz, device = device)
            fake_img = netG(z,condition)
            pred = netD(fake_img, condition)
            loss_fake = criterion(pred, fake_label)
            D_loss = loss_real + loss_fake
            D_loss.backward()
            optimD.step()
            D_total_loss += D_loss.mean().item()

            optimG.zero_grad()
            z = torch.randn(b_size, nz, device = device)
            fake_img = netG(z, condition)
            pred = netD(fake_img,condition)
            G_loss = criterion(pred, real_label)
            G_loss.backward()
            optimG.step()
            G_total_loss += G_loss.mean().item()
        
        print(f'epoch: {epoch}') 
        print(f'loss(g)= {G_total_loss/len(loader_train):.3f}') 
        print(f'loss(d)= {D_total_loss/len(loader_train):.3f}') 
        G_losses.append(G_total_loss/len(loader_train))
        D_losses.append(D_total_loss/len(loader_train))
        
        
        netG.eval()
        netD.eval()
        fixed_noise = torch.randn(len(label_test), nz, 1, 1, device = device)
        evaluator = evaluation_model()
        with torch.no_grad():
            fake_img = netG(fixed_noise, label_test)
        score = evaluator.eval(fake_img, label_test)
        scores.append(score)
        print(f'score= {score:.3f}')
        save_image(fake_img, os.path.join('pic/', f'{epoch}.png'), nrow=8, normalize=True)                
                        
        new_fixed_noise = torch.randn(len(new_label_test), nz, 1, 1, device = device)
        with torch.no_grad():
            new_fake_img = netG(fixed_noise, new_label_test)                
        new_score = evaluator.eval(new_fake_img, new_label_test)
        new_scores.append(new_score)
        print(f'new score= {new_score:.3f}')
        save_image(new_fake_img, os.path.join('pic/', f'{epoch}.png'), nrow=8, normalize=True)                
                        
                        
        if (score + new_score) / 2 > best_score:
            best_score = (score + new_score) / 2 
            torch.save(netG.state_dict(), f'models/best_{epoch}.pth')
            
    plt.figure(figsize=(10,5))
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig(fname='Loss_figure.png')

    plt.figure(figsize=(10,5))
    plt.title("Evalation testing")
    plt.plot(scores, label="score")
    plt.plot(new_scores, label="new score")
    plt.xlabel("iterations")
    plt.ylabel("Score")
    plt.legend()
    plt.show()
    plt.savefig(fname='Test_figure.png')                    
    