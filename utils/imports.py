from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import json
import time
import glob
from collections import defaultdict
import sys, os
import pickle
import ipdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

pylab.rcParams.update({'legend.fontsize': 'large',
                       'font.size'      : 14,
                       'figure.figsize' : (18, 4),
                       'axes.labelsize' : 'medium',
                       'axes.titlesize' : 'medium',
                       'axes.grid'      : 'on',
                       'xtick.labelsize': 'large',
                       'ytick.labelsize': 'large'})

language_models = {'multi-qa-mpnet-base-dot-v1': 'multiqampnet',
                   'all-MiniLM-L12-v2': 'MiniLmv2',
                   'all-distilroberta-v1': 'dr1'}


def print_runtime(start, pflag=True):
    end = time.time()
    if pflag:
        print(f'Runtime: {int((end-start)//60)} min {int((end-start)%60):2d} sec')
        return None
    else:
        return f' (...Runtime: {int((end-start)//60)} min {int((end-start)%60):2d} sec)'


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def train(model, optimizer, train_loader, train_losses, train_counter, 
          test_loader, test_losses, test_counter, 
          batch_size, epoch, wanna_test=True):
    
    model.train()
    for q, (batch_data, batch_target) in enumerate(train_loader):
        if wanna_test==True and q in [0, len(train_loader) // 2]:
            test(model, test_loader, test_losses, test_counter, (
                (q+1)*batch_size) / len(train_loader.dataset) + epoch)
            model.train()
            
        optimizer.zero_grad()
        output = model(batch_data) 
        loss = F.nll_loss(output, batch_target)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        train_counter.append(((q+1)*batch_size) / len(train_loader.dataset) + epoch)

        arr_train_counter = np.array(train_counter)
        idx = np.argwhere((arr_train_counter > epoch) & (arr_train_counter <= epoch+1))[:,0]
        epoch_train_loss = np.mean(np.array(train_losses)[idx])

        print(f'epoch: {epoch} \tTrain_Loss: {loss.item():.6f}', end='\r')
        
    print(f'epoch: {epoch} \tTrain_Loss: {epoch_train_loss:.6f}', end='  ')
        
    # torch.save(model.state_dict(), '../models/model.pth')
    # torch.save(optimizer.state_dict(), '../models/optimizer.pth')


def test(model, test_loader, test_losses, test_counter, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for q, (batch_data, batch_target) in enumerate(test_loader):
            output = model(batch_data)
            test_loss += F.nll_loss(output, batch_target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(batch_target.data.view_as(pred)).sum()

        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        test_counter.append(epoch)
        # print(f'Test loss: {test_loss:.4f}, Accuracy: ({100. * correct / len(test_loader.dataset):.2f}%)' + '  '* 30, end='\n')


def get_train_loader(batch_size, shuffle=False):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('../data/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size, 
        shuffle=shuffle)
    
    return train_loader


def get_test_loader():
    batch_size_test = 10000
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('../data/', 
                                                                         train=False, 
                                                                         download=True,
                                                                         transform=torchvision.transforms.Compose([
                                                                             torchvision.transforms.ToTensor(),
                                                                             torchvision.transforms.Normalize(
                                                                                 (0.1307,), (0.3081,))
                                                                         ])),
                                              batch_size=batch_size_test, 
                                              shuffle=False)
    
    return test_loader


def plotter_random_samples(test_loader, num_examples=6):    

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    fig = plt.figure(figsize=(16, num_examples//2))
    for i, idx in enumerate(sorted(np.random.choice(len(example_data), num_examples, False))):
        plt.subplot(num_examples//6, 6, i+1)
        plt.imshow(example_data[idx][0], cmap='gray', interpolation='none')
        plt.title(f'idx={idx}, label: {example_targets[idx]}', fontsize=14)
        plt.xticks([])
        plt.yticks([])

