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


""" index of test sample in test_loader is 9799, with label == 1
    Other alternatives:
            idx_xt = 900, label = 1
            idx_xt = 3906, label = 1
            idx_xt = 703, label = 7
            idx_xt = 1477, label = 7
            idx_xt = 9036, label = 7
            idx_xt = 5835, label = 7
            
"""

# 'output0' is the network output, -ln(p) of X_test_original[idx_xt] forward propped on 
# the model trained for 10 epochs on the entirety train set.
# idx_xt = 9799 
# output0 = np.array([-1.7283735e+01, -1.3398226e-04, -9.9062271e+00, -1.4030624e+01,
#                     -1.0261919e+01, -1.6715294e+01, -1.5527508e+01, -1.0730595e+01,
#                     -1.0553279e+01, -1.5390485e+01]) 

idx_xt = 5835 
# output0 = np.array([ -4.5651717 ,  -9.473113  ,  -1.6655078 ,  -5.235294  ,
#                      -3.545306  ,  -9.924194  , -11.835953  ,  -0.78873056,
#                      -3.7437139 ,  -1.2444761 ])  


def print_runtime(start, pflag=True):
    end = time.time()
    if pflag:
        print(f'Runtime: {int((end-start)//60)} min {int((end-start)%60):2d} sec')
        return None
    else:
        return f' (...Runtime: {int((end-start)//60)} min {int((end-start)%60):2d} sec)'


def softmax(x):
    return(np.exp(x)/np.exp(x).sum())


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
            test(model, test_loader, test_losses, test_counter, 
                 ((q+1)*batch_size) / len(train_loader.dataset) + epoch)
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

    print(f'epoch: {epoch}   Train_Loss: {epoch_train_loss:.6f}', end='')


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
        acc = 100. * correct / len(test_loader.dataset)
        # print(f'Test loss: {test_loss:.4f}, Accuracy: ({100. * correct / len(test_loader.dataset):.2f}%)' + '  '* 30, end='\n')
        return acc


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


def plotter_random_samples(test_loader, num_examples=6, target_label=7):    

    examples = enumerate(test_loader)
    _, (example_data, example_targets) = next(examples)
    target_indices = np.argwhere(example_targets == target_label)[0]

    fig = plt.figure(figsize=(16, num_examples//2))
    for q, idx in enumerate(sorted(np.random.choice(target_indices, num_examples, False))):
        plt.subplot(num_examples//6, 6, q+1)
        plt.imshow(example_data[idx][0], cmap='gray', interpolation='none')
        plt.title(f'idx={idx}, label: {example_targets[idx]}', fontsize=14)
        plt.xticks([])
        plt.yticks([])


def plotter_ranked_sample_importance(X_train_original, X_test_original, target_label, grad, num_examples=96):  
    ranked_idx = [i for (i, g) in sorted(grad.items(), key=lambda x:-x[1][target_label])]
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(X_test_original[idx_xt][0], cmap='gray', interpolation='none')
    plt.title(f'idx_xt={idx_xt}, label:{target_label}', fontsize=14)
    plt.xticks([])
    plt.yticks([])
    
    fig = plt.figure(figsize=(16, 2 * (num_examples//6) + 6))
    for q, idx in enumerate(ranked_idx[:num_examples]):
        plt.subplot(num_examples//6, 6, q+1)
        plt.imshow(X_train_original[idx][0], cmap='gray', interpolation='none')
        plt.title(f'idx={idx}, label: {target_label}', fontsize=14)
        plt.xticks([])
        plt.yticks([])


def train_with_torch_tensors(model, optimizer, train_losses, train_counter, test_loader, test_losses, test_counter,
                             X_train, y_train, batch_size, epoch):

    model.train()
    # for-loop goes for 938 cycles.
    for batch_no in range(int(np.ceil(len(y_train) / batch_size))):
        batch_data = X_train[batch_no*batch_size : (batch_no+1) * batch_size]
        batch_target = y_train[batch_no*batch_size : (batch_no+1) * batch_size]

        optimizer.zero_grad()
        output = model(batch_data) 
        loss = F.nll_loss(output, batch_target)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        train_counter.append(((batch_no+1)*batch_size) / len(y_train) + epoch)

        arr_train_counter = np.array(train_counter)
        idx = np.argwhere((arr_train_counter > epoch) & (arr_train_counter <= epoch+1))[:,0]
        epoch_train_loss = np.mean(np.array(train_losses)[idx])

    acc = test(model, test_loader, test_losses, test_counter, epoch+1)
    print(f'epoch: {epoch}   Train_Loss: {epoch_train_loss:.6f}   Test_Loss: {test_losses[-1]:.6f}   Test_Accuracy: {acc:.2f}%')


def save_output0(model, idx_xt, X_test_original):
    """ Saves softmax probabilities of test sample `X_test_original[idx_xt]` under ../models 
    """
    
    model.eval()
    print(f'idx_xt:{idx_xt}')
    output0 = np.exp(model(X_test_original[idx_xt:idx_xt+1]).detach().numpy())[0]
    fname = f'../models/output0_idx_xt{idx_xt}.npy'
    with open(fname, 'wb') as f:
        np.save(f, output0)
        
    print(f'output0 saved to {fname}')
    

    
def load_output0(idx_xt):
    """ Returns softmax probabilities of test sample `X_test_original[idx_xt]` that's saved to disk
    """
    
    with open(f'../models/output0_idx_xt{idx_xt}.npy', 'rb') as f:
        output0 = np.load(f)
    return output0


def initialize_grad_dict():
    fname = '../models/grad_dict.pkl'
    if glob.glob(fname):
        with open(fname, 'rb') as f:
            grad = pickle.load(f)
        print("loaded grad from  ../models/grad_dict.pkl")
    else:
        grad = dict()
        print('initialized empty dictionary: grad = dict()')
        
    return grad






