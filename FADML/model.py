from __future__ import print_function
import argparse
import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import argparse


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(3, 6, kernel_size=5, stride=1),
            ReLU(inplace=True),
            MaxPool2d(2,2),
            # Defining another 2D convolution layer
            Conv2d(6, 6, kernel_size=5, stride=1),
            ReLU(inplace=True),
            MaxPool2d(2,2)
        )
        self.linear_layers = Sequential(
            Linear(6*5*5, 120),
            ReLU(inplace=True),
            Linear(120, 84),
            ReLU(inplace=True),
            Linear(84, 10)
            )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
 

def convfc():
    model_arch = Net()
    return model_arch
