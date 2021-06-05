'''
@Author: Akash Choudhary
         20BM6JP46
'''
## Test module for attention based seq2seq translator

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.Model(
    eng_vocab_size=5763,
    hin_vocab_size=9840,
    embedding_size = 256,
    hidden_size = 256,
    layers = 1,
    bidirection = True)
model.load_state_dict(torch.load())
model.eval()
