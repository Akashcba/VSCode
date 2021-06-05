'''
@Author : Akash Choudhary
          20BM6JP46
'''
import numpy as np
import pandas as pd
import re
from tqdm import trange, tqdm

import torch
from torch.utils.data import Dataset

## Utility functions for the model
### Data Cleaning
def cleanEng(x):
  x = str(x).strip("\n")
  x = x.lower()
  x = re.sub(r'[^a-z0-9]+',' ',x)
  if len(x) > 150:
    x = x[:150]
  return x

## Clean the dataset from all punctuations and stopwords....
def cleanHindi(x):
  x = str(x).strip("\n")
  x = re.sub(r'[-.।|,?;:<>&$₹]+',' ',x)
  if len(x) > 150:
    x = x[:150]
  return x

### Add Start and ENd Tokens
### Tokenize the dataset
def addToken(x, start=False):
  x.append('<END>')
  if start:
    x.insert(0,'<START>')
  return list(x)

### Create Vocabulory

class vocab:

  def __init__(self,data,token=True):
    self.data = data
    if token:
      self.word2idx = {'<START>':1, '<END>':2, '<PAD>':0}
      self.idx2word = {1:'<START>', 2:'<END>', 0:'<PAD>'}
      self.idx = 2

    else:
      self.word2idx = {'<PAD>':0, '<END>':1}
      self.idx2word = {0:'<PAD>', 1:'<END>'}
      self.idx = 1

    self.x = []
    self.create()
    self.vocab_size = self.idx + 1

  def create(self):
    max_len = 0;
    for sentence in  self.data:
      max_len = max(max_len, len(sentence))
      for word in sentence:
        if self.word2idx.get(word) is None:
          self.idx += 1
          self.word2idx[word] = self.idx
          self.idx2word[self.idx] = word
    
    for sentence in self.data:
      sent = []
      for word in sentence:
        sent.append(self.word2idx[word])
      
      for i in range(len(sentence),max_len+1):
        sent.append(0)
      
      self.x.append(torch.Tensor(sent))

class parallelData(Dataset):

  def __init__(self, eng_voc, hin_voc):
    self.x = eng_voc.x
    self.y = hin_voc.x

  def __getitem__(self,i):
    return self.x[i], self.y[i]
  
  def __len__(self):
    return len(self.x)
