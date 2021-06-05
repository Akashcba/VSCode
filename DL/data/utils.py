'''
@Author: Akash Choudhary
         20BM6JP46
'''
## Utility Functions

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

def cleanerEng(x):
  x = str(x)
  x = x.lower()                     
  x = re.sub(r'[^a-z0-9]+',' ',x)   # remove punctuations
  if len(x) > 150:                  # trim length of sentences greater than 150
    x = x[:150]
  return x

def cleanerHindi(x):
  x = str(x)
  x = re.sub(r'[-.।|,?;:<>&$₹]+',' ',x)     #remove punctuations
  if len(x) > 150:                          #trim length of sentences greater than 150
    x = x[:150]
  return x

def addTokens(x,start=False):
  x.append('<END>')
  if start:
    x.insert(0,'<START>')
  return list(x)

### Vocabulory class
class vocab:

  def __init__(self,data,token=True):
    self.data = data
    if token:
      self.word2idx = {'<START>':1, '<END>':2, '<PAD>':0}     #word2idx has unique words and their corresponding values in key:value format
      self.idx2word = {1:'<START>', 2:'<END>', 0:'<PAD>'}     #idx2word is reverse of word2idx in value:key format
      self.idx = 2

    else:
      self.word2idx = {'<PAD>':0, '<END>':1}
      self.idx2word = {0:'<PAD>', 1:'<END>'}
      self.idx = 1

    self.x = []
    self.create()                                             # calling create function
    self.vocab_size = self.idx + 1

  def create(self):                                           # a function to pick up each sentence and add it to vocabulary
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

## Parallelize the Dataset
class parallelData(Dataset):
    def __init__(self):
        self.x = English_vocab.x
        self.y = Hindi_vocab.x

    def __getitem__(self,i):
        return self.x[i], self.y[i]
  
    def __len__(self):
        return len(self.x)