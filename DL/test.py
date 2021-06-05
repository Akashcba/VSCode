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
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

import utils
import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.Model(
    eng_vocab_size=5763,
    hin_vocab_size=9840,
    embedding_size = 256,
    hidden_size = 256,
    layers = 1,
    bidirection = True)
model.load_state_dict(torch.load('/content/VSCode/DL/model_95k.pt'))
model.eval()

### Loading the reduced dataset for builind the vocabulory
data = pd.read_csv("/content/VSCode/DL/Reduced_Dataset.csv")
English_vocab = utils.vocab(data['source'],token=False)
Hindi_vocab = utils.vocab(data['hindi'],token=True)

### Load the Test File .....
source = []
with open('/content/VSCode/DL/test.en', 'r') as eng:
  i = 0
  try:
      for en in eng :
        source.append(en)
        print(f"Line Number : {i}",end="\r")
        i+=1
  except Exception as e:
      print(e)

test_df = pd.DataFrame(source, columns=['source'])
del source
print(test_df.head(10))

### Cleaning the Input test data ...
test_df['source'] = test_df['source'].apply(func=utils.cleanEng)

# Break Sentences into a list
test_df['source'] = test_df['source'].apply(func= lambda x : (str(x).split()))

### Add Start and ENd Tokens
test_df['source'] = test_df['source'].apply(func=utils.addToken)

print("\nData Head :\n")
print(test_df.head())

#  Saving the reduced dataset
test_df.to_csv("test_converted.csv",index=False)

## Predict function
def prediction(x):
    for idx in x:
        if idx == 0:
            break
        print(English_vocab.idx2word[int(idx)],end=' ')
    print()
    x = x.long().reshape(1,-1).to(device)
    ans = translate(x)
    res = []
    for id in ans:
        res.append(Hindi_vocab.idx2word.get(id))
    return res

def translate(input):
      #input = batch of english sentences[batch, sentece(padded)]
      with torch.no_grad():
        guess = []
        encoder_states, hidden, cell = model.encoder(input)
        # x = torch.ones((1)).float().to(device) # <START> token
        x = torch.ones((1)).long().to(device)
        while True:
          out, hidden, cell = model.decoder(x, hidden, cell, encoder_states) #out shape = [batch, vocab_size]
          x = out.argmax(1)# taking the word with max value(confidence)  shape = [batch of words]
          guess.append(int(x[0].detach().cpu()))
          if x == 2:
            break
      return guess

def get(sent):
  # sentence = sentence.lower()
  # sent = sentence.split()
  # sent.append('<END>')
  # print(sent)
  toks = []
  for word in sent:
    if English_vocab.word2idx.get(word) is None:
      toks.append(English_vocab.word2idx['the'])            # the words which are not there in the vocabulary are added with 'the' padding
    else:
      toks.append(English_vocab.word2idx[word])
  # print(toks)
  sent = torch.tensor(toks).float()
  res = prediction(sent)
  # print(res)
  return res

tdata = test_df['source'].tolist()
input_list = [ ]
model_output = [ ]
for i in tqdm(range(len(tdata))):
  input_list.append(tdata[i][:-1])
  model_output.append((get(tdata[i])[:-1]))

print("Source\t", "Output\n")
for en,hi in zip(input_list, model_output):
    print(en, hi)
