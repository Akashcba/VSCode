'''
@Author: Akash Choudhary
         20BM6JP46
'''

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
#from torch.utils.data import DataLoader
import torch
import random

import utils
import models

## Input training data needed for building the vocabulory
data = pd.read_csv("/content/VSCode/DL/Reduced_Dataset.csv")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Data Preprocessing
data.iloc[:,0] = data['source'].apply(func=utils.cleanerEng)
data.iloc[:,1] = data['hindi'].apply(func= utils.cleanerHindi)
data.iloc[:,0] = data['source'].apply(func= lambda x : (str(x).split()))
data.iloc[:,1] = data['hindi'].apply(func= lambda x : (str(x).split()))

## Tokenize the data
data.iloc[:,0] = data['source'].apply(func= utils.addTokens,start=False)
data.iloc[:,1] = data['hindi'].apply(func= utils.addTokens,start=True)


data = data.iloc[:50000, ].values

#print(data[:,1].shape)

English_vocab = utils.vocab(data[:,0],token=False)
Hindi_vocab = utils.vocab(data[:,1],token=True)

for idx in Hindi_vocab.x[5]:
  print(Hindi_vocab.idx2word[int(idx)],end=' ')

## Modeling
model = models.Model()
model.load_state_dict(torch.load('/content/VSCode/DL/data/model50thousand.pt'))
## Loaded MOdel
print(model.eval())


test_en=open('/content/VSCode/DL/data/test.en', encoding='utf8').read().split('\n')

df_test=pd.DataFrame(test_en,columns=['source'])

print(df_test.head())

## Data Preprocessing
df_test.iloc[:,0] = df_test['source'].apply(func=utils.cleanerEng)
df_test.iloc[:,0] = df_test['source'].apply(func= lambda x : (str(x).split()))
## Tokenize the dataset
df_test.iloc[:,0] = df_test['source'].apply(func= utils.addTokens,start=False)

print(df_test.head())

tdata = df_test.values
test_dataset = utils.vocab(tdata[:,0],token=False)

## Prediction

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
      res.append(Hindi_vocab.idx2word[id])
    
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


input_list = [ ]
model_output = [ ]

#for j in trange(int(tdata.shape[0])):
for j in tqdm(range(int(tdata.shape[0]/2))):
  try:
    res = get(tdata[j,0])[:-1]
  except Exception as e:
    continue
  print('English Sentence converted and Predicted Hindi Sentence:')
  #print(tdata[j,0][:-1])
  print(res)
'''
i=i+1

print("Source\t", "Output\n")
for en,hi in zip(input_list, model_output):
    print(en, hi)
'''