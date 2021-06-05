'''
@Author: Akash Choudhary
         20BM6JP46
'''
### Training File For The model
import numpy as np
import pandas as pd
import re
from tqdm import trange, tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import utils
import models
### Training files .......

hindi = []; english = []
with open('/content/drive/MyDrive/PRoj/IITB.en-hi_source.en', 'r') as eng, open('/content/drive/MyDrive/PRoj/IITB.en-hi_target.hi', 'r')as hind:
  i = 0
  try:
      for en, hi in zip(eng, hind):
        english.append(en)
        hindi.append(hi)
        print(f"Line Number : {i}",end="\r")
        i+=1
  except Exception as e:
      print(e)

data = pd.DataFrame(zip(english,hindi), columns=['source','hindi'])
del english
del hindi
print(data.head(10))

## Reducing the dataset
data = data.iloc[:95000, ]
print(data.shape)
print(data.head(20))
## Clean the dataset
data.iloc[:,0] = data['source'].apply(func=utils.cleanEng)
data.iloc[:,1] = data['hindi'].apply(func=utils.cleanHindi)
# Break Sentences into a list
data['source'] = data['source'].apply(func= lambda x : (str(x).split()))
data['hindi'] = data['hindi'].apply(func= lambda x : (str(x).split()))

print(data.head(20))

### Add Start and ENd Tokens
data.iloc[:,0] = data['source'].apply(func=utils.addToken)
data.iloc[:,1] = data['hindi'].apply(func=utils.addToken,start=True)
print("\nData Head :\n")
print(data.head())

### Storing the vocab
eng_voc = utils.vocab(data[:,0],token=False)
hin_voc = utils.vocab(data[:,1],token=True)

dataset = utils.parallelData(eng_voc, hin_voc)

## Model
model = models.Model(
    eng_voc.vocab_size,
    hin_voc.vocab_size,
    embedding_size = 256
    hidden_size = 256
    layers = 1
    bidirection = True)

loader = DataLoader(dataset, batch_size=100, shuffle=True)
it = iter(loader)
x,y = next(it)
print(x.shape,y.shape)

optimizer = optim.Adam(model.parameters(),lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=0)

train_loss = []
for epoch in trange(epochs):
  for id,(x,y) in (enumerate(tqdm(loader))):
    x = x.long().to(device)
    y = y.long().to(device)#[batch,seq]

    output = model(x,y,1)# [seq, batch, vocab]
    output = output[1:].reshape(-1,output.shape[2])
    y = y.permute(1,0)#[seq, batch]
    y = y[1:].reshape(-1)

    optimizer.zero_grad()
    loss = criterion(output,y)
    loss.backward()
    optimizer.step()

    # if id%20 == 0:
  print(f'[{epoch+1}/{epochs}] loss=>{loss.item()}')
  train_loss.append(loss.item())

## Plotting
import matplotlib.pyplot as plt
plt.plot(train_loss)
plt.show()

## Save the model
torch.save(model.state_dict(),'model_95k.pt')
