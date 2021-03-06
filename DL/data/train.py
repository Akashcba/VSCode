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
with open('/content/IITB.en-hi_source.en', 'r') as eng, open('/content/IITB.en-hi_target.hi', 'r')as hind:
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
data = data.iloc[:50000, ]

#  Saving the reduced dataset
data.to_csv("Reduced_Dataset.csv",index=False)

print(data.shape)
print(data.head(20))
## Clean the dataset
data.iloc[:,0] = data['source'].apply(func=utils.cleanerEng)
data.iloc[:,1] = data['hindi'].apply(func=utils.cleanerHindi)
# Break Sentences into a list
data['source'] = data['source'].apply(func= lambda x : (str(x).split()))
data['hindi'] = data['hindi'].apply(func= lambda x : (str(x).split()))

print(data.head(20))

### Add Start and ENd Tokens
data.iloc[:,0] = data['source'].apply(func=utils.addTokens)
data.iloc[:,1] = data['hindi'].apply(func=utils.addTokens,start=True)
print("\nData Head :\n")
print(data.head())

#  Saving the reduced & cleaned dataset
data.to_csv("Reduced_Dataset.csv",index=False)

### Storing the vocab
eng_voc = utils.vocab(data.iloc[:,0],token=False)
hin_voc = utils.vocab(data.iloc[:, 1],token=True)

print("eng_voc size: ", eng_voc.vocab_size)
print("hin_voc size: ", hin_voc.vocab_size)

dataset = utils.parallelData(eng_voc, hin_voc)

## Model
model = models.Model(
    eng_voc.vocab_size,
    hin_voc.vocab_size,
    embedding_size = 256,
    hidden_size = 256,
    layers = 1,
    bidirection = True)
learning_rate = 0.0006
loader = DataLoader(dataset, batch_size=100, shuffle=True)
it = iter(loader)
x,y = next(it)
print(x.shape,y.shape)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
optimizer = optim.Adam(model.parameters(),lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=0)

epochs = 25
train_loss = []
for epoch in trange(epochs):
  for id,(x,y) in (enumerate(tqdm(loader))):
    x = x.long().to(device)
    y = y.long().to(device)

    output = model(x,y,1)
    output = output[1:].reshape(-1,output.shape[2])
    y = y.permute(1,0)
    y = y[1:].reshape(-1)

    optimizer.zero_grad()
    loss = criterion(output,y)
    loss.backward()
    optimizer.step()
  print(f'[{epoch+1}/{epochs}] loss=>{loss.item()}')
  train_loss.append(loss.item())

## Plotting
import matplotlib.pyplot as plt
plt.plot(train_loss)
plt.show()

## Save the model
torch.save(model.state_dict(),'model_50k_.pt')