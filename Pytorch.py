#Pytorch Tutorial
import torch
#torch.cuda.is_available() -> True if cuda is available
print(torch.cuda.is_available())
## Creating tensors
'''
In pytorch everything is a Tensor
'''
x = torch.empty(3) # 1 D vector with 3 elements
y = torch.empty(2,3,4) # 3 D tensor
print(x)
print(y)
## Torch with random values
a = torch.rand(2,3)
print(a)
## Zero Tensor
b = torch.zeros(2,3)
print(b)
# Ones tensor
c = torch.ones(2,3)
print(c)
## check data type of a tensor
print(x.dtype)
## Create tensor of certain datatype
c = torch.ones(2,3, dtype=torch.int)
print(c.dtype)
# check size of tensor
print(c.size())
# Python list to tensor
x = torch.tensor([2,3,4,5,6])
print(x)
### Operations
x=torch.rand(2,2)
y=torch.rand(2,2)
z=x+y  ## z=torch.add(x,y)
print(z)
y.add_(x) ## Trailing underscore Represent an inplacing function
## -, *, / => Operations
## Slicing operations in tensor
x=torch.rand(5,3)
print(x[:,:2])
## Each value is returned as a tensor object
## To retreive the actual value we need item function
## But item func requires only single element
print(x[1,1].item())
## Reshaping a tensor
y=x.view(15) ## 1D Tensor with 15 = 3*5 elements
## or do -1 so that torch will determine its size
print(y)
y=x.view(-1)
print(y)
y=x.view(-1, 3) ## We specified 3 columns and rest it will determine
print(y)
## Numpy array to tensor
import numpy as np
a=torch.ones(5)
print(a)
b=a.numpy()
print(type(b)) ## Numpy array
## If tensor on cpu and not on gpu and both objects will share the same memory
# changes in one will be reflected in other.
a.add_(10)
print(a)
print(b)
## Numpy to tensor
