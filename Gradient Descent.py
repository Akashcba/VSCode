'''
autograd package of torch allows us to compute gradients automatically
'''
## Here we will do everything from scratch
'''
import numpy as np

x=np.array([1,2,3,4], dtype=np.float32)
y=np.array([2,4,6,8], dtype=np.float32)

w=0.0

## Model Prediction
def forward(x):
    return w*x

## Loss
def loss(y, y_pred):
    return ((y_pred-y)**2).mean()

## gradient
# MSE = 1/N * (w*x - y)**2
#  dLoss/dw = 1/N * 2x(w*x - y)
def gradient(x,y,y_pred):
    return np.dot(2*x, y_pred - y).mean()

print(f'Prediction before training: f(5) = {forward(5)}')

## Training
learning_rate = 0.01
n_iters = 15
for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(x)
    # loss
    l=loss(y,y_pred)
    #gradients
    dw = gradient(x,y,y_pred)
    # update
    w = w-learning_rate*dw
    print(f"epoch{epoch+1}: w = {w:.3f}, loss:{l:.8f}")

print(f"Prediction after training: f(5) = {forward(5):.3f}")
'''
## Pytorch Implementation
import torch

x = torch.tensor([1,2,3,4], dtype=torch.float32)
y = torch.tensor([2,4,6,8], dtype=torch.float32)

w=torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


## Model Prediction
def forward(x):
    return w*x

## Loss
def loss(y, y_pred):
    return ((y_pred-y)**2).mean()
y_pred = forward(x)
l=loss(y,y_pred)
learning_rate = 0.01
## gradient
l.backward() # dLoss/dw

#update weights
with torch.no_grad():
    w-=learning_rate*w.grad
w.grad.zero_()
print(f'Prediction before training: f(5) = {forward(5)}')
