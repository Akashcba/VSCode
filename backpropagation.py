# x->a(x) = y ->b(y) = z
# by chain rule :
# dz/dx = dz/dy * dy/dx
## Pytorch creates a computational graph on their own
## We compute the d(Loss)/dx for updating the weights
'''
Forward Pass
Compute Loss
BackPropagate : Compute dLoss/dWeights using chain rule
'''
import torch

x=torch.tensor(1.0)
y=torch.tensor(2.0)

w=torch.tensor(1.0, requires_grad=True)
y_hat = w*x
loss=(y_hat-y)**2
## Backward Pass
loss.backward()
print(w.grad)
## Update weights
with torch.no_grad():
    w=w-0.001*w.grad.data + 0 ## 0.001 is the learning rate
w.requires_grad_(True)
print(w)
## Next Forward pass
y_hat = w*x
loss=(y_hat-y)**2
## Backward Pass
loss.backward()
print(w.grad)
## Update weights
with torch.no_grad():
    w=w-0.001*w.grad.data + 0 ## 0.001 is the learning rate
w.requires_grad_(True)
print(w)