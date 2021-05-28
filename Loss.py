## pytorch automated loss calculation and
## optimizers
''' Pipeline Process
Design the model (input_size, output_size, forward_pass)
Construct Loss and Optimizer
Training Loop
- Forward pass => Prediction
- backward pass => Gradients
- update weights
'''
import torch
import torch.nn as nn

x = torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)
y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

num_samples ,n_features = x.shape
input_size=n_features
output_size=n_features
model = nn.Linear(input_size, output_size)  ## Single Layer

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(20):
    y_pred=model(x)
    l=loss(y,y_pred)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()

print(model(torch.tensor([5], dtype=torch.float32)).item())

## Custom model
'''
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        ## Define Layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)
'''