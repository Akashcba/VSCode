# FADML Assignment
'''
@Author : Akash Choudhary
20BM6JP46
'''
## Sample Code Shared
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import model


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
## Load the DataSet

print('Data transformation')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

#Model
print('Model creation')

net = model.convfc()

net = net.to(device)

# Training the model........
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Write your code here
    train_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs).to(device)
        ## Loss Computation
        if mse:
            target_encoded = torch.nn.functional.one_hot(targets, 10).float()
            loss= criterion(outputs, target_encoded)
        else:
            loss= criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    training_loss_history.append(train_loss)

    #Write training losses to losstrain.txt for each epoch
    global training_loss
    training_loss[epoch] = train_loss

    print(f'Training Loss: {train_loss}')
    wandb.log({"Epoch":epoch," Training Loss": train_loss})

 
def test(epoch):
    global best_acc
    net.eval()
    test_images = []
    test_loss = 0.0
    correct = 0
    model_accuracy = 0.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            if mse:
                target_encoded = torch.nn.functional.one_hot(targets, 10).float()
                loss= criterion(outputs, target_encoded)
            else:
                loss= criterion(outputs, targets)

            #Predicting on the input data
            _, pred = torch.max(outputs.data, 1)
            ## Loss calculation
            test_loss += loss.item()
            correct += (pred == targets).float().sum()#.item()
            test_images.append(wandb.Image(inputs[0], caption=f"Pred: {pred[0].item()} Truth: {targets[0]}")

            ## Accurcay of the model
        model_accuracy = 100 * correct / output.shape[0]
        if model_accuracy > best_acc:
            #Saving the new value of best_accuracy
            best_acc = model_accuracy
            # Save checkpoint for the model which yields best accuracy
            PATH = f'./best_model{epoch}.pth'
            torch.save(net.state_dict(), PATH)
    # Write test set losses to losstest.txt for each epoch
        global test_losses
        test_losses[epoch] = test_loss
    #Write test set accuracies to acctest.txt for each epoch
        global test_accurcay
        test_accuracy[epoch] = model_accuracy
        # Printing the results
        print(f'Testing Loss: {test_loss}')
        print(f'Testing Accuracy: {model_accuracy}')
        wandb.log({"Epoch":epoch,"Test_images": test_images,"Test_Accuracy": 100. * correct / len(testloader.dataset),"Test_loss": test_loss})


def c_matrix_plot(net):
  all_crt = 0
  all_img = 0
  c_matrix = np.zeros([10,10], int)
  with torch.no_grad():
      for batch_id, (img, lbl) in enumerate(testloader):
          img, lbl = img.to(device), lbl.to(device)
          opt = net(img)
          _, pred = torch.max(opt.data, 1)
          all_img += lbl.size(0)
          all_crt += (pred == lbl).float().sum()#.item()
          for j, k in enumerate(lbl):
              c_matrix[k.item(), pred[j].item()] += 1 

  _acc = all_crt / all_img * 100
  print('Model accuracy on {all_img} , test images: {1:.2f _acc}')
  fig, ax = plt.subplots(1,1,figsize=(10,10))
  sns.heatmap(c_matrix,annot = True,fmt='d', cmap="mako")
  plt.ylabel('Actual Category')
  plt.yticks(range(10), classes)
  plt.xlabel('Predicted Category')
  plt.xticks(range(10), classes)
  plt.show()


for epoch in range(0,500):
    print("Training")
    train(epoch)
    print("Testing")
    test(epoch)

if __name__ == "__main__":

early_stopping_round = 10
  epoch_counter = 0
  for epoch in range(0,50):
    print("Model Training")
    train(epoch,net,criterion,optimizer,mse)
    print("\n Model Testing")
    test(epoch,net,criterion,mse)

    loss_df = pd.DataFrame.from_dict(training_loss)
    loss_df.to_csv(r'losstrain.txt', header=None, index=None, sep=' ', mode='a')
    data = pd.DataFrame.from_dict(df)
    data.to_csv(r'losstest.txt', header=None, index=None, sep=' ', mode='a')
    accuracy_df = pd.DataFrame.from_dict(testaccuracy)
    accuracy_df.to_csv(r'test_accuracy.txt', header=None, index=None, sep=' ', mode='a')

