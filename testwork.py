# This is a test model for images from one of Pytorch's built-in datasets 
# It's from TorchVision: holds image datasets, models, transformers, and utilties

import torch 
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets 
from torchvision.transforms.transforms import ToTensor
import torchvision.transforms 
import torchvision.transforms.transforms

# Training data 
training_data = datasets.FashionMNIST(      # Load training data from torchvision's datasets (called FashionMNIST)
    root = "data",                          # Root directory where the dataset exists
    train = True,
    download = True,                        # Downloads dataset from internet and puts it in the root directory (no installation required) 
    transform = ToTensor()                    # Changes features from PIL image format to tensors (features = data you're using to predict outcome)
)

# Testing data
testing_data = datasets.FashionMNIST(
    root = "data", 
    train = False,                                  # This is testing data, so it's not marked as training data (training = False)
    download = True,
    transform = ToTensor()
)

# Feed datasets into dataloaders to be processed

batch_size = 64                              # amount of samples processed by model; includes features paired with their respective labels

# Data Loaders
train_dataloader = DataLoader(training_data, batch_size = batch_size, shuffle = True)
test_dataloader = DataLoader(testing_data, batch_size, shuffle = True)


# Before was just prepping the data; next is defining the neural network


# Get cpu, gpu, or mps device to run neural network on 
device = (
    "cuda"      # CUDA tensor type: uses GPUs for computation 

    if torch.cuda.is_available()    # if system doesn't support CUDA for the GPU, use the MPS file
    else "mps"

    if torch.backends.mps.is_available()    # if system doesn't support MPS, use the CPU
    else "cpu"
)

# Defining the model

class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()              # inherents initialization from nn.Module 
        self.flatten = nn.Flatten()     # "flattening" process done in neural network (explained in notes

        self.linear_relu_stack = nn.Sequential(     # Creates the layers of the network 
            nn.Linear(28*28, 512),                  # Linear layer with 784 inputs and 512 outputs; linear transformation 
            nn.ReLU(),                                # Type of activation function: "Rectified Linear Unit"; turns any negative values to 0; positive values remain unchanged
            nn.Linear(512, 512),
            nn.ReLU(), 
            nn.Linear(512, 10) 
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return(logits)
    
model = NeuralNetwork().to(device)

# Optimize Model parameters and train model (needs to be explained later)

loss_fn = nn.CrossEntropyLoss()                             # loss function for classification problems: calculates loss with loss.backward()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)  # optimizer: adjusts weights/biases to minimize loss with optimize.step()


def train(dataloader, model, loss_fn, optimizer):   # function to train the model: makes predictions + backpropagation

    size = len(dataloader.dataset)      # returns number of data points
    model.train()                       # sets model into training mode
    
    for batch, (X,y) in enumerate(dataloader):  # for each pair of training/testing data point, move them to the device for computation
        X, y = X.to(device), y.to(device)

    # Compute prediction error:
    prediction = model(X)               # pass training data through NN
    loss = loss_fn(prediction, y)       # calculate the loss by comparing NN results to the testing data

    # Backpropagation
    loss.backward()                     # computes gradients in respect to loss (math stuff)
    optimizer.step()                    # uses loss.backward() to optimize weights/biases
    optimizer.zero_grad()               # reset the gradients to zero so changes don't accumulate over epochs

    # Print specific info every 100th batch
    if batch % 100 == 0:
        loss, current = loss.item(), (batch +1 ) * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



def test(dataloader, model, loss_fn):   # compare model's performance to test data to make sure it's learning
    
    size = len(dataloader.dataset)  # returns number of data points

    num_batches = len(dataloader)   # returns number of batches

    model.eval()                    # sets model into testing mode

    test_loss, correct = 0, 0

    with torch.no_grad():   # disables gradient calculation for the following code; re-enabled afterwards

            # iterator = iter(dataloader)
            # X, y = next(iterator)
        for X, y in dataloader:

            X, y = X.to(device), y.to(device)                                   # brings training/testing datapoints to device
            pred = model(X)                                                     # passes training data through model
            test_loss += loss_fn(pred, y).item()                                # loss of NN results with gradient calculation disabled
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()     # compute accuracy of the results

    test_loss /= num_batches            # update test_loss by dividing it by the number of batches
    correct /= size                     # do the same for correct with size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")     # String format stuff

# Training process: iterate model by epochs

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n--------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
    print("Done!")
