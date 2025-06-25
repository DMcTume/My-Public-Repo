# Data from: https://www.kaggle.com/datasets/dmcgow/birds-200
# NOTE: THIS IS BY NO MEANS ELEGANT CODE (WILL FIX UP LATER...MAYBE)

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# For neural network
import torch.nn as nn 
import torch.nn.functional as F

# For training
import torch.optim as optim

# Step 1: Data Preparation

means, sds = "", ""     # Grab the means and stds for standardization of the training tensors
with open("image_stats.txt", "r") as file:
    means = file.readline()
    sds = file.readline()

means = means.strip().strip("[").strip("]").split(" ")
sds = sds.strip("[").strip("]").split(" ")

mean_tuple = (float(means[0]), float(means[1]), float(means[2]))
std_tuple = (float(sds[0]), float(sds[1]), float(sds[2]))


image_side_length = 256 # for resolution and model definition

transform = transforms.Compose(
    [transforms.Resize((128, 128)), # arbitrary; did not calculate image resolution for this
     transforms.ToTensor(),
     transforms.Normalize(mean_tuple, std_tuple)]
)

train_imgs = ImageFolder(root = r"bird_data\birds_train", transform = transform)
test_imgs = ImageFolder(root = r"bird_data\birds_test", transform = transform)    # NOTE: ImageFolder only works with subdirectories (so I made one)

batch_size = 64
trainloader = DataLoader(train_imgs, batch_size = batch_size, shuffle = True)

testloader = DataLoader(test_imgs, batch_size = batch_size, shuffle = False)

# Step 2: Model Configuration and Loading

class BirdNet(nn.Module):

    # FORMULA FOR OUPUT OF A CONV HIDDEN LAYER
    # (W - F + 2 * P)/S + 1
    # W = input size, F = filter size, P = padding, S = stride

    def __init__(self): # DEFINES the neural network's structure
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 3 color channels, ouput channel size = 6, kernel is 5x5
        self.pool = nn.MaxPool2d(2, 2) # kernel size of 2 (2x2), stride of 2 (moves 2 pixels to the right each time)
        self.conv2 = nn.Conv2d(6, 16, 5) # accepts output from conv1, then ouputs 16
        
        # linear layers (classification part)
        
        
        self.fc1 = nn.Linear(16 * 29 * 29, 300) # was 16 * 5 * 5 (find out how to figure out dimensions for this stuff)
        self.fc2 = nn.Linear(300, 250)
        self.fc3 = nn.Linear(250, 200)
        
        # avg = (16 * 29 * 29 + 200)/2
        # self.fc1 = nn.Linear(16*29*29, int(avg))
        # self.fc2 = nn.Linear(int(avg), 200)

    def forward(self, x): # PASSES in data using the previously defined structure
        
        # Convulution (with activation functions)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten feature map and pass it through the linear layers
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x)) # Has activation function
        x = F.relu(self.fc2(x)) # Has activation function
        x = self.fc3(x)         # NO activation function

        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        return x

# device = "xpu" if torch.xpu.is_available() else "cpu"
device = "xpu"
print(f"Using {device}...")

model_path = "saved_checkpoint.pth"
model = 0
loss_func = nn.CrossEntropyLoss()
optimizer = 0

learning_rate = 0.001
momentum = 0.9

epochs_ran = 0
losses = []

def train_model():

    num_epochs = int(input("How many epochs to train for? "))
    print("\nStarting training:\n")

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0): # i corresponds to index of the mini batch ("data")
            inputs, labels = data

            inputs.to(device)
            labels.to(device)
            optimizer.zero_grad()

            predictions = model(inputs)
            batch_loss = loss_func(predictions, labels) # IndexError is because the model does not have a proper output of 200 (corresponds to each class)
            
            batch_loss.backward() # computes gradients
            optimizer.step()      # updates parameters with the computed gradients

            running_loss += batch_loss.item()
            if i % 50 == 0:
                print(f"Epoch {epoch + 1}, mini-batch #{i}: {running_loss}")
                running_loss = 0.0

        epoch_loss = running_loss / len(trainloader)
        losses.append(epoch_loss)

    print("Epochs ran before saving: " + str(epochs_ran))
    new_checkpoint = {
        "epoch": epochs_ran + num_epochs, 
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": losses,     
    }

    print("Saving trained model...\n")
    torch.save(new_checkpoint, model_path)
    print("Checkpoint saved, ending program...\n")

def test_model(mode):
    model.eval()
    
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            input, labels = data
            
            predictions = model(input)

            # Keeps the indicies of the predicitions that the model got right:
            _, predictions = torch.max(predictions, 1) 
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    print(f"Model Accuracy: {100 * correct // total}%")

# Most of Configuration + Training, Saving, Loading

def load_model():
    model = BirdNet()
    checkpoint = torch.load("saved_checkpoint.pth")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = momentum)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    epochs_ran = checkpoint["epoch"]
    losses = checkpoint["loss"]

    return model, optimizer, epochs_ran, losses

while(True):  # save/load options (user-determined)
    
    model_choice = input("Are we overwriting or updating a model?\n" + 
                        "Type O for overwrite/to save a new model;\n" +  
                         "Type U for update;\n" +
                         "Type S to just view the current checkpoint configuration;\n" +
                         "Type V to run validation;\n" 
                         "Type Q to quit:\n")
    
    if (model_choice == "O"):
        print("Chose to overwrite/create new...no loading need be done...\n")
        model = BirdNet().to(device)
        optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = momentum)

        train_model()

    elif (model_choice == "U"):
        model, optimizer, epochs_ran, losses = load_model()
        train_model()
    
    elif (model_choice == "S"):
        
        checkpoint = torch.load("saved_checkpoint.pth")

        print(f"\nEpochs ran: {checkpoint["epoch"]}")
        
        print(f"Average loss: {sum(checkpoint["loss"]) / checkpoint["epoch"]}")

        # print(f"Average val loss: {sum(checkpoint["val_loss"]) / len(testloader)}\n")

        # print(f"Model Status:\n {checkpoint["model_state_dict"]}")

        # print(f"Optimizer Status:\n {checkpoint["optimizer_state_dict"]}")
    
    elif (model_choice == "V"):
        
        if (model == 0):
            model, optimizer, epochs_ran, losses = load_model()
            print("Loaded model")

        print("Running validation...")
        test_model("this works")

    elif (model_choice == "Q"):
        print("Ending program....")
        break

    else:
        print("Invalid option\n")


