import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm

transform = transforms.Compose(
    [transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()]
)

dataset = datasets.ImageFolder("Dataset", transform=transform)

indices = list(range(len(dataset)))

split = int(np.floor(0.85 * len(dataset)))  # train_size

validation = int(np.floor(0.70 * split))  # validation

print(0, validation, split, len(dataset))

print(f"length of train size :{validation}")
print(f"length of validation size :{split - validation}")
print(f"length of test size :{len(dataset)-validation}")

np.random.shuffle(indices)

train_indices, validation_indices, test_indices = (
    indices[:validation],
    indices[validation:split],
    indices[split:],
)

train_sampler = SubsetRandomSampler(train_indices)
validation_sampler = SubsetRandomSampler(validation_indices)
test_sampler = SubsetRandomSampler(test_indices)

targets_size = len(dataset.class_to_idx)


class CNN(nn.Module):
    def __init__(self, K):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            # conv2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            # conv3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            # conv4
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )

        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(50176, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, K),
        )

    def forward(self, X):
        out = self.conv_layers(X)

        # Flatten
        out = out.view(-1, 50176)

        # Fully connected
        out = self.dense_layers(out)

        return out
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = CNN(targets_size)
model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

def batch_gd(model, criterion, train_loader, test_loader, epochs, validation_loader):
    train_losses = np.zeros(epochs)
    validation_losses = np.zeros(epochs)
    print("Initialized train_losses and validation_losses")

    for e in range(epochs):
        t0 = datetime.now()
        print(f"Start of epoch {e+1}, time: {t0}")
        train_loss = []
        print("Initialized train_loss list")

        for inputs, targets in tqdm(train_loader, desc=f"Training Epoch {e+1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            print("Moved inputs and targets to device")

            optimizer.zero_grad()
            print("Cleared gradients")

            output = model(inputs)
            print("Computed model output")

            loss = criterion(output, targets)
            print(f"Computed loss: {loss.item()}")

            train_loss.append(loss.item())
            print("Appended loss to train_loss list")

            loss.backward()
            print("Performed backpropagation")

            optimizer.step()
            print("Updated model parameters")

        train_loss = np.mean(train_loss)
        print(f"Computed mean train_loss for epoch {e+1}: {train_loss}")

        validation_loss = []
        print("Initialized validation_loss list")

        for inputs, targets in tqdm(validation_loader, desc=f"Validation Epoch {e+1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            print("Moved inputs and targets to device")

            output = model(inputs)
            print("Computed model output")

            loss = criterion(output, targets)
            print(f"Computed loss: {loss.item()}")

            validation_loss.append(loss.item())
            print("Appended loss to validation_loss list")

        validation_loss = np.mean(validation_loss)
        print(f"Computed mean validation_loss for epoch {e+1}: {validation_loss}")

        train_losses[e] = train_loss
        validation_losses[e] = validation_loss
        print(f"Stored train_loss and validation_loss for epoch {e+1}")

        dt = datetime.now() - t0
        print(f"Epoch {e+1} duration: {dt}")

        print(
            f"Epoch : {e+1}/{epochs} Train_loss:{train_loss:.3f} Test_loss:{validation_loss:.3f} Duration:{dt}"
        )

    return train_losses, validation_losses



batch_size = 64
train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, sampler=train_sampler
)
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, sampler=test_sampler
)
validation_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, sampler=validation_sampler
)

train_losses, validation_losses = batch_gd(
    model=model, 
    criterion=criterion, 
    train_loader=train_loader, 
    validation_loader=validation_loader, 
    test_loader=test_loader,
    epochs=5
)

torch.save(model.state_dict() , 'plant_disease_model_1.pt')


targets_size = 39
model = CNN(targets_size)
model.load_state_dict(torch.load("plant_disease_model_1.pt"))
model.to(device)
model.eval()

def accuracy(loader):
    n_correct = 0
    n_total = 0

    for inputs, targets in tqdm(loader, desc="Evaluating"):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)

        _, predictions = torch.max(outputs, 1)

        n_correct += (predictions == targets).sum().item()
        n_total += targets.shape[0]

    acc = n_correct / n_total
    return acc

train_acc = accuracy(train_loader)
test_acc = accuracy(test_loader)
validation_acc = accuracy(validation_loader)

print(
    f"Train Accuracy : {train_acc:.3f}\nTest Accuracy : {test_acc:.3f}\nValidation Accuracy : {validation_acc:.3f}"
)
