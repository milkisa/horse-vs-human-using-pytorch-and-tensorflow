import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms
from torchvision import models
import PIL
import matplotlib.pyplot as plt

EPOCHS = 2
BATCH_SIZE = 10
LEARNING_RATE = 0.003
TRAIN_DATA_PATH = "D:/tolearn/datasets/horse-or-human"
VALIDATE_DATA= "D:/tolearn/datasets/validation-horse-or-human"
# TEST_DATA_PATH = "./images/test/"
TRANSFORM_IMG = transforms.Compose([  transforms.RandomHorizontalFlip(),transforms.RandomRotation(30),transforms.ToTensor()])
validate_IMG = transforms.Compose([transforms.ToTensor()])


train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
validate_data= torchvision.datasets.ImageFolder(root= VALIDATE_DATA, transform= validate_IMG)
validate_data_loader=data.DataLoader(validate_data, batch_size= BATCH_SIZE, shuffle= True, num_workers= 4)
print("Number of train samples: ", len(train_data))
# print("Number of test samples: ", len(test_data))
print("Detected Classes are: ", train_data.class_to_idx) # classes are detected by folder structure
model= models.resnet50(pretrained= True)
for param in model.parameters():
    param.requires_grad= False
from torch.autograd import Variable
device= ("cuda" if torch.cuda.is_available() else "cpu" )
model.fc = nn.Sequential(nn.Linear(2048,1024),nn.ReLU(), nn.Dropout(0.2),nn.Linear(1024,2))
criterion= nn.CrossEntropyLoss()
optimizer= optim.RMSprop(model.parameters(), lr= 0.0001)
model.to(device)
Epochs = 10
accuracy_total_train = []
train_loss = []
validate_loss = []
accuracy_total_validate = []
for epochs in range(Epochs):
    total_train_loss = 0
    total = 0
    total_validate_loss = 0
    for step, (image, label) in enumerate(train_data_loader):
        image, label = image.to(device), label.to(device)
        #         image= Variable(image)
        #         label= Variable(label)

        #         image= image.type(torch.FloatTensor)
        #         label= label.type(torch.FloatTensor)
        #         print(label)

        optimizer.zero_grad()
        pred = model.forward(image)
        #         print(pred)
        # #         pred= pred.view(-1)
        #         print(pred.shape)

        loss = criterion(pred, label)
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        #         _,yhat= torch.max(pred,1)
        #         print(yhat)
        _, yhat = torch.max(pred.data, 1)
        _, yhat = torch.max(pred.data, 1)
        total += (yhat == label).sum().item()
    accuracy = total / len(train_data)
    accuracy_total_train.append(accuracy)
    train_loss.append(total_train_loss)
    model.eval()
    with torch.no_grad():
        total_validate_loss = 0
        total = 0
        for step, (image, label) in enumerate(validate_data_loader):
            image, label = image.to(device), label.to(device)
            print(label.shape)
            #             image= image.type(torch.FloatTensor)
            #             label= label.type(torch.FloatTensor)

            pred = model.forward(image)
            #             print(pred.shape)

            loss = criterion(pred, label)
            total_validate_loss += loss.item()
            _, yhat = torch.max(pred, 1)
            #             print(yhat)
            total += (yhat == label).sum().item()

        test_accuracy = total / len(validate_data)
        accuracy_total_validate.append(test_accuracy)
        validate_loss.append(total_validate_loss)
    print("Epoch: {}/{}  ".format(epochs, Epochs),
          "Training loss: {:.4f}  ".format(total_train_loss),
          "Testing loss: {:.4f}  ".format(total_validate_loss),
          "Train accuracy: {:.4f}  ".format(accuracy),
          "Test accuracy: {:.4f}  ".format(test_accuracy))
    model.train()

