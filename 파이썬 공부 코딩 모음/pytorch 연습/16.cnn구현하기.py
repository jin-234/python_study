# 구현하는 CNN 구조
# (1,28,28)
# Conv2d(1→16)
# ReLU
# MaxPool2d
# Conv2d(16→32)
# ReLU
# MaxPool2d
# Flatten
# Linear
# 출력(10)
import torch
import torch.nn as nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

transform=transforms.ToTensor()
train_datasets=datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)
test_dataset=datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)
train_loader=DataLoader(train_datasets,batch_size=64,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=64,shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,16,3,padding=1)
        self.conv2=nn.Conv2d(16,32,3,padding=1)
        self.pool=nn.MaxPool2d(2)
        self.relu=nn.ReLU()
        self.flatten=nn.Flatten()
        self.fc=nn.Linear(32*7*7,10)
    def forward(self,x):
        x=self.relu(self.conv1(x))
        x=self.pool(x)
        x=self.relu(self.conv2(x))
        x=self.pool(x)
        x=self.flatten(x)
        x=self.fc(x)
        
        return x

model=CNN().cuda()
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.0001)

epochs=10
for epoch in range(epochs):
    model.train()
    running_loss=0
    for images,labels in train_loader:
        images,labels=images.cuda(),labels.cuda()
        
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    epoch_loss=running_loss/len(train_loader)    
    print(f'Epoch: {epoch},Loss: {epoch_loss:.4f}')    
model.eval()
correct=0
total=0

with torch.no_grad():
    for images ,labels in test_loader:
        
        images,labels=images.cuda(),labels.cuda()
        
        outputs=model(images)
        loss=criterion(outputs,labels)
        
        predict=outputs.argmax(dim=1)
        
        total+=labels.size(0)
        correct+=(predict==labels).sum().item()
    print(f'Accuracy: {correct/total}')