import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torchvision import transforms

from torch.utils.data import DataLoader

transform=transforms.ToTensor()

train_dataset=datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset=datasets.MNIST(
    root='./data',  
    train=False,
    download=True,
    transform=transform
)

# print(len(train_dataset))
# print(train_dataset[0])

train_loader=DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader=DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(784,256)

        self.fc2=nn.Linear(256,128)
        self.fc3=nn.Linear(128,10)
        self.relu=nn.ReLU()
        
    def forward(self,x):
        x=x.view(-1,784)
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        
        return x
    
model= MyNet().cuda()

criterion=nn.CrossEntropyLoss()

optimizer=optim.Adam(model.parameters(),lr=0.001)
epochs=5
for epoch in range(epochs):
    for images,labels in train_loader:
        images=images.cuda()
        labels=labels.cuda()
        
        outputs=model(images)
        
        loss=criterion(outputs,labels)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
    print(f'Epoch {epoch+1}, Loss={loss.item():.4f}')

correct=0
total=0

model.eval()

with torch.no_grad():
    for images,labels in test_loader:
        images=images.cuda()
        labels=labels.cuda()
        
        outputs=model(images)
        
        _,predict=torch.max(outputs,1)
        
        total+=labels.size(0)
        
        correct+=(predict==labels).sum().item()
        
accuracy=100*correct/total

print(f'Accuracy:{accuracy:.2f}%')

