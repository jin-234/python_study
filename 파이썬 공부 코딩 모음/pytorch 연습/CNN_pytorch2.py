from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import torch,numpy as np
import torch.nn as nn

transform=transforms.ToTensor()

train_dataset=datasets.MNIST(
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
# print(train_dataset[0][0].shape)#torch.Size([1, 28, 28])

train_dl=DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)

test_dl=DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=True
)
# 모델 구조
# 1,28,28 conv2d 1,16
# relu
# 16,28,28 maxpool 2
# 16,14,14 conv2d 16,32
# relu
# 32,14,14 maxpool2
# 32,7,7 conv2d 32,64
# relu
# 64,7,7 flatten
# 64*7*7,256 linear
# relu
# dropout
# 256,128 linear
# relu
# dropout
# 128,64 liear
# relu
# dropout
# 64,10 linear
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cv1=nn.Conv2d(1,16,kernel_size=3,stride=1,padding='same')
        self.cv2=nn.Conv2d(16,32,kernel_size=3,stride=1,padding='same')
        self.cv3=nn.Conv2d(32,64,kernel_size=3,stride=1,padding='same')
        
        self.output1=nn.Linear(64*7*7,256)
        self.output2=nn.Linear(256,128)
        self.output3=nn.Linear(128,64)
        self.output4=nn.Linear(64,10)
        #과적합 방지
        self.dropout=nn.Dropout()
        self.relu=nn.ReLU()
        #이미지의 특정 범위중 최대값 뽑아냄(특징 추출)
        self.pool=nn.MaxPool2d(kernel_size=2)
        #이미지 데이터를 1차원으로 바꿈
        self.flatten=nn.Flatten()
    def forward(self,x):
        x=self.relu(self.cv1(x))
        x=self.pool(x)
        x=self.relu(self.cv2(x))
        x=self.pool(x)
        x=self.relu(self.cv3(x))
        x=self.flatten(x)
        x=self.relu(self.output1(x))
        x=self.dropout(x)
        x=self.relu(self.output2(x))
        x=self.dropout(x)
        x=self.relu(self.output3(x))
        x=self.dropout(x)
        x=self.relu(self.output4(x))
        
        
        return x
    
#모델 호출 GPU사용
model=CNN().cuda()
#다중 분류 문제이므로 crossentorpy 사용(softmax)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(params=model.parameters(),lr=0.0001)

epochs=10
for epoch in range(epochs):
    model.train()
    running_loss=0#loss값
    correct=0#맞은 개수
    total=0#총 개수
    for images,label in train_dl:
        images,label=images.cuda(),label.cuda()
        #순전파
        optimizer.zero_grad()
        #모델 예측값
        output=model(images)
        #손실값
        loss=criterion(output,label)
        #역전파
        loss.backward()
        #가중치 업데이트
        optimizer.step()
        #총 손실값 구하기 위해 더하기
        running_loss+=loss.item()
        #정확도 계산을 위한 예측값
        _, predicted = torch.max(output, 1)
        #맞은개수
        correct += (predicted == label).sum().item()
        #총 개수
        total += label.size(0)
    
    epoch_loss = running_loss / len(train_dl)
    epoch_acc = correct / total * 100
    #에폭마다 현제 진행도,손실값, 정확도 출력
    print(f'Epoch:{epoch+1},loss:{epoch_loss:.4f},Accuracy:{epoch_acc:.4f}')

model.eval()

correct=0
total=0

with torch.no_grad():
    for images,label in test_dl:
        images,label=images.cuda(),label.cuda()
        
        output=model(images)
        predict=output.argmax(dim=1)
        total+=label.size(0)
        correct+=(predict==label).sum().item()
    print(f'Accuracy:{correct/total}')
        