import torch
import torch.nn as nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

#텐서 형태로 변환
transform=transforms.ToTensor()
#데이터셋 제작
train_dataset=datasets.FashionMNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)
test_dataset=datasets.FashionMNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)   

# print(train_dataset[0][0].shape)#torch.Size([1, 28, 28])
# print(torch.unique(train_dataset.targets))#tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
#데이터 로더 만들기
train_dl=DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
    
)

test_dl=DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
    
)

#모델 설계
# 1,28,28 conv2d
# 32,28,28 pool
# 32,14,14 coon2d
# 64,14,14 pool
# 64,7,7 con2d
# 128,7,7 flatten
# 128*7*7->10 linear
class CNN_MODEL(nn.Module):
    def __init__(self):
        super().__init__()
        self.cv1=nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1)
        self.cv2=nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
        self.cv3=nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2)
        self.fc2=nn.Linear(128*7*7,10)
        self.flatten=nn.Flatten()
        self.relu=nn.ReLU()
    def forward(self,x):
        x=self.relu(self.cv1(x))
        x=self.pool(x)
        x=self.relu(self.cv2(x))
        x=self.pool(x)
        x=self.relu(self.cv3(x))
        x=self.flatten(x)
        x=self.fc2(x)
        
        return x

#모델 호출(GPU사용)
model=CNN_MODEL().cuda()
#softmax 다중분류이므로 Crossentoripyloss사용
criterion=nn.CrossEntropyLoss()
# optimizer adam사용
optimizer=torch.optim.Adam(params=model.parameters(),lr=0.0001)

#학습 10번
epochs=50
#모델 학습
for epoch in range(epochs):
    model.train()
    total_loss=0# 손실값
    total=0#총개수
    correct=0#맞은개수
    
    for images,label in train_dl:
        
        images,label=images.cuda(),label.cuda()
        
        #기울기 초기화
        optimizer.zero_grad()
        
        #모델 예측값
        output=model(images)
        #모델 손실값
        loss=criterion(output,label)
        #역전파 시행
        loss.backward()
        # 가중치 업데이트
        optimizer.step()
        
        total_loss+=loss.item()
        #모델 예측값 뽑아냄
        _,predict=torch.max(output,1)
        #얘측 성공시 추가
        correct += (predict == label).sum().item()
        #데이터의 전체 길이를 추출
        total += label.size(0)
    #맞은 개수/전체 개수 로 정확도 구함
    total_acc = correct / total * 100

    #전체 손실값 계산
    total_loss=total_loss/len(train_dl)
    
    print(f'Epochs:{epoch+1},Loss:{total_loss},Accuracy:{total_acc}')
    

with torch.no_grad():
    
    for images,label in test_dl:
        images,label=images.cuda(),label.cuda()
        output=model(images)
        predict=output.argmax(dim=1)
        total+=label.size(0)
        correct+=(predict==label).sum().item()
    print(f'Accuracy:{correct/total}')
        