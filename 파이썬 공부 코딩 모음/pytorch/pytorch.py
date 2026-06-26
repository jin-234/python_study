# iris 데이터 사용 mlp 분류 기초

# 다중 퍼셉트론 기반 분류 모델 구현 
# 신경망의 레이어 구조와 차원 계산
# 다양한 활성화 함수의 역할과 비교
# 데이터 로더와 배치 처리
# 학습의 3단계 구현(순전파 손실계산,역전파 구현)

#1.라이브러리 임포트및 환결설정

import torch,numpy as np,pandas as pd , matplotlib.pyplot as plt
from torch import nn , optim
from torch.utils.data import TensorDataset,DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#재현성을 위해 시드고정(randomstate=42)
np.random.seed(42)
torch.manual_seed(42)

#2. iris데이터셋 로드 및 탐색
iris=load_iris()
x=iris.data
y=iris.target

print(f'데이터 형태:{x.shape}')
print(f'특성 형태:{x.shape[1]}')
print(f'클래스 개수:{len(np.unique(y))}')
print(f'클래스 레이블:{iris.target_names}')

df=pd.DataFrame(x,columns=iris.feature_names)
df['target']=y
#데이터 기술 통계 기본
df.describe()

# 데이터 전처리

#신경망 학습을 위해 데이터 표준화, Pytorch Tensor로 변환

#표준화(평균0,표준편차1)
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
x_tensor=torch.FloatTensor(x_scaled)
y_tensor=torch.LongTensor(y)

#데이터 분할
#Random split을 사용하여 학습 데이터와 테스트 데이터를 분리합니다
#훈련세트와 테스트세트 분할  80%학습 20%테스트
x_train,x_test,y_train,y_test=train_test_split(x_tensor,y_tensor,stratify=y,random_state=42,train_size=0.8)
print(f'훈련 데이터 형태:{x_train.shape}')
print(f'테스트 데이터 형태:{x_test.shape}')

#데이터 로더 생성
###배치(batch)개념(batch_size인듯?)
#전체 데이터를 한 번에 학습하지 않고 작은 묶음(배치)로 나누어 학습
# 메모리 효율 성과 학습 안정성 향상
# DataLoader가 자동으로 배치 분할 처리

# TensorDataset생성
train_dataset=TensorDataset(x_train,y_train)
test_dataset=TensorDataset(x_test,y_test)

# DataLoader생성

batch_size=16
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=batch_size)

print(f'배치크기:{batch_size}')
print(f'훈련 데이터 배치 개수:{len(train_loader)}')
print(f'태스트 데이터 배치 개수:{len(test_loader)}')

#배치 하나의 내용 확인

for data,target in train_loader:
    print(f'배치 x 형태:{data.shape}')
    print(f'배치y형태:{target.shape}')
    break

#6신경망 모델 구조 설계

# 차원 계산 방법
# 각 레이어를 통과할때마다 데이터의 차원이 변환됩니다

# 입력층(4)->은닉1층(16) ->은닉2층(8)->출력층(3)

# -입력차원: 특성개수=4
# 은닉 차원 설계자가 결정(일반적으로 입력보다 크거나 같게)

# -출력차원 클래스 개수=3


#모델 구조 파라미터
input_dim=x_train.shape[1]#4
hidden_dim1=16
hidden_dim2=8
output_dim=len(np.unique(y))#3
print(f'입력차원 :{input_dim}')
print(f'은닉1차원 :{hidden_dim1}')
print(f'은닉2차원 :{hidden_dim2}')
print(f'출력차원 :{output_dim}')

##레이어별 차원 변화 계산

# Linear 레이어의 출력 차원= 레이어에 설정된 출력 뉴런 개수

# nn.Linear(in_features,out_features)
# 입력:(batch_size,in_features)
# 출력:(batch_size,out_features)

#차원 변화 시뮬레이션

sample_input=torch.randn(batch_size,input_dim)
print(f'입력차원:{sample_input.shape}')
layer1=nn.Linear(input_dim,hidden_dim1)
output1=layer1(sample_input)
print(f'레이어1 통과후 차원:{output1.shape}')

layer2=nn.Linear(hidden_dim1,hidden_dim2)
output2=layer2(output1)
print(f'레이어2 통과후 차원{output2.shape}')

layer3=nn.Linear(hidden_dim2,output_dim)
output3=layer3(output2)
print(f'레이어3 통과후 차원:{output3.shape}')


# 7.MLP모델 정ㅇ의(Sequential API)

# 활성화 함수의 역할
# -선형 변환만으로는 복잡한 패턴을(ex,XOR문제) 학습할 수 없음
# -비선형성(non-linearity)을 추가하여 신경망의 표현력 향상시킴
# -각 은닉층 뒤에 활성화 함수 적용

# Sequential API를 사용한 모델 정의
model=nn.Sequential(
    nn.Linear(input_dim,hidden_dim1),
    nn.ReLU(),
    nn.Linear(hidden_dim1,hidden_dim2),
    nn.ReLU(),
    nn.Linear(hidden_dim2,output_dim),
    
)

print(model)

# 활성화 함수 비교

# 1.sigmoid:
#     출력 범위: 0-1사이
#     기울기 소실 문제 발생
# 2.Than:
    # 출력 범위:-1-1사이 
    # zero center(0을 기준으로 좌우대칭)여서 sigmoid보다는 기울기 소실이 덜함

# 3.Relu:
#     출력범위 양수면 숫자 그대로, 0이하면 0으로(dying relu 문제 발생 가능)
#     기울기 소실이 적음
#     가장 널리 사용됨
#     계산 간단, 학습속도 빠름

# 4.leaky_relu:
#     f(x)가 0이하는 매우 적은 기울기로, 양수일때는 그대로 반환
#     dying relu 문제 완화(음수가0)
    #   음수 입력에 대해 작은 기울기를 유지  
    
#8손실 함수 및 옵티마이저 설정

#crossentropyloss를 사용하는 이유
    # 다중 클래스 분류 문제 적합
    # softmax+NLLLoss를 결합한 형태
    # 예측 확률과 실제 레이블간의 차이 측정
    
# 옵티마이저-SGD(stochastic gradiant descent)
# 가장 기본적인 옵티마이저
# learning rate 학습 속도를 조절하는 하이퍼 파라미터

#손실함수
critrion=nn.CrossEntropyLoss()

#옵티마이저
learning_rate=0.01
optimizer=optim.SGD(model.parameters(),lr=learning_rate)

print(f'손실함수:{critrion}')
print(f'옵티마이저:{optimizer}')
print(f'Learning rate:{learning_rate}')

# 9학습의 3단계

# 신경망 학습은 다음 3단계 반복:
    
# 1순전파(Feed Forward Propagation)
# -입력 데이터를 모델에 통과시켜 예측값 계산
# -output=model(input)

# 2손싫 계산(Loss Calculation)
# -예측값과 실제값의 차이를 손실함수로 계산
# -loss=critrion(output,target)

# 3역전파(backpropagation)
# -손실을 기준으로 가중치 업데이트
# -loss.backward():그래디언트 계산
# -optimizer.step():가중치 업데이트

# 한개의 배치에 대한 3단계 예시
# model 을 학습 전용으로 변경을 하기위해 .train()호출
model.train()

#한 배치 가져오기
for batch_x,batch_y in train_loader:
    #1번째 단계:순전파
    outputs=model(batch_x)
    print(f'순전파 출력 형태:{outputs.shape}')
    
    #2단계 손실계산
    loss=critrion(outputs,batch_y)
    print(f'손실값:{loss.item():.4f}')
    
    # 3단계 역전파
    optimizer.zero_grad()#그래디언트 초기화
    loss.backward()#그래디언트 계산
    optimizer.step()#가중치 업데이트
    break

#10 학습 함수 작성
# 전체 데이터셋에 대해 여러 epoch 동안 학습을 반복하는 함수 작성

def train_model(model,train_loader,criterion,optimizer,num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss=0.0
        for batch_x,batch_y in train_loader:
                #1번째 단계:순전파
            outputs=model(batch_x)
        
            #2단계 손실계산
            loss=critrion(outputs,batch_y)
        
            # 3단계 역전파
            optimizer.zero_grad()#그래디언트 초기화
            loss.backward()#그래디언트 계산
            optimizer.step()#가중치 업데이트
            running_loss+=loss.item()
        epoch_loss=running_loss/len(train_loader)
        print(f'epoch[{epoch+1}/{num_epochs}],Loss:{epoch_loss:.4f}')
        
