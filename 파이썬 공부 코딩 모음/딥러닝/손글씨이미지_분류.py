from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt



mnist=datasets.load_digits()
feature=mnist['data']
print(len(feature[0]))
print(len(feature))
lebels=mnist['target']
print(len(lebels))
print(np.unique(lebels,return_counts=True))

# print(feature[0].reshape(8,8))
# plt.imshow(feature[8].reshape(8,8),cmap='gray')
# plt.savefig('mnist_0.jpeg')

print(feature.shape)

#다중분류형으로 출력할것이므로 형태 맞춰주기
# from keras.utils import to_categorical
# lebels=to_categorical(lebels)
#==>(1797,8,8,1)=>몇개씩 묶을거냐 ,(batch size,이미지가로,이미지 세로,채널) (깊이를 채널이라고함)
feature=feature.reshape(-1,8,8,1)/255.0 #사이즈 변경+스케일 정규화

# print(feature[0])

#features와 labels을 train_x,val_x,로 분함 분할 비율은 0.2

from sklearn.model_selection import train_test_split

train_x,val_x,train_y,val_y=train_test_split(feature,lebels,test_size=0.2,random_state=42)

# print(len(train_x))
# print(len(val_x))

#데이터 전처리 및 데이터 준비 완료

#모델 준비=>이미지 분류하는 모델 설계(10개 이미지 분류,다중분류)
#이미지 분류에 특화된 CNN모델 사용

#개의 Conv ,2개의 pooling,
#Flatten, dropout, FC layer 층추가
#마지막 출력층은 10개의 뉴런으로 설정 

#손실함수는 categorical_crossentrophy ,softmax

#모델 설계 이후 모델 학습

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.layers import Conv2D,MaxPooling2D#1D는 1차원 데이터,2D는  이미지,3D는 비디오

model=Sequential()
#모듈의 첫번째 입력층이므로 입력 데이터의 형태 명시
model.add(Conv2D(filters=32,padding='same',kernel_size=3,activation='leaky_relu',input_shape=(8,8,1)))
model.add(MaxPooling2D(pool_size=2))
#필터는 뒤로갈수록 수가 많아져야 더 복잡한 데이터를 학습 시키기 좋음
model.add(Conv2D(filters=64,padding='same',kernel_size=3,activation='leaky_relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())

model.add(Dense(units=100,activation='leaky_relu'))
model.add(Dropout(0.3))#오차역전파 과정에서 이 비율 만큼의 뉴런을 종료해서 의존도를 낮춰줌(과적합 방지)

model.add(Dense(units=60,activation='leaky_relu'))
model.add(Dense(units=10,activation='softmax'))#출력층

model.summary()

#콜백 생성
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

stop=EarlyStopping(restore_best_weights=True,verbose=1,patience=3)#조기 종료
#가장 좋은 데이터를 저장
check=ModelCheckpoint(save_best_only=True,filepath='/home/jin/deeplearning_prj/20260615/손글씨이미지_분류.keras')

#target(정답 )==>정수형태로 사용==>sparse_categorical_crossentropy
# categorical_crossentropy=>정답을 원핫 인코딩 상태로 변환해서 전달
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train_x,train_y,validation_data=(val_x,val_y), verbose=1,batch_size=4,epochs=100,callbacks=[stop,check])