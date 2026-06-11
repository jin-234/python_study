import numpy as np
import pandas as pd

np.set_printoptions(precision=8,suppress=True)
np.set_printoptions(threshold=np.inf)

#e지수 표현하는 과학적 표기 대신 소수점이하 8자리까지 표현
np.set_printoptions(precision=8,suppress=True)
fishdf=pd.read_csv('/home/jin/deeplearning_prj/20260611/fish_data.csv')
print(fishdf['Species'].unique())#물고기의 종이 몇종(총7종)
#7개 물고기중 어떤 물고기야
fishdf.info()

#serires를 넘파이 배열로 변환==>to_numpy()
fish_target=fishdf['Species'].to_numpy()
print(fish_target)
print(fishdf.columns)
fish_train=fishdf[[ 'Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
print(fish_train)

#라벨 문자열을 수치 형태로 변환 ===>Labelencoder

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

le=LabelEncoder()
y_encode=le.fit_transform(fish_target)
print('ㅇ!!!',le.classes_)
print(y_encode)# 타켓이 수치형태로 변환됨
#categorical_crossentropy()==>정답이 원-핫인코딩 상태이어야함

from tensorflow.keras.utils import to_categorical
y_onehot=to_categorical(y_encode)
print(y_onehot)

#train/test 분리
train_x,test_x,train_y,test_y=train_test_split(fish_train,y_onehot,random_state=42)
print(train_x.shape)
print(test_x.shape)

#특종 데이터에대한 스케일 조정
scaled=StandardScaler()

train_scaled=scaled.fit_transform(train_x)
test_scaled=scaled.transform(test_x)

import joblib
joblib.dump(scaled,'fish_scaler.pkl')

#읽어들일때 joblib.load('fish_scaler.pkl')
#모델설계
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

multi_model=Sequential()
multi_model.add(Dense(units=10,input_dim=5,activation='leaky_relu'))
multi_model.add(Dense(units=7,activation='softmax'))

# multi_model.summary()

multi_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#모델 학습
multi_model.fit(train_scaled,train_y,batch_size=1,epochs=500,verbose=1)
#model성능평가

print('Test acc:',multi_model.evaluate(test_scaled,test_y)[1])
multi_model.save('fish_multi_clf.keras')

print(train_scaled)