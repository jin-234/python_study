from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
bst=pd.read_csv('/home/jin/deeplearning_prj/20260610/BostonHousing.csv')
# print(bst)
# bst.info()

bst_value=bst.iloc[:,0:13]
bst_target=bst.iloc[:,13]
train_x,test_x,train_y,test_y=train_test_split(bst_value,bst_target,test_size=0.3,random_state=42)

scaled=StandardScaler()
train_scale=scaled.fit_transform(train_x)
test_scale=scaled.transform(test_x)

multi_model=Sequential()
multi_model.add(Dense(units=30,input_dim=13,activation='leaky_relu'))
multi_model.add(Dense(units=6,activation='leaky_relu'))

multi_model.add(Dense(units=10,activation='leaky_relu'))
multi_model.add(Dense(units=1,activation='linear'))

multi_model.summary()


multi_model.compile(loss='mse',optimizer='adam',metrics=['mse'])
multi_model.fit(train_scale,train_y,batch_size=1,epochs=100,verbose=1)

pre=multi_model.predict(train_scale).flatten()

for i in range(1,10):
    print('실제 가격 : {:.3f},예상 가격: {:3f}'.format(test_y.iloc[i],pre[i]))
pre=multi_model.predict(train_x)
