
import pandas as pd
import numpy as np

pokemon_df=pd.read_csv('/home/jin/deeplearning_prj/20260613 개인복습/Pokemon.csv')

# print(pokemon_df)
# pokemon_df.info()
pokemon_df['Legendary']=pokemon_df['Legendary'].map({True:1,False:0})#정수형태로 변환
# print(pokemon_df['Legendary'])

#문자열 데이터를 숫자 인코딩으로 변환
from sklearn.preprocessing import LabelEncoder
encode=LabelEncoder()
pokemon_target=pokemon_df['Type 1'].copy()
pokemon_target=encode.fit_transform(pokemon_target)
# print(pokemon_target.max())
#타겟 데이터를 원핫인코딩으로 변환
from keras.utils import to_categorical
pk_onehot=to_categorical(pokemon_target)

pokemon_values=pokemon_df.iloc[:,4:].copy()#'Name', 'Type 1', 'Type 2', 'Total', 'HP', 'Attack', 'Defense','Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 'Legendary'],
# print(len(pokemon_values.columns))

#트레인,테스트세트 분리
from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y=train_test_split(pokemon_values,pk_onehot,random_state=12)

#데이터 스케일링
from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
scale_train=scale.fit_transform(train_x)
scale_test=scale.transform(test_x)

#모델 생성
from keras.models import Sequential
from keras.layers import Dense

pk_model=Sequential()

pk_model.add(Dense(units=36,input_dim=9,activation='leaky_relu'))
pk_model.add(Dense(units=24,activation='leaky_relu'))
pk_model.add(Dense(units=20,activation='leaky_relu'))
pk_model.add(Dense(units=18,activation='softmax'))

pk_model.summary()

#조기종료 콜백 추가
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

modelpath='/home/jin/deeplearning_prj/20260613 개인복습/pk_best_model.keras'
checkpoint=ModelCheckpoint(filepath=modelpath,monitor='val_loss',verbose=1,save_best_only=True)

erlydstop=EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)
pk_model.compile(metrics=['accuracy'],optimizer='adam',loss='categorical_crossentropy')

#모델 학습
pk_model.fit(scale_train,train_y,validation_data=(scale_test,test_y) ,verbose=1,epochs=1000,batch_size=32,callbacks=[checkpoint,erlydstop])

print(pk_model.evaluate(scale_test,test_y))