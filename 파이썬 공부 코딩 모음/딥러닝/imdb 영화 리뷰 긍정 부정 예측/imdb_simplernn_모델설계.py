from tensorflow.keras.datasets import imdb
import numpy as np
#사용안하는 함수는 _로 표현
(train_x,train_y),(test_x,test_y)=imdb.load_data(num_words=500)#내부적으로 0,1,2,3,은 특수토큰

print(len(train_x))#25000
print(len(test_x))#25000
print(train_x[0])
#타깃 이진분류 :0(부정)/12500개,1(긍정)/12500개
print(np.unique(train_y[0],return_counts=True))#라벨(정답)도 수치형으로 변환되어있음

from sklearn.model_selection import train_test_split

train_x,val_x,train_y,val_y=train_test_split(train_x,train_y,test_size=0.2,random_state=43)
print(len(train_x))#20000
print(len(val_x))#5000
#train_x =>정수백터화 되어있음
print(len(train_x[0]))#20000
print(len(train_x[1]))
word_index=imdb.get_word_index()#어떤 단어를 어떤 수치로 변환했는지에대한 정보
#1차==>길이가 다른 리뷰 정수데이터 배열을 길이가 동일한 정수 배열로 변경
from tensorflow.keras.preprocessing.sequence import pad_sequences

#길이를 100으로 모두 변경할때 짧은거는 0으로 채우고, 긴거는 버린다
train_seq=pad_sequences(train_x,maxlen=100)
print(train_seq[0])
print(train_seq.shape)
print(train_seq[30])

val_seq=pad_sequences(val_x,maxlen=100)
print(val_seq[0])
print(val_seq.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN,Dense,Input,Dropout
rnnmodel=Sequential()
input=Input(shape=(100,))#keras3버전이라 입력층 필요
rnnmodel.add(input)
rnnmodel.add(Embedding(input_dim=500,input_length=100,output_dim=16))
#input_dim=최대 숫자 크기 +1,input_length= 문장의 길이,output_dim=원핫인코딩을 16개의 임베딩으로 가공)
rnnmodel.add(SimpleRNN(8,input_shape=(100,500)))
rnnmodel.add(Dense(units=32,activation='leaky_relu'))
rnnmodel.add(Dropout(0.3))
rnnmodel.add(Dense(units=16,activation='leaky_relu'))
rnnmodel.add(Dropout(0.3))
rnnmodel.add(Dense(units=8,activation='leaky_relu'))
rnnmodel.add(Dropout(0.3))
rnnmodel.add(Dense(units=4,activation='leaky_relu'))
rnnmodel.add(Dropout(0.3))
rnnmodel.add(Dense(units=1,activation='sigmoid'))

rnnmodel.summary()

import tensorflow as tf
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001) # 0.00001
rnnmodel.compile(loss='binary_crossentropy', optimizer = optimizer,
                 metrics = ['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

checkpoint=ModelCheckpoint(save_best_only=True,filepath='/home/jin/deeplearning_prj/20260617/imbd_model.keras')
stop=EarlyStopping(patience=3,restore_best_weights=True)

history=rnnmodel.fit(train_seq,train_y,epochs=100,batch_size=64,validation_data=(val_seq,val_y),callbacks=[checkpoint,stop])


import matplotlib.pyplot as plt
plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')
plt.savefig('/home/jin/deeplearning_prj/20260617/graph.jpeg')

# print(word_index)

# for word ,index in word_index.items():
#     # print(word,index)#빈도수가 가장 높은건 index 1
#     if index==1:
#         print(word)

# #train_x[0]==>정수백터를 ==>단어들의 모음으로 치환시키는 작업
# conv_word_index =dict( [ (idx+3, word) for (word, idx) in word_index.items() ] )
# # print(conv_word_index)

# # for word,idx in conv_word_index.items():
# #     if idx==4:
# #         print(word)

# decode_sentance=' '.join([  conv_word_index[i]  if i in conv_word_index else '?'    for i in train_x[0]     ])
# print(decode_sentance)