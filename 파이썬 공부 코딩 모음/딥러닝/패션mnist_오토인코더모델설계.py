import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D,Input  # Maxpooling 반대 Upsanpoling2D

(x_train,_),(x_test,_)=fashion_mnist.load_data()#28,28 패션 관련 흑백 이미지

# print(x_train)
# print(len(x_train))#60000

x_train=x_train.reshape(-1,28,28,1)/255.0#-1: 자동으로 결정해라
x_test=x_test.reshape(-1,28,28,1)/255.0

# print(x_train.shape)
# print(x_test.shape)

# print(x_train[0])#255.0으로 나누어서 정규화하자

mnist_model=Sequential()

mnist_model.add(Input((28,28,1)))

mnist_model.add(Conv2D(filters=16,kernel_size=(3,3),padding='same',activation='leaky_relu'))
mnist_model.add(MaxPooling2D(pool_size=(2,2)))
mnist_model.add(Conv2D(filters=8,kernel_size=(3,3),padding='same',activation='leaky_relu'))
mnist_model.add(MaxPooling2D(pool_size=(2,2)))
mnist_model.add(Conv2D(filters=8,kernel_size=(3,3),strides=2,padding='same',activation='leaky_relu'))

mnist_model.add(Conv2D(filters=8,kernel_size=(3,3),padding='same',activation='leaky_relu'))
mnist_model.add(UpSampling2D())
mnist_model.add(Conv2D(filters=8,kernel_size=(3,3),padding='same',activation='leaky_relu'))
mnist_model.add(UpSampling2D())
mnist_model.add(Conv2D(filters=16,kernel_size=(3,3),padding='valid',activation='leaky_relu'))
mnist_model.add(UpSampling2D())
mnist_model.add(Conv2D(filters=1,kernel_size=(3,3),padding='same',activation='sigmoid'))
#28,28,1 모양의 자료에 모든 손실 자료에 0-1사이의 값을 주어서 손실값을 구함

mnist_model.summary()


# #아담의 디폴트 학습률(lr)==>0.001(1e-3) 경사하강법에서 기울기 조절 정도
# mnist_model.compile(optimizer='adam',loss='mse')#,metrics=['accuracy']

# from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

# checkpoint=ModelCheckpoint(save_best_only=True,filepath='/home/jin/deeplearning_prj/20260619/mnist_model_callback.keras',monitor='val_loss',verbose=1)
# stop=EarlyStopping(patience=3,restore_best_weights=True,verbose=1)

# history=mnist_model.fit(x_train,x_train,batch_size=128,epochs=50,verbose=1,validation_data=(x_test,x_test),callbacks=[stop,checkpoint])
# mnist_model.save('/home/jin/deeplearning_prj/20260619/autoencoder.keras')

# train_loss=history.history['loss']
# valloss=history.history['val_loss']
# import matplotlib.pyplot as plt

# plt.plot(train_loss)
# plt.plot(valloss)
# plt.savefig('/home/jin/deeplearning_prj/20260619/graph.jpeg')