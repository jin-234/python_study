import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D,Input  # Maxpooling 반대 Upsanpoling2D

(x_train,_),(x_test,_)=fashion_mnist.load_data()#28,28 패션 관련 흑백 이미지

# print(x_train)
# print(len(x_train))#60000

x_train=x_train.reshape(-1,28,28,1)/255.0#-1: 자동으로 결정해라
x_test=x_test.reshape(-1,28,28,1)/255.0
from tensorflow.keras.models import load_model

encode_model=load_model('/home/jin/deeplearning_prj/20260619/mnist_model_callback.keras')

predict_img=encode_model.predict(x_test)
print(predict_img[0])
print(predict_img.shape)

import matplotlib.pyplot as plt
num=5#원본과 5개 비교
plt.figure(figsize=(15,7))
for i in range(num):
    ax1=plt.subplot(2,num,i+1)
    ax1.imshow(x_test[i].reshape(28,28),cmap='gray')
    ax1.set_title('original_image %d'%i)
    ax1.axis('off')
    
    ax2=plt.subplot(2,num,i+num+1)
    ax2.imshow(predict_img[i].reshape(28,28),cmap='gray')
    ax2.set_title('autoenc_img%d'%i)
    ax2.axis('off')
plt.savefig('/home/jin/deeplearning_prj/20260619/predict_image.jpeg')