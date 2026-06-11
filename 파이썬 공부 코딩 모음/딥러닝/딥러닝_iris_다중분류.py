from sklearn.datasets import load_iris
iris=load_iris()
# print(iris)


#iris['data]==>4개의 특성 데이터를 trqainx
# iris['target']==trainy
# #setosa,versicolor,virginica 3가지 브ㅜㅅ꽃 클래스 분류(다중분류)
# 1.데이터 전처리

from tensorflow.keras.utils import to_categorical
os_target=to_categorical(iris['target'])
# print(iris['data'])
# print(iris['target'])
# 2.분할


from sklearn.model_selection import train_test_split
trian_x,test_x,train_y,test_y=train_test_split(iris['data'],os_target,random_state=42)

#3.스케일 조저ㅗㅇ

from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
train_scale=scale.fit_transform(trian_x)
test_scale=scale.transform(test_x)


# 4.다중분류 모델 성계

from keras.models import Sequential
from keras.layers import Dense

iris_model=Sequential()
iris_model.add(Dense(units=8,input_dim=4,activation='leaky_relu'))
iris_model.add(Dense(units=3,activation='softmax'))
iris_model.compile(metrics=['accuracy'],optimizer='adam',loss='categorical_crossentropy')
# iris_model.summary()

# 5.학습
iris_model.fit(train_scale,train_y,batch_size=32,epochs=200,verbose=1)

print(iris_model.evaluate(test_scale,test_y)[1])
# 별도)스케일 과 모델은 별도 저장
import joblib
joblib.dump(scale,'iris_scale.pkl')

iris_model.save('iris_model.keras')
# 단) categorical_crossentrypy만 사용