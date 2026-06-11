from tensorflow.keras.models import load_model #학습된 모델을 불러올때 사용
import pandas as pd
import joblib
scaler=joblib.load('scaler.pkl')
titanic_bestmodel=load_model('titanic_bestmodel.keras')

titanic_bestmodel.summary()

#머신러닝 처럼 새로운 3사람의 정보를 만들어서 생존 여부 예픅

test_pep=pd.DataFrame([[0,23,0,1],[0,8,1,0],[1,26,1,0],[0,40,1,0]],index=['김명진','마오카이','아리','노틸러스'])
test_scaled=scaler.transform(test_pep)
for i in range(0,len(test_scaled)):
    if titanic_bestmodel.predict(test_scaled)[i] >=0.5:
        print('survive')
    else:
        print('fail')
        
#     print(titanic_bestmodel.predict(test_pep)[i])
print(titanic_bestmodel.predict(test_scaled))
# [0,23,0,1],[0,8,1,0],[1,26,0,1],[0,40,1,0]
# [[0.09854521]
#  [0.35846743]
#  [0.7514918 ]
#  [0.23395282]]

# [[0,23,0,1],[0,8,1,0],[1,26,0,1],[1,40,1,0]]
# [[0.07933711]
#  [0.26748154]
#  [0.46831718]
#  [0.77714634]]


# [0,23,0,1],[0,8,1,0],[1,26,1,0],[0,40,1,0]
# [[0.04324083]
#  [0.29844353]
#  [0.919866  ]
#  [0.27839082]]