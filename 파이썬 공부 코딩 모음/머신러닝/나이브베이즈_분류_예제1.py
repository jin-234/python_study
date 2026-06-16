import pandas as pd
import numpy as np
from sklearn.datasets import load_iris#붓꽃 데이터
from sklearn.model_selection import train_test_split
#나이브 베이즈==>조건부 확률로 분류를 하는 모델
from sklearn.naive_bayes import GaussianNB#데이터특징 가우시안 정규분포를 취할때 분류 모델로 사용==>GaussianNB
from sklearn import metrics #혼동행렬
from sklearn.metrics import accuracy_score #정확도 평가
dataset=load_iris()
print(dataset)

iris_df=pd.DataFrame(dataset['data'],columns=dataset['feature_names'])
print(iris_df)

iris_df['target']=dataset['target']
print(iris_df)

#타깃 시각화를 위해서 0==>'setosa', 1==>versicolor,2==>virginica
iris_df['target']=iris_df['target'].map({ 0:'setosa', 1:'versicolor',2:'virginica'})
print(iris_df)

#setosa DF
setosa_DF=iris_df.loc[iris_df['target']=='setosa'].copy()
#versicolor DF
versicolor_DF=iris_df.loc[iris_df['target']=='versicolor'].copy()
#virsinical DF
virsinical_DF=iris_df.loc[iris_df['target']=='virginica'].copy()

import matplotlib.pyplot as plt
import seaborn as sns

fig,axes=plt.subplots(1,3,figsize=(15,7))

sns.histplot(data=setosa_DF,x='sepal length (cm)',kde=True,ax=axes[0])

sns.histplot(data=versicolor_DF,x='sepal length (cm)',kde=True,ax=axes[1])

sns.histplot(data=virsinical_DF,x='sepal length (cm)',kde=True,ax=axes[2])

plt.savefig('iris_histplot.jpeg')