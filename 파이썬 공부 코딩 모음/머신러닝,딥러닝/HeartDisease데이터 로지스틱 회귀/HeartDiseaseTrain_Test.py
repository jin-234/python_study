import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#pd.set_option('display.colheader_justify','center')  # 컬럼 중앙 출력

heart_df=pd.read_csv('/home/jin/deeplearning_prj/20260609/HeartDiseaseTrain-Test.csv')

heart_df['sex']=heart_df['sex'].map({'Male':1,'Female':0})
heart_df['chest_pain_type']=heart_df['chest_pain_type'].map({'Typical angina':1,'Atypical angina' :2,'Non-anginal pain':3, 'Asymptomatic':4})
heart_df['fasting_blood_sugar']=heart_df['fasting_blood_sugar'].map({'Lower than 120 mg/ml':0,'Greater than 120 mg/ml':1})#'Lower than 120 mg/ml' 'Greater than 120 mg/ml
heart_df['rest_ecg']=heart_df['rest_ecg'].map({'ST-T wave abnormality':1,'Normal':0, 'Left ventricular hypertrophy':2})# ['ST-T wave abnormality' 'Normal' 'Left ventricular hypertrophy']
heart_df['exercise_induced_angina']=heart_df['exercise_induced_angina'].map({'No': 0,'Yes':1})
heart_df['slope']=heart_df['slope'].map({'Downsloping':3, 'Upsloping':1 ,'Flat':2})#['Downsloping' 'Upsloping' 'Flat']
heart_df['vessels_colored_by_flourosopy']=heart_df['vessels_colored_by_flourosopy'].map({'Two':2 ,'Zero' :0,'One' :1,'Three':3, 'Four':4})#['Two' 'Zero' 'One' 'Three' 'Four']
heart_df['thalassemia']=heart_df['thalassemia'].map({'Reversable Defect':7, 'Fixed Defect':6, 'Normal' :3,'No':0})#['Reversable Defect' 'Fixed Defect' 'Normal' 'No']
# heart_df.info()
# print(heart_df['thalassemia'].unique())
heart_target=heart_df['target'].values
heart_df_data=heart_df[['age', 'sex', 'chest_pain_type', 'resting_blood_pressure','cholestoral', 'fasting_blood_sugar', 'rest_ecg', 'Max_heart_rate', 'exercise_induced_angina', 'oldpeak', 'slope','vessels_colored_by_flourosopy', 'thalassemia']]
#print(heart_target)

train_x,test_x,train_y,test_y=train_test_split(heart_df_data,heart_target,random_state=35) 

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
train_scale=scaler.fit_transform(train_x)
test_scale=scaler.transform(test_x)
#print(train_scale)

lr_model=LogisticRegression()
lr_model.fit(train_scale,train_y)
print(lr_model.score(train_scale,train_y))
print(lr_model.score(test_scale,test_y))

print(lr_model.predict(test_scale))
