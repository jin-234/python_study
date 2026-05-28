import pandas as pd
import numpy as np
import re
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width',1000)
pd.set_option('max_colwidth', 1000)


tipsgen=pd.read_csv('tips.csv')
#tips.csv 파일을 읽어서 gender 컬럼의 'male'을 0으로 'female을 1로 일괄 변경시키기
def Gendercheck(arg):
    if arg=='Male':
        return 0
    else:
        return 1
tipsgen['gender']= tipsgen['gender'].apply(Gendercheck)
print(tipsgen)
# tipsgen['gender'] = tipsgen['gender'].apply(lambda x: '0' if x == 'Male' else '1')#더 간단하게 하는법
# print(tipsgen)
#day 컬럼에 몇개의 종류가 있는지

print(len(tipsgen['day'].unique()))
print('='*80)
tipsgen.info()
#요일이 'Sat'와 'Thur'인 항목만 추출출력
subset=tipsgen.loc[tipsgen['day'].isin(['Sat','Thur']) , : ].copy()
print(subset)
#토요일과 목요일의 테이블 인원수의 평균은?
#size 컬럼의 평균 계산
print(subset['tip'].mean())