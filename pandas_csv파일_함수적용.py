from unittest import result

from logging import info

import pandas as pd
import numpy as np
import re
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width',1000)
pd.set_option('max_colwidth', 1000)


passdf=pd.read_csv('서울특별시_지하철 승하차 승객수.csv',encoding='CP949')
# print(passdf)

#문제)
#==>호선_명칭 컬럼 데이터 중 숫자 문자가 있는 호선_명칭 데이터만 추출해서 출력

# print(passdf["호선_명칭"])


def Numbercheck(ags):
    result=re.findall(r'[0-9]+',ags)
    if result==[]:
        return False
    else:
        return True

subset=passdf.loc[passdf['호선_명칭'].apply(Numbercheck),:]

print(subset)
