import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import os
import math

os.getcwd()
grade_basic = pd.read_table('../Raw Data/01.Subject/01_ELECTIVES_BASIC.txt', sep='|')
grade_basic.columns
col = ['YR', 'TERM', 'COUR_CD', 'PROF_NM', 'CREDIT', 'DEPT_CD', 'COUR_DIV', 'COUR_KIND', 'GROUP_CD', 
'ABSOLUTE_YN', 'ENG_LEC_YN', 'LEC_CONV', 'MOOC_YN', 'FLIPPED_CLASS_YN', 'FLEXIBLE_SCHOOL_YN', 'ACTION']

# 계절학기는 없음
np.sum(grade_basic.TERM.apply(lambda x : 'S' in x))
sample = grade_basic[col]

# 10개 까지만 쓸모있음.
sample.loc[sample.ACTION.isnull(),'ACTION'] = '0000000000'
sample['ACTION'] = sample.ACTION.apply(lambda x: x[:10])
sample

n=0
for i in ['강의','발표','토론','실험','실습','협동학습','개별지도','집단지도','퀴즈','Q&A']:
    sample[i]=sample.ACTION.apply(lambda x: x[n])
    n+=1

# action에 대한 결측 값은 다 N처리
for i in ['강의','발표','토론','실험','실습','협동학습','개별지도','집단지도','퀴즈','Q&A']:
    sample.loc[sample[i]=='0',i]='N'
    

sample.drop(['ACTION'],axis=1, inplace=True)
sample.GROUP_CD.unique()
cour = sample.COUR_CD.unique().tolist()


# 확인 해봤는데 같은 학수번호들의 결측값은 ACTION제외하고 모두 그 학수번호에서 똑같이 나왔음.
# 앞의 열들에는 결측 값이 없었음
for i in range(len(cour)):
    for j in sample.columns:
        sample.loc[sample.COUR_CD==cour[i], j] = sample.loc[sample.COUR_CD==cour[i], j].fillna(method='ffill')


for i in sample.columns:
    sample.loc[sample[i]=='Y',i] = 1
    sample.loc[sample[i]=='N',i] = 0
    sample.loc[sample[i].isnull(), i] = 0


''' 여기서 group_by할 때 조건 대로 해줘야 함'''
sample.iloc[:,4:] = sample.iloc[:,4:].astype(int)
sample[(sample.COUR_CD=='ARCH203') & (sample.PROF_NM=='김자영')]
#### 여기구간
# g_table.LEC_CONV.unique() 교수 명 넣으면 학수번호보다 교수가 더 많음.

g_ = pd.DataFrame(sample.groupby(['COUR_CD', 'PROF_NM', 'DEPT_CD','GROUP_CD']))
lis=[]
for i in range(4921):
    lis.append(g_.iloc[i,1].iloc[np.where(g_.iloc[i,1]==(max(g_.iloc[i,1].YR)))[0][0]].name)
    
sample = sample.loc[lis]

# 네 개에 대하여 그룹바이함.
g_table = sample.groupby(['COUR_CD', 'PROF_NM', 'DEPT_CD','GROUP_CD']).mean()
g_table = g_table.reset_index()
g_table.ABSOLUTE_YN.unique()


# 연도만 max인 것 골랐을 때, 겹치는 것이 없다는 것을 보여주는 코드
for i in range(6,g_table.shape[-1]):
    if ( len(g_table[(g_table.iloc[:,i] >0) & (g_table.iloc[:,i] <1)]))>0:
        print(i)
g_table
'''
# 반올림 해버림
for i in ['강의','발표','토론','실험','실습','협동학습','개별지도','집단지도','퀴즈','Q&A']:
    g_table[i] = g_table[i].apply(lambda x: round(x))

for i in range(6,10):
    g_table.iloc[:,i] = g_table.iloc[:,i].apply(lambda x: round(x))
g_table.iloc[:,1:] = g_table.iloc[:,1:].astype(int)
'''

g_table.drop('YR',axis=1,inplace=True)
g_table.to_pickle('cour_opt_table.pickle')
