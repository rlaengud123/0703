#%%
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
#%%
def save_scoring(df_merge, department_name):
    department_table = df_merge[df_merge['DEPT_CD_NM']== department_name]
    department_table.reset_index(inplace=True) # 해당 학과 뽑아서
    department_table['ratio'] = 0
    department_table['elder'] = 0
    department_table['raw_GRADE_x'] = department_table['GRADE_x']
    department_table['GRADE_x'] = department_table['GRADE_x'] - 4.0 # normalize
    print()
    # 선배 별 rating
    elder_rating = pd.DataFrame(columns=['year','rating'])
    elder_rating['year'] = range(2000,2020)
    rating = 0.6
    for j in range(0, 20):
        elder_rating['rating'][j] = rating
        rating += 0.02

    cour_cd_sum = department_table['COUR_CD'].value_counts().sum() # 해당학과의 교양과목 개수
    cour_cd_list = dict(department_table['COUR_CD'].value_counts())

    for k,v in cour_cd_list.items():
        cour_cd_list[k] = v/cour_cd_sum # 과목 비율 리스트
        
    for i in trange(len(department_table)):
        for k,v in cour_cd_list.items():
            if k == department_table.at[i,'COUR_CD']:
                department_table.at[i,'GRADE_x'] = department_table.at[i,'GRADE_x'] * v #이부분 평점으로 바꿔야함. 25000 raw하는데 2분걸림.
                department_table.loc[i,'ratio'] = v
        for j in range(len(elder_rating)):
            if elder_rating['year'][j] == department_table.at[i, 'YR_x']:
                department_table.at[i,'GRADE_x'] = department_table.at[i,'GRADE_x'] * elder_rating['rating'][j] #이부분 
                department_table.loc[i,'elder'] = elder_rating['rating'][j]
    if os.path.isdir('./department_pickle') == False:
        os.mkdir('./department_pickle')
    
    department_table.to_pickle('department_pickle/' + str(department_name) + '.pickle')

    return None

#%%
df_merge = pd.read_pickle('./df_merge.pickle')

for department_name in df_merge['DEPT_CD_NM'].value_counts().index :
    save_scoring(df_merge, department_name)

# %%