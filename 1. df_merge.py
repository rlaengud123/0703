# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm import tqdm, trange


# %%
#subject = pd.read_table('../Raw Data/01.Subject/01_ELECTIVES_BASIC.txt')
under_basic = pd.read_table('../Raw Data/03.User/03_STD_ENR_BASIC.txt', sep='|', error_bad_lines=False)
under_grade = pd.read_table('../Raw Data/03.User/03_STD_ENR_GRADE.txt', sep='|', error_bad_lines=False)
graduate_basic = pd.read_table('../Raw Data/03.User/03_STD_GRD_BASIC.txt', sep='|', error_bad_lines=False)
graduate_grade = pd.read_table('../Raw Data/03.User/03_STD_GRD_GRADE.txt', sep='|', error_bad_lines=False)
temp_df = pd.read_csv('./I_to_1.csv')

# %%
# 수강평가 모두 불러오는 Fucntion
def read_data(path):
    '''
    path: 수강평가 평점 Table Folder
    '''
    total_df = pd.DataFrame()
    for file in os.listdir(path):
        if 'txt' in file:
            df = pd.read_table(os.path.join(path, file), sep = '|', error_bad_lines=False)
        elif 'csv' in file:
            df = pd.read_csv(os.path.join(path, file), sep = '|', error_bad_lines=False)
            
        
        total_df = pd.concat([total_df, df], axis = 0)
    
    total_df = total_df[total_df['QST_DIV1'] == 'B']   # 자기평가 제외
    total_df.reset_index(drop = True, inplace = True)

    return total_df

# %%
path = '../Raw Data/02.Score/02_RATINGS_LIKERT'

df_likert = read_data(path)
# %%
# 남겨야 할 과목분야 뽑아내기
stay_list = ['교양', '학부공통', '군사학'] 

under_grade = under_grade[under_grade.COUR_DIV_NM.isin(stay_list)]
graduate_grade = graduate_grade[graduate_grade.COUR_DIV_NM.isin(stay_list)]


# %%
# 학생의 기본정보와 수강이력 Table Join
def merge_table(df_basic, df_merge):

    df_merge = pd.merge(left=df_basic, right=df_merge, how='inner', on='STD_ID')
    
    return df_merge

# %%
df_under_merge = merge_table(under_basic, under_grade)   # 재학생
df_grad_merge = merge_table(graduate_basic, graduate_grade)  # 졸업생
df_basic = pd.concat([under_basic, graduate_basic])


# %%
# 재학생과 졸업생 Concat
df_merge = pd.concat([df_under_merge, df_grad_merge], axis=0)
df_merge = pd.merge(left=df_likert, right=df_merge, how='inner', on=['STD_ID', 'COUR_CD'])


# %%
# 중복하는 인덱스를 찾아 리스트로 반환하는 함수 -> 각 학생당 여러개의 수강 평점을 평균내기 위해
def dupli_index_list(df_merge):
    dupli_list =  df_merge[(df_merge.duplicated(['STD_ID', 'COUR_CD']) == False)].index
    
    return dupli_list

# 각 과목당10문항 평점을 평균내는 함수
def likert_mean(df_merge):
    dupli_list = dupli_index_list(df_merge)
    
    rating_temp = df_merge['GRADE_x']
    
    df_merge = df_merge.loc[dupli_list]
    df_merge.drop(['GRADE_x'], axis=1, inplace=True)
    
    temp_list = []
    for i in range(len(dupli_list)):
        if i == len(dupli_list)-1:
            temp = rating_temp[dupli_list[i]:].mean()
        else:
            temp = rating_temp[dupli_list[i]:dupli_list[i+1]].mean()

        temp_list.append(temp)

    df_merge['GRADE_x'] = temp_list   # 10문항 평균한 평점
    df_merge.reset_index(drop=True, inplace=True)

    return df_merge

# %%
df_merge['COUR_NM']

# %%
df_merge = likert_mean(df_merge)
df_merge.reset_index(drop=True, inplace=True)

# 양쪽 공백제거
strip_columns = list(df_merge.select_dtypes('object').columns)
for column in strip_columns:
    df_merge[column].str.strip()


#%%
for i in range(len(temp_df)):
    df_merge = df_merge.replace({ 'COUR_NM' : {temp_df['before'][i] : temp_df['after'][i]}})

# %%

df_merge.to_pickle('./df_merge.pickle')