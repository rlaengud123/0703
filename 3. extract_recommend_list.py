# %%
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm import tqdm, trange, tqdm_notebook
import math
import csv
from multiprocessing import Process
#%%
under_basic = pd.read_table('../Raw Data/03.User/03_STD_ENR_BASIC.txt', sep='|', error_bad_lines=False)
graduate_basic = pd.read_table('../Raw Data/03.User/03_STD_GRD_BASIC.txt', sep='|', error_bad_lines=False)
df_basic = pd.concat([under_basic, graduate_basic])
opened_lec_df = pd.read_csv('../Raw Data/01.Subject/01_ELECTIVES_BASIC.txt', sep='|')
df_merge = pd.read_pickle('./df_merge.pickle')
open2020_df = pd.read_pickle('./open2020.pickle')    # 해당학기에 열리는 교양과목
hanguel = pd.read_csv('./hanguel.csv')
gongtong = pd.read_table('./01_공통교양과목리스트.txt', sep='|', error_bad_lines=False)
sports = pd.read_table('./01_체육과목리스트.txt', sep='|', error_bad_lines=False)

# %%
# 공백제거
df_list = [under_basic, graduate_basic, df_basic, opened_lec_df, open2020_df]
for df in df_list:
    strip_columns = list(df.select_dtypes('object').columns)
    for column in strip_columns:
        df[column] = df[column].str.strip()

        
temp_df = pd.read_csv('./I_to_1.csv')

for i in range(len(temp_df)):
    gongtong = gongtong.replace({ 'COUR_NM' : {temp_df['before'][i] : temp_df['after'][i]}})

for i in range(len(temp_df)):
    sports = sports.replace({ 'COUR_NM' : {temp_df['before'][i] : temp_df['after'][i]}})

gongtong_list = list(set(gongtong.COUR_NM))
sports_list = list(set(sports.COUR_NM))
# %%
# 재학생, 휴학생만 추출 (추천해줄 대상)
under_basic = under_basic[(under_basic['REC_STS']=='재학') | (under_basic['REC_STS']=='휴학')]   
# %%
# 학과별로 pivot table 및 유사도 매트릭스 구하기
def get_department_table(df_merge, department_name, user_id, freshman=False):
    """
    df_merge: df_merge.py로 만든 df_merge.pickle 파일
    department_name: 학과명
    user_id: 학번
    freshman : 신입생일 경우 True (Default : True)
    """
    department_table = pd.read_pickle('department_pickle/'+str(department_name)+'.pickle') 

    department_table['ENGLISH_YN']=0
    for i in range(len(department_table)):
        if '(외국어강의)' in department_table['COUR_NM'][i] or '(영강)' in department_table['COUR_NM'][i]:
            department_table['ENGLISH_YN'][i] = 1

    if freshman:
        return department_table[['STD_ID', 'COUR_CD', 'COUR_NM', 'ENGLISH_YN', 'GRADE_x']].sort_values(by='GRADE_x', ascending=False)

    department_pivot_table = department_table.pivot_table('GRADE_x', index='STD_ID', columns=['COUR_CD','COUR_NM','ENGLISH_YN'])
    department_pivot_table.fillna(0, inplace=True)   # NaN으로 두면 유사도 계산할때 에러남
    
    similarity_matrix = cosine_similarity(department_pivot_table)
    np.fill_diagonal(similarity_matrix, 0)    # 대각선 자기자신과의 유사도는 0으로 채워줌
    similarity_matrix = pd.DataFrame(similarity_matrix, index=department_pivot_table.index, columns=department_pivot_table.index)
    
    return department_pivot_table, similarity_matrix


# %%
# 자신과 유사한 k명 추출
def find_knn(similarity_matrix, k):
    '''
    similarity_matrix : User간의 유사도 Matrix
    k : 주변의 k명으로 평점을 계산하는 인자 (e.g. k=10)
    '''
    order = np.argsort(similarity_matrix.values, axis=1)[:, :k]
    similar_user_topk = similarity_matrix.apply(lambda x: pd.Series(x.sort_values(ascending=False).iloc[:k].index, index=['top{}'.format(i) for i in range(1, k+1)]), axis=1)

    # 해당 user와 k명의 유사도 (1이 가장 가까운 것)
    knn_weight = similarity_matrix[user_id].sort_values(ascending=False)[:k]

    return similar_user_topk, knn_weight

#%%
def english(opened_list, user_lecture, young = True):
    if young:
        num = 4
        string = '(영강)'
    else:
        num = 7
        string = '(외국어강의)'
    #추천과목 리스트에서 (영강), (외국어강의)을 제거하였을 때, 사용자가 들은 과목과 일치하는가?
    english_list = []
    for i in opened_list:
        if string in i:
            english_list.append(i[:len(i)-num])
    
    temp = list(set(english_list) & set(list(user_lecture)))

    for i in range(len(temp)):
        temp[i] = temp[i] + string

    #사용자가 들은 과목 리스트에서 (영강), (외국어강의)을 제거하였을 때, 추천과목 리스트의 과목과 일치하는가?
    english_list = []
    for i in user_lecture:
        if string in i:
            english_list.append(i[:len(i)-num])
    
    temp2 = list(set(english_list) & set(list(opened_list)))

    return temp, temp2


# %%
# 최종 추천 리스트 Return
def recommend(df_merge, open2020_df, department_name, user_id, k, num_recommends, less_recommend=False, sports_len=0):
    
    freshman = False
    freshman = (len(df_merge[df_merge['STD_ID'] == user_id]) == 0)  # 유저의 수강정보가 없는경우 True
    opened_lec_list = list(set(open2020_df.COUR_NM))

    if freshman or less_recommend:    # 신입생 or 추천이 5개 미만인 사람(less_recommend) 
        department_table = get_department_table(df_merge, department_name, user_id, freshman = True)
        recommend_list = department_table.groupby(by=['COUR_NM','COUR_CD','ENGLISH_YN']).mean().sort_values(by='GRADE_x', ascending=False)
        print("\n추천에 필요한 정보가 부족합니다.\n%s의 추천 강의 리스트를 출력합니다.\n"%(department_name))
        opened_list = list(set(opened_lec_list) & set(list(recommend_list.reset_index()['COUR_NM'])))
        user_lecture = list(set(df_merge[df_merge['STD_ID'] ==user_id]['COUR_NM']))

        # 사용자 체육과목 리스트
        user_sports_list = list(set(sports_list) & set(user_lecture))

        if sports_len > 3:
            opened_list = list(set(opened_list) - set(sports_list))
        else:
            opened_sports_list = list(set(sports_list) & set(opened_list))
            opened_sports_list = list(set(opened_sports_list) - set(user_lecture))
            if len(opened_sports_list) >= 3-sports_len:
                opened_list = list(set(opened_list) - set(opened_sports_list[3-sports_len:]))

        opened_list = list(set(opened_list) - set(gongtong_list))

        final_recommend = list(set(opened_list) - set(user_lecture))
        
        final_recommend = list(recommend_list.loc[final_recommend].sort_values('GRADE_x',ascending=False).index)[:num_recommends]        
        
        final_recommend = pd.DataFrame(final_recommend, columns=['COUR_NM','COUR_CD','ENGLISH_YN'])
        less = 1

        return final_recommend[['COUR_NM','COUR_CD','ENGLISH_YN']], less, _
    else:
        department_pivot_table, similarity_matrix = get_department_table(df_merge, department_name, user_id, freshman)
    
    total_similar_user_topk, knn_weight = find_knn(similarity_matrix, k)   # 해당학과의 사람별로 top k, (해당학과인원, k) shape
    personal_topk = list(total_similar_user_topk.loc[user_id])  # 추천하고자 하는 User 1명의 유사도 top k
    
    recommend_list = knn_weight.values.reshape(-1, 1) * department_pivot_table.loc[personal_topk].replace(0, np.NaN)
                      # replace: 평점이 없는 사람의 경우 0으로 표현되지만 평균낼때 인원으로 넣어주지 않기 위해

    recommend_list = recommend_list.sum().replace(0, np.NaN).sort_values(ascending=False)
    
    recommend_list = recommend_list[recommend_list.isnull() == False].round(5)   # 추천 리스트를 많이할 경우 평가가 많이 없어 NaN으로 출력되는 과목 필터
    
    opened_list = list(set(opened_lec_list) & set(list(recommend_list.reset_index()['COUR_NM'])))
    user_lecture = list(set(df_merge[df_merge['STD_ID'] ==user_id]['COUR_NM']))

    temp, temp2 = english(opened_list, user_lecture, young=True)
    temp3, temp4 = english(opened_list, user_lecture, young=False)
    
    nation = df_merge[df_merge['STD_ID'] == user_id]['NATION_CD'].values[0]
    
    hanguel_list = list(hanguel['0'])
    if nation == 'KOR':
        opened_list = list(set(opened_list) - set(hanguel_list))

    opened_list = list(set(opened_list) - set(temp) - set(temp2) - set(temp3) - set(temp4))

    # 공통과목 리스트 제거
    opened_list = list(set(opened_list) - set(gongtong_list))

    # 사용자 체육과목 리스트
    user_sports_list = list(set(sports_list) & set(user_lecture))

    opened_sports_list_len = 3

    if len(user_sports_list) > 3:
        opened_list = list(set(opened_list) - set(sports_list))
    else:
        opened_sports_list = list(set(sports_list) & set(opened_list))
        opened_sports_list = list(set(opened_sports_list) - set(user_lecture))
        if len(opened_sports_list) > 3:
            opened_list = list(set(opened_list) - set(opened_sports_list[3:]))
        opened_sports_list_len = len(opened_sports_list)


    # 첫 추천 리스트에서 수강한 과목 제외
    final_recommend = list(set(opened_list) - set(user_lecture))

    # 점수순으로 정렬하고 지정한 추천 개수(num_recommends)만큼 출력
    final_recommend = recommend_list.reset_index(['COUR_CD', 'ENGLISH_YN']).loc[final_recommend].sort_values(by=0,ascending=False)[:num_recommends]
    less = 0
    final_recommend = final_recommend.reset_index()
    return final_recommend[['COUR_NM','COUR_CD','ENGLISH_YN']], less, opened_sports_list_len

#%%
dept_kratio = pd.read_csv('./학과별_k비율.csv', encoding='cp949')
#%%
dept = pd.DataFrame(columns=dept_kratio.columns)
count = 0
i = 0
while True:
    dept.loc[count] = dept_kratio.loc[i]
    count += 1
    i += 12
    if i > 647:
        break
#%%
std_dept = pd.DataFrame(df_merge[['STD_ID','DEPT_CD_NM']].drop_duplicates()['DEPT_CD_NM'].value_counts())
std_dept.reset_index(inplace=True)
#%%
# 학생 학번 입력 받는 곳
###########################여기########################################
# user_df = pd.read_csv('./.csv') ##########<<<<<<<<<<<<<<<<<<<<<<<파일
# user_list = list(user_df.STD_ID)

#user_list = ['PYI0ge21vtO6G6AQ059rvK/P1YtFTlSwHOklEtGe6ME='] # 의학과
#%%
user_id = list(user_list)[0]
# %%
# 추천
# std_dept = pd.DataFrame(df_merge[['STD_ID','DEPT_CD_NM']].drop_duplicates()['DEPT_CD_NM'].value_counts())
# std_dept.reset_index(inplace=True)
result_df = pd.DataFrame(columns=['STD_ID','DEPT','k','k_ratio','recommends','less'])
if __name__ == "__main__":
    with open('result.csv', 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['STD_ID','DEPT','k','k_ratio','ranking','recommends_COUR_CD','recommends_COUR_NM','recommends_ENGLISH_YN','less']
        result = csv.DictWriter(csvfile, delimiter = ',', fieldnames=fieldnames)
        result.writeheader()
        user_list = under_basic[(under_basic['REC_STS']=='재학') | (under_basic['REC_STS']=='휴학')]['STD_ID']
        user_list.reset_index(drop=True, inplace=True)
        count = 0
        for user_id in tqdm_notebook(user_list):
            department_name = df_basic[df_basic['STD_ID'] == user_id]['DEPT_CD_NM'].values[0]
            k_list = list(dept[dept['DEPT']==department_name]['k'])
            for k_ratio in k_list:
                k = math.ceil(k_ratio * std_dept[std_dept['index']==department_name]['DEPT_CD_NM'].values[0])
                num_recommends = 21
                final_recommend, less, sports_len = recommend(df_merge, open2020_df, department_name, user_id , k, num_recommends, less_recommend=False, sports_len = 0)
                ranking = 1
    #            if len(final_recommend) != 0:
    #                print("추천 강의 리스트 입니다.\n", final_recommend)
                if len(final_recommend) < 21:   # 다시 해봐야함, 기존에 들은거 뺀게 5이하면 추천 or 완전 추천리스트가 5이하면 추천
                    less_recommend = True
                    for j in range(len(final_recommend)):
                        result.writerow({'STD_ID' : user_id, 'DEPT' : department_name, 'k' : k, 'k_ratio' : k_ratio, 'ranking' : ranking,'recommends_COUR_CD' : final_recommend['COUR_CD'][j],'recommends_COUR_NM' : final_recommend['COUR_NM'][j], 'recommends_ENGLISH_YN' : final_recommend['ENGLISH_YN'][j],'less' : less})
                        count += 1
                        ranking += 1
                    final_recommend, less, _ = recommend(df_merge, open2020_df, department_name, user_id,k,num_recommends,less_recommend=True, sports_len = sports_len)
                    #print(final_recommend[:num_recommends])
                for j in range(len(final_recommend)):
                    result.writerow({'STD_ID' : user_id, 'DEPT' : department_name, 'k' : k, 'k_ratio' : k_ratio, 'ranking' : ranking, 'recommends_COUR_CD' : final_recommend['COUR_CD'][j],'recommends_COUR_NM' : final_recommend['COUR_NM'][j], 'recommends_ENGLISH_YN' : final_recommend['ENGLISH_YN'][j],'less' : less})
                    count += 1
                    ranking += 1
                    if ranking > 21:
                        break



# %%
