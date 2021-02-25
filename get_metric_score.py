# %%
import os
import math
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

# %%
under_basic = pd.read_table('../Raw Data/03.User/03_STD_ENR_BASIC.txt', sep='|', error_bad_lines=False)
graduate_basic = pd.read_table('../Raw Data/03.User/03_STD_GRD_BASIC.txt', sep='|', error_bad_lines=False)
df_basic = pd.concat([under_basic, graduate_basic])
opened_lec_df = pd.read_csv('../Raw Data/01.Subject/01_ELECTIVES_BASIC.txt', sep='|')   # 지금까지 열렸던 모든 교양과목
df_merge = pd.read_pickle('./df_merge.pickle')

open2020_df = pd.read_pickle('./open2020.pickle')    # 추천하고자 하는 해당 학기에 열리는 과목
test2020_dataset = pd.read_table('../Raw Data/03.User/03_STD_ENR_GRADE_2020.txt', sep='|', error_bad_lines=False)    # Metric2 계산을 위한 테스트 셋

stay_list = ['교양', '학부공통', '군사학']
test2020_dataset = test2020_dataset[test2020_dataset['COUR_DIV_NM'].isin(stay_list)]

# 재학생, 휴학생만 추출 (추천해줄 대상)
under_basic = under_basic[(under_basic['REC_STS']=='재학') | (under_basic['REC_STS']=='휴학')]   
# %%
# 공백제거
df_list = [under_basic, graduate_basic, df_basic, opened_lec_df, open2020_df, test2020_dataset]
for df in df_list:
    strip_columns = list(df.select_dtypes('object').columns)
    for column in strip_columns:
        df[column] = df[column].str.strip()

# %%
# 학과별로 pivot table 및 유사도 매트릭스 구하기
def get_department_table(df_merge, department_name, user_id, score=True, freshman=False): #여기기기기
    """
    score : score에 대한 방법론을 적용할 것인지에 대한 인자 (Default : True)
    freshman : 신입생일 경우 True (Default : True)
    """
    if score:  # score1 or score2
        department_table = pd.read_pickle('department_pickle/'+str(department_name)+'.pickle') 
        if freshman:
            department_table['GRADE_x'] = department_table['raw_GRADE_x'] * department_table['ratio']   # score1 생성
            return department_table[['STD_ID', 'COUR_NM', 'GRADE_x']].sort_values(by='GRADE_x', ascending=False)
        #여기가 수정되었음.
    else:   # score0
        department_table = df_merge[df_merge['DEPT_CD_NM']== department_name]
        department_table.reset_index(inplace=True)
        department_table = department_table[['STD_ID', 'COUR_NM', 'GRADE_x']]

    department_pivot_table = department_table.pivot_table('GRADE_x', index='STD_ID', columns='COUR_NM')  # score0, 1, 2구할때 수정하는 부분
    department_pivot_table.fillna(0, inplace=True)   # NaN으로 두면 유사도 계산할때 에러남
    
    department_length = len(department_pivot_table)    # 해당학과의 학생수 (k의 비율 구하기 위해)

    similarity_matrix = cosine_similarity(department_pivot_table)
    np.fill_diagonal(similarity_matrix, 0)    # 대각선 자기자신과의 유사도는 0으로 채워줌
    similarity_matrix = pd.DataFrame(similarity_matrix, index=department_pivot_table.index, columns=department_pivot_table.index)
    
    return department_pivot_table, similarity_matrix, department_length


# %%
# 자신과 유사한 k명 추출
def find_knn(similarity_matrix, k):
    '''
    similarity_matrix : User간의 유사도 Matrix
    k : 주변의 k명으로 평점을 계산하는 인자 (e.g. k=10)
    '''
    # Max K (학과 학생이 너무 많은 경우 비율로 해도 너무많아지므로)
    # if k > 100:
    #     k = 100

    order = np.argsort(similarity_matrix.values, axis=1)[:, :k]
    similar_user_topk = similarity_matrix.apply(lambda x: pd.Series(x.sort_values(ascending=False).iloc[:k].index, 
          index=['top{}'.format(i) for i in range(1, k+1)]), axis=1)

    # 해당 user와 k명의 유사도 (1이 가장 가까운 것)
    knn_weight = similarity_matrix[user_id].sort_values(ascending=False)[:k]

    return similar_user_topk, knn_weight


# %%
# 최종 추천 리스트 Return
def recommend(df_merge, open2020_df, test2020_dataset, department_name, user_id, k_ratio, num_recommends, score=True, less_recommend=False):
    '''
    df_merge: df_merge.py로 만든 df_merge.pickle 파일
    open2020_df: 추천하고자 하는 해당 학기에 열리는 과목
    test2020_dataset: 2020년 1학기로 테스트한 파일
    '''
    try:
        freshman = False
    #    freshman = (len(df_merge[df_merge['STD_ID'] == user_id]) == 0)  # 유저의 수강정보가 없는경우
        opened_lec_list = list(set(open2020_df.COUR_NM))   # 해당학기에 열리는 과목

        # 신입생 or 추천이 5개 미만인 사람(less_recommend) 
        if freshman or less_recommend:    
            department_table = get_department_table(df_merge, department_name, user_id, score, freshman = True)
            recommend_list = department_table.groupby(by='COUR_NM').mean().sort_values(by='GRADE_x', ascending=False)
            print("\n추천에 필요한 정보가 부족합니다.\n%s의 추천 강의 리스트를 출력합니다.\n"%(department_name))
            opened_list = list(set(opened_lec_list) & set(list(recommend_list.index)))
            
            final_recommend = recommend_list.loc[opened_list].sort_values('GRADE_x',ascending=False)[:num_recommends]        
        
            return final_recommend
        else:
            department_pivot_table, similarity_matrix, department_length = get_department_table(df_merge, department_name, user_id, score, freshman)
        
        k = math.ceil(department_length * k_ratio)    # 해당학과 인원 비율을 적용한 k

        total_similar_user_topk, knn_weight = find_knn(similarity_matrix, k)   # 해당학과의 사람별로 top k, (해당학과인원, k) shape
        personal_topk = list(total_similar_user_topk.loc[user_id])  # 추천하고자 하는 User 1명의 유사도 top k
        
        recommend_list = knn_weight.values.reshape(-1, 1) * department_pivot_table.loc[personal_topk].replace(0, np.NaN)
                        # replace: 평점이 없는 사람의 경우 0으로 표현되지만 평균낼때 인원으로 넣어주지 않기 위해

        recommend_list = recommend_list.sum().replace(0, np.NaN).sort_values(ascending=False)
        
        recommend_list = recommend_list[recommend_list.isnull() == False].round(5)   # 추천 리스트를 많이할 경우 평가가 많이 없어 NaN으로 출력되는 과목 필터
        
        opened_list = list(set(opened_lec_list) & set(list(recommend_list.index)))   # 추천리스트와 해당학기에 열린 과목 교집합
        
        # 사용자 수강한 과목 이력
        user_lecture = list(set(df_merge[df_merge['STD_ID'] ==user_id]['COUR_NM']))

        # 첫 추천 리스트에서 수강한 과목 제외
        final_recommend = list(set(opened_list) - set(user_lecture))

        # 점수순으로 정렬하고 지정한 추천 개수(num_recommends)만큼 출력
        final_recommend = list(recommend_list[final_recommend].sort_values(ascending=False).index)[:num_recommends]
        
        recommend_length = len(final_recommend)

        label = list(test2020_dataset[test2020_dataset['STD_ID'] == user_id]['COUR_NM'])    # 학생이 테스트 학기에 들은 과목

        intersection_subject = list(set(final_recommend) & set(label))
        intersection_subject_count = len(intersection_subject)

        accuracy = intersection_subject_count / len(label)
        target_length = len(label)
    
    except:
        print(department_name, k)
        return False, False, False, False

    return intersection_subject_count, recommend_length, target_length, accuracy

# %%
testset_user_list = list(set(test2020_dataset['STD_ID']))
user_list = under_basic[(under_basic['SCHOOL_YEAR'] == 4) & (under_basic['REC_STS'] != '제적') & (under_basic['ENT_YEAR'] > 2015)]['STD_ID']
user_list = pd.DataFrame(set(user_list) & set(testset_user_list), columns=['STD_ID'])      # 2019년기준 3학년인 학생들이 2020년에는 4학년으로 올라가는 Case 발생, 겹치는 사람만 뽑아내기
df_result = pd.DataFrame(columns=['k','score','DEPT','STD_ID','total recommend','target length', 'intersection length','acc'])

# %%
count = 0
for i in trange(len(user_list)):
    user_id = user_list['STD_ID'][i]
    department_name = df_basic[df_basic['STD_ID'] == user_id]['DEPT_CD_NM'].values[0]

    k_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

    for k_ratio in k_list:
        num_recommends = 30
        intersection_subject_count, recommend_length, target_length, accuracy = recommend(df_merge, open2020_df, test2020_dataset, department_name, user_id , k_ratio, num_recommends, score=True, less_recommend=False)
        if recommend_length == False:    # Except문으로 넘어간 것
            count += 1
            df_result.loc[count] = (k_ratio, 'score1',department_name,user_id,recommend_length, target_length, intersection_subject_count, accuracy)
        else:
            count += 1
            df_result.loc[count] = (k_ratio,'score1',department_name,user_id,recommend_length, target_length, intersection_subject_count, accuracy)

# 결과 저장
df_result.to_csv('./score1_k_ratio_0-0.6.csv', encoding='utf-8-sig', index=False)    

# %%
# 결과 보기
result = pd.read_csv('./score1_k_ratio_0-0.6.csv', encoding='utf-8-sig')
df_result = result[result['total recommend'] != 'False']   # Except문으로 넘어간것 제거

# String으로 저장된 값 int형으로 변환
int_list = ['k', 'total recommend', 'target length', 'intersection length', 'acc']
for i in int_list:
    df_result[i] = df_result[i].apply(float)

total_result = df_result[int_list].groupby(['k']).mean().sort_values('acc', ascending=False)  # 총 통계량
dept_result = df_result.groupby(['DEPT', 'k']).mean().sort_values(['DEPT', 'acc'], ascending=False)  #학과별 통계량
dept_first_k = dept_result.reset_index().drop_duplicates(['DEPT'], keep='first')['k'].value_counts()  #학과별 Top1
