import pandas as pd

opened_lec_df = pd.read_csv('../Raw Data/01.Subject/01_ELECTIVES_BASIC_0630.txt', sep='|')
temp_df = pd.read_csv('./I_to_1.csv')

for i in range(len(temp_df)):
    opened_lec_df = opened_lec_df.replace({ 'COUR_NM' : {temp_df['before'][i] : temp_df['after'][i]}})


# 제일 최근 학기 필터링   # open2020.pickle로 저장
def opened_lec_filter(opened_lec_df):
    yr = max(opened_lec_df.YR)
    term_list = list(set(opened_lec_df[opened_lec_df['YR']==yr].TERM))
    if '2R' in term_list:
        term = '2R'
    else:
        term = '1R'
        
    opened_lec_df = opened_lec_df[(opened_lec_df['YR']==yr) & (opened_lec_df['TERM']==term)]
    return opened_lec_df
    
open2020_df = opened_lec_filter(opened_lec_df)


open2020_df.to_pickle('open2020.pickle')