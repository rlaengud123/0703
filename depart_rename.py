# %%
import os
import pandas as pd

# 변경하고자 하는 DataFrame Path
load_path = '../Raw Data'
abs_path = os.path.abspath('C:/Users/korea/Desktop/교양과목추천/0612')

subject_basic = pd.read_table(os.path.join(load_path, '01.Subject/01_ELECTIVES_BASIC.txt'), sep='|', error_bad_lines=False, dtype='str')
under_basic = pd.read_table(os.path.join(load_path, '03.User/03_STD_ENR_BASIC.txt'), sep='|', error_bad_lines=False, dtype='str')
graduate_basic = pd.read_table(os.path.join(load_path, '03.User/03_STD_GRD_BASIC.txt'), sep='|', error_bad_lines=False, dtype='str')

rename_file = pd.read_excel(os.path.join(abs_path, './dept_rename.xlsx'), dtype='str')    # 수정하기 전과 수정할 학과명(코드)를 포함하는 Excel 파일

# 공백제거
df_list = [under_basic, graduate_basic, subject_basic]

for df in df_list:
    strip_columns = list(df.select_dtypes('object').columns)
    for column in strip_columns:
        df[column] = df[column].str.strip()

# %%
def make_mapping_dict(rename_file):
    '''
    rename_file: 수정하기 전과 수정할 학과명(코드)를 포함하는 Excel 파일
    '''
    before_dept = list(rename_file['DEPT_CD_before'])
    now_dept = list(rename_file['DEPT_CD_now'])
    before_dept_nm = list(rename_file['DEPT_CD_NM_before'])
    now_dept_nm = list(rename_file['DEPT_CD_NM_now'])

    dept_mapping_dict = dict(zip(before_dept, now_dept))  # 학과코드
    dept_name_mapping_dict = dict(zip(before_dept_nm, now_dept_nm))  # 학과이름

    return dept_mapping_dict, dept_name_mapping_dict


def replace_name(column, mapping_dict):
    '''
    column: DataFrame에서 변경할 학과명에 해당하는 Column
    mapping_dict: 수정 전과 수정 후의 Mapping Dictionary`
    '''
    for key,value in mapping_dict.items():
        column = column.replace(str(key), str(value))

    return column


def depart_rename(rename_file, df, depart_code_column, depart_name_column):
    '''
    rename_file: 수정하기 전과 수정할 학과명(코드)를 포함하는 Excel 파일
    df: 학과명을 수정해야하는 DataFrame
    depart_code_column: 해당 DataFrame에서 학과코드에 해당하는 Column명
    depart_name_column: 해당 DataFrame에서 학과명에 해당하는 Column명
    '''
    dept_mapping_dict, dept_name_mapping_dict = make_mapping_dict(rename_file)

    df[depart_code_column] = replace_name(df[depart_code_column], dept_mapping_dict)
    df[depart_name_column] = replace_name(df[depart_name_column], dept_name_mapping_dict)

    return df


# 저장 경로
save_path = os.path.abspath('C:/Users/korea/Desktop/교양과목추천/Raw Data')

if __name__ == "__main__":
    subject_basic = depart_rename(rename_file, subject_basic, 'DEPT_CD', 'DEPT_CD_1')
    under_basic = depart_rename(rename_file, under_basic, 'DEPT_CD', 'DEPT_CD_NM')
    graduate_basic = depart_rename(rename_file, graduate_basic, 'DEPT_CD', 'DEPT_CD_NM')

    if os.path.isdir(save_path + '/01.Subject') == False:
        os.mkdir(save_path + '/01.Subject/')
    elif os.path.isdir(save_path + '/03.User') == False:
        os.mkdir(save_path + '/03.User/')

    subject_basic.to_csv(os.path.join(save_path, '01.Subject\\01_ELECTIVES_BASIC.txt'), index=False, sep='|')
    under_basic.to_csv(os.path.join(save_path, '03.User\\03_STD_ENR_BASIC.txt'), index=False, sep='|')
    graduate_basic.to_csv(os.path.join(save_path, '03.User\\03_STD_GRD_BASIC.txt'), index=False ,sep='|')

# %%
