import pandas as pd
import numpy as np
import os

g_table = pd.read_pickle('./cour_opt_table.pickle')
lis_val = [np.nan, np.nan, np.nan, 3 , 1, np.nan, np.nan, 1, 
np.nan, np.nan, np.nan, np.nan, 1, np.nan, np.nan, np.nan, np.nan, 0, 0, 0, np.nan, np.nan]
lis_key = g_table.columns[1:]


dic = {}
for i in range(len(lis_val)):
    dic[lis_key[i]] = lis_val[i]


def filter(lis):
    dic = {}
    for i in range(len(lis_key)):
        dic[lis_key[i]] = lis[i]
    for i in lis_key[np.where(np.isnan(list(dic.values())))]:
        del dic[i]
    
    a = pd.DataFrame(dic.items()).T
    a.columns = a.iloc[0,:]
    a.drop(0, axis=0, inplace=True)
    return (pd.merge(g_table, a, on=a.columns.tolist())[['COUR_CD','PROF_NM']])

filter(lis_val)