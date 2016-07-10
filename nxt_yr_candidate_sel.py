# -*- coding: utf-8 -*-
"""
Created on Thu May 12 15:09:47 2016

@author: jli9
"""

import csv
import pandas as pd
#import functools as ft
#import tushare as ts
import numpy as np

# recover data
with open('stock_p_change_2004-2-2016.csv') as f:
    reader = csv.reader(f)
    i = 0
    for row in reader:
        if i == 0: 
            df_p_change_list = pd.DataFrame(columns = row)
            i = 1
        else:
            df_p_change_list.loc[row[0],:] = row
            
#df_p_change_list = df_p_change_list.convert_objects(convert_numeric=True)
df_p_change_list = df_p_change_list.apply(lambda x: pd.to_numeric(x, errors='coerce'))

da_start_yr = 2006
da_end_yr = 2015
df_nxt_yr_perf = pd.DataFrame(index=np.arange(da_start_yr,da_end_yr))
if 0:
    prob_beyond_2sigma = 0.022
    prob_minus_2sigma_to_1sigma = 0.136
    prob_1sigma_to_mean = 0.341 
else:
    prob_beyond_2sigma = 1
    prob_minus_2sigma_to_1sigma = 1
    prob_1sigma_to_mean = 1    
for y in np.arange(da_start_yr,da_end_yr):
    nxt_yr = '%d_total' %(y+1)
    curr_q = '%d-10' %(y)
    t_df = df_p_change_list[[curr_q,nxt_yr]] 
    t_df=t_df[t_df[curr_q]!=-np.inf]        # filter any new IPO in next year
    [q4_mean, q4_std]=t_df[t_df[curr_q]!=-np.inf][curr_q].describe()[['mean','std']]
    [yr_mean, yr_std]=t_df[t_df[nxt_yr]!=-np.inf][nxt_yr].describe()[['mean','std']]

    exam_left = q4_mean+q4_std
    exam_right = q4_mean+2*q4_std

    df_nxt_yr_perf.loc[y,'stock_in_total'] = len(t_df[t_df[curr_q] != -np.inf])
    #df_nxt_yr_perf.loc[y,'top_perf_stocks'] = len(t_df[t_df[curr_q] > (q4_mean+2*q4_std)])
    #nxt_yr_perf = t_df[t_df[curr_q] > (q4_mean+2*q4_std)][nxt_yr]
    df_nxt_yr_perf.loc[y,'top_perf_stocks'] = len(t_df[t_df[curr_q] >  exam_left][t_df[curr_q] < exam_right])
    nxt_yr_perf = t_df[t_df[curr_q] > exam_left][t_df[curr_q] < exam_right][nxt_yr]

    df_nxt_yr_perf.loc[y,'less_than_-2sigma']=nxt_yr_perf[nxt_yr_perf<yr_mean-2*yr_std].count()
    df_nxt_yr_perf.loc[y,'-2sigma_2_-sigma']=nxt_yr_perf[nxt_yr_perf<yr_mean-yr_std][nxt_yr_perf>=yr_mean-2*yr_std].count()
    df_nxt_yr_perf.loc[y,'-sigma_2_mean']=nxt_yr_perf[nxt_yr_perf<yr_mean][nxt_yr_perf>=yr_mean-yr_std].count()
    df_nxt_yr_perf.loc[y,'mean_2_sigma']=nxt_yr_perf[nxt_yr_perf<yr_mean+yr_std][nxt_yr_perf>=yr_mean].count()
    df_nxt_yr_perf.loc[y,'sigma_2_2sigma']=nxt_yr_perf[nxt_yr_perf<yr_mean+2*yr_std][nxt_yr_perf>=yr_mean+yr_std].count()
    df_nxt_yr_perf.loc[y,'larger_than_2sigma']=nxt_yr_perf[nxt_yr_perf>=yr_mean+2*yr_std].count()

    df_nxt_yr_perf['less_than_-2sigma_ratio']=(df_nxt_yr_perf['less_than_-2sigma']/df_nxt_yr_perf['top_perf_stocks'])/prob_beyond_2sigma
    df_nxt_yr_perf['-2sigma_2_-sigma_ratio']=(df_nxt_yr_perf['-2sigma_2_-sigma']/df_nxt_yr_perf['top_perf_stocks'])/prob_minus_2sigma_to_1sigma
    df_nxt_yr_perf['-sigma_2_mean_ratio']=(df_nxt_yr_perf['-sigma_2_mean']/df_nxt_yr_perf['top_perf_stocks'])/prob_1sigma_to_mean
    df_nxt_yr_perf['mean_2_sigma_ratio']=(df_nxt_yr_perf['mean_2_sigma']/df_nxt_yr_perf['top_perf_stocks'])/prob_1sigma_to_mean
    df_nxt_yr_perf['sigma_2_2sigma_ratio']=(df_nxt_yr_perf['sigma_2_2sigma']/df_nxt_yr_perf['top_perf_stocks'])/prob_minus_2sigma_to_1sigma
    df_nxt_yr_perf['larger_than_2sigma_ratio']=(df_nxt_yr_perf['larger_than_2sigma']/df_nxt_yr_perf['top_perf_stocks'])/prob_beyond_2sigma
    
df_nxt_yr_perf[['less_than_-2sigma_ratio','-2sigma_2_-sigma_ratio', '-sigma_2_mean_ratio', 'mean_2_sigma_ratio',
       'sigma_2_2sigma_ratio', 'larger_than_2sigma_ratio']].plot(kind='barh',figsize=(16,8),grid=True,legend='reverse')

