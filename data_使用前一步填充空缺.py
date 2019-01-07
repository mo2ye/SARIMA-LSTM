# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 08:57:01 2018

@author: morphy
"""

"""
Created on Tue Nov 13 16:19:43 2018
@author: morphy
PM: PM2.5 concentration (ug/m^3) 
DEWP: Dew Point (Celsius Degree) 
TEMP: Temperature (Celsius Degree) 
HUMI: Humidity (%) 
PRES: Pressure (hPa) 
cbwd: Combined wind direction 
Iws: Cumulated wind speed (m/s) 
precipitation: hourly precipitation (mm) 
Iprec: Cumulated precipitation (mm)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']


PMdata_Beijing=pd.read_csv('./data/BeijingPM20100101_20151231.csv')
#del PMdata_Beijing['No']
print(PMdata_Beijing.columns)
#['year', 'month', 'day', 'hour', 'season', 'PM_Dongsi', 'PM_Dongsihuan',
#       'PM_Nongzhanguan', 'PM_US Post', 'DEWP', 'HUMI', 'PRES', 'TEMP', 'cbwd',
#       'Iws', 'precipitation', 'Iprec']
print(PMdata_Beijing.shape)#（52584,17）

#查看NAN数量,占比
for _ in PMdata_Beijing.columns:
    num=PMdata_Beijing[_].isnull().sum()
    print(str(_)+':',num,'占比：',num/PMdata_Beijing.shape[0])
    
'''
imputer missing values:
    丢弃2013-1-17 6:00之前的数据
    'PM_US Post'的空缺，用前三列的数据填充  
    以前一小时的值impute
'''
PMdata=PMdata_Beijing.drop(np.arange(23)) #沈阳26304   成都21279  广州19312  上海17442  北京26694
print('缺失值个数：',np.isnan(PMdata['PM_US Post']).sum(),'占比:', 306/25890)

#'PM_US Post'的空缺，用前三列的数据填充
temp=PMdata[np.isnan(PMdata['PM_US Post'])]
for i in temp.index:
    a=0
    count=0
    if not np.isnan(temp['PM_Dongsi'][i]):
        a=a+temp['PM_Dongsi'][i]
        count=count+1
    if not np.isnan(temp['PM_Dongsihuan'][i]):
        a=a+temp['PM_Dongsihuan'][i]
        count=count+1
    if not np.isnan(temp['PM_Nongzhanguan'][i]):
        a=a+temp['PM_Nongzhanguan'][i]
        count=count+1
    if count!=0:
        temp['PM_US Post'][i]=int(a/count)

#将temp中填充的值存入PMdata_Beijing
df=PMdata['PM_US Post'].copy()
df.loc[np.isnan(PMdata['PM_US Post'])] = temp['PM_US Post']

#用前一天的PM填充后一天
na_index = df[df.isnull()].index
for _ in na_index :
    df.loc[_] = df.loc[_-1]

df.to_csv('./Beijing.csv')