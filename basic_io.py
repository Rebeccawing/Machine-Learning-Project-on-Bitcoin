#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import datetime as dt
from functools import reduce


# In[2]:


def value_timeconvert(v):
    if isinstance(v, int):
        return v
    if isinstance(v, str) and len(v) == 8:
        return int(v)
    if isinstance(v, str) and len(v) == 10:
        assert v[4] == '-'
        assert v[7] == '-'
        return int(dt.datetime.strptime(v, '%Y-%m-%d').strftime('%Y%m%d'))
    if isinstance(v, dt.date):
        return int(v.strftime('%Y%m%d'))


# In[3]:


def int_index(data):
    date = data.index
    date = np.array([value_timeconvert(i) for i in date])
    data.index = date
    return data


# In[4]:


class DataReader:
    def __init__(self, basic_name):
        self.unprep_data = pd.read_csv(basic_name + '.csv', index_col=0)
        self.date = self.unprep_data.index
        self.date = np.array([value_timeconvert(i) for i in self.date])
        self.unprep_data = int_index(self.unprep_data)

    def get_trading_date(self):
        return self.date
    
    def combine_date(self, data):
        data = int_index(data)
        data = data[data.index.isin(self.date)]
        return data
    
    def update_columns(self, name):
        data = pd.read_csv(name + '.csv', index_col=0)
        data = int_index(data)
        self.unprep_data = self.unprep_data.join(data, how='left')
    
    def format_data(self):
        self.unprep_data = self.unprep_data.fillna(0)


# In[5]:


'''
DataSplit is a tool that can split dataset into:
train -- Jan 1, 2010 - June 30, 2018
test -- Jul 1, 2018 - Dec 31, 2018
validation -- Jan 1, 2019 - Jun 30, 2019
time features -- look back 28 days
time gap -- after 7 days

inputs:
    - data: data to be splitted
    - train_start: int 20100101 in this case
    - train_end: int 20180630 in this case
    - test_end: int 20181231 in this case
    - vld_end: int 20190630 in this case
    - gap: int 7 in this case
Class 
'''
    
class DataSplit:
    def __init__(self, x, y, train_start: int, train_end: int, vld_end: int, test_end: int, gap: int):
        self.initialX = x
        self.initialY = y
        self.X = x
        self.Y = y
        self.Y = self.Y[self.Y.index.isin(self.X.index)]
        self.train_start = train_start
        self.train_end = train_end
        self.test_end = test_end
        self.vld_end = vld_end
        self.gap = gap
        self.trading_dt = self.X.index
    
    def upd_X(self, Nlen):
        if Nlen > 1:
            df_list = [self.X]
            for i in range(1, Nlen):
                df_list.append(self.X.shift(i))
            self.X = reduce(lambda  left,right: pd.concat([left,right], axis=1), df_list)
    
    def upd_Y(self, Nlen):
        if Nlen > 1:
            df_list = [self.Y]
            for i in range(1, Nlen):
                df_list.append(self.Y.shift(-i))
            self.Y = reduce(lambda  left,right: pd.concat([left,right], axis=1), df_list)
            self.Y = np.mean(self.Y, axis=1)
    
    def get_train(self):
        self.trainX = self.X[(self.X.index >= self.train_start) & (self.X.index <= self.train_end)].values
        self.trainY = self.Y[(self.Y.index >= self.train_start) & (self.Y.index <= self.train_end)].values
        self.train_date = self.trading_dt[(self.trading_dt >= self.train_start) & (self.trading_dt <= self.train_end)]
        return self.trainX, self.trainY
    
    def get_vld(self):
        end_flag = np.where(self.trading_dt <= self.train_end)[0].max() 
        self.vld_start = self.trading_dt[end_flag + self.gap]
        self.vldX = self.X[(self.X.index >= self.vld_start) & (self.X.index <= self.vld_end)].values
        self.vldY = self.Y[(self.Y.index >= self.vld_start) & (self.Y.index <= self.vld_end)].values
        self.vld_date = self.trading_dt[(self.trading_dt >= self.vld_start) & (self.trading_dt <= self.vld_end)]
        return self.vldX, self.vldY

    def get_test(self):
        end_flag = np.where(self.trading_dt <= self.vld_end)[0].max() 
        self.test_start = self.trading_dt[end_flag + self.gap]
        self.testX = self.X[(self.X.index >= self.test_start) & (self.X.index <= self.test_end)].values
        self.testY = self.Y[(self.Y.index >= self.test_start) & (self.Y.index <= self.test_end)].values
        self.test_date = self.trading_dt[(self.trading_dt >= self.test_start) & (self.trading_dt <= self.test_end)]
        return self.testX, self.testY
    
    def get_train_date(self):
        return self.train_date
    
    def get_vld_date(self):
        return self.vld_date
    
    def get_test_date(self):
        return self.test_date

    '''
    -- prepared_data helper function -- 
    outputs:
        this will return a dictionary which key is 
            'train X', 'train Y', 'validation X', 'validation Y', 'test X', 'test Y', 'trading date'.
    inputs:
        - lookback: the date that need to look back (so to add the feature dimensions of X), must >= 0 and int. 
                    if = 0, means you just input a single day's features.
        - lookafter: the period length that need to predict (so to add dimensions of Y), must >= 0 and int.
                    if = 0, means you only want to predict a single day (eg: next day) return.
    warning: usually we need to make lookafter equal to self.gap. It could be smaller than self.gap in some situations.
    use:
        - object.data['train X'] will return a 2-d array of your train X data, etc.
    '''

    def reset(self):
        self.X = self.initialX
        self.Y = self.initialY
    
    def prepared_data(self, lookback: int, lookafter: int):
        self.reset()
        self.upd_X(lookback)
        self.upd_Y(lookafter)
        self.data = dict()
        self.data['train X'], self.data['train Y'] = self.get_train()
        self.data['validation X'], self.data['validation Y'] = self.get_vld()
        self.data['test X'], self.data['test Y'] = self.get_test()
        self.data['trading date'] = self.trading_dt
        return self.data
        
        


# In[7]:


try:    
    get_ipython().system('jupyter nbconvert --to python basic_io.ipynb')
except:
    pass


# In[ ]:




