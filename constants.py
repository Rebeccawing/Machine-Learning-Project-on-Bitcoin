#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from functools import reduce


# In[2]:


from basic_io import value_timeconvert, int_index
from basic_io import DataReader, DataSplit


# In[3]:


from model import mae, rmse, decide_weight, combine_model
from model import Model, BenchPredict, NN, RNN, Arma


# In[4]:


#######################################################################################
####  Data Processing  ################################################################
#######################################################################################

'''
Constants: 
           input  - feature list: what data you want to read
           output - daily return list: y
                  - trading date list: date
                  - datasplit dictionary: see helper function in basic_io.py
'''

instance = DataReader('price')
feature_list = ['price-volatility', 'daily-transactions', 'chain-value-density',
                'hash-rate', 'inflation', 'LTC-USD', 'market-cap', 'miner-revenue',
                'transaction-amount', 'transaction-fees', 'transaction-size', 'transaction-value']
for feature in feature_list:    
    instance.update_columns(feature)
instance.format_data()
trading_date_list = instance.get_trading_date()
instance.unprep_data['return'] =  instance.unprep_data['Bitcoin Core (BTC) Price'].pct_change().replace(np.inf,0).fillna(0).values
daily_return = instance.unprep_data['return']
df_list = [daily_return]
for i in range(1, 7):
    df_list.append(daily_return.shift(-i))
    daily_return= reduce(lambda  left,right: pd.concat([left,right], axis=1), df_list)
daily_return = np.mean(daily_return, axis=1).fillna(0)
daily_return = daily_return.fillna(0)
data = DataSplit(instance.unprep_data, daily_return, 20100101, 20180630, 20181231, 20190630, 7)


# In[5]:


#######################################################################################
####  Model Framework  ################################################################
#######################################################################################
tot_mod = Model(data) # -- PARENT CLASS of Models
benchmark_model = BenchPredict(data_model=data, lb=28, la=1) # -- BENCHMARK CLASS
NN_model = NN(data_model=data, lb=28, la=1) # -- NEURAL NETWORK CLASS
RNN_model = RNN(data_model=data, lb=28, la=1) # -- RNN CLASS
ARMA_data = DataSplit(instance.unprep_data, instance.unprep_data['return'], 20100101, 20180630, 20181231, 20190630, 7) # -- ARMA DATA PROCESS
ARMA_model = Arma(data_model=ARMA_data) # -- ARMA CLASS
Embed_model = combine_model(RNN_model, ARMA_model)


# In[9]:


try:    
    get_ipython().system('jupyter nbconvert --to python constants.ipynb')
except:
    pass


# In[ ]:




