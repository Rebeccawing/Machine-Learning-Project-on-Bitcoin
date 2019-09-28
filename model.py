#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import datetime as dt
from functools import reduce
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, GRU
import keras.utils
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[2]:


from basic_io import value_timeconvert, int_index
from basic_io import DataReader, DataSplit


# In[ ]:


def mae(truth, predict):
    assert truth.shape == predict.shape
    return abs(truth - predict).reshape(-1).sum() / (truth.shape[0])


# In[ ]:


def rmse(truth, predict):
    assert truth.shape == predict.shape
    return np.sqrt(((truth - predict)**2).reshape(-1).sum() / (truth.shape[0]))


# In[3]:


class Model:
    def __init__(self, data_model, train_pred=0, valid_pred=0, test_pred=0, train_tru=0, valid_tru=0, test_tru=0, train_par=0, 
                train_x=0, valid_x=0, test_x=0):
        self.data_model = data_model
        self.train_pred = 0
        self.valid_pred = 0
        self.test_pred = 0
        self.train_tru = 0
        self.valid_tru = 0
        self.test_tru = 0
        self.train_par = 0
        self.train_x = 0
        self.valid_x = 0
        self.test_x = 0
    
    def get_train_pred(self):
        return self.train_pred
    
    def get_valid_pred(self):
        return self.valid_pred
    
    def get_test_pred(self):
        return self.test_pred
    
    def norm_data(self):
        x_mean = np.nanmean(self.train_x, axis=0)
        x_std = np.nanstd(self.train_x, axis=0)
        self.train_x = (self.train_x - x_mean) / x_std
        self.valid_x = (self.valid_x - x_mean) / x_std
        self.test_x = (self.test_x - x_mean) / x_std
#         y_mean = np.nanmean(self.train_tru, axis=0)
#         y_std = np.nanstd(self.train_tru, axis=0)
#         self.train_tru = (self.train_tru - y_mean) / y_std
#         self.valid_tru = (self.valid_tru - y_mean) / y_std
    
    def eval_train(self):
        res = dict()
        use_mae = mae(self.train_tru, self.train_pred)
        use_rmse = rmse(self.train_tru, self.train_pred)
        res['MAE'] = use_mae
        res['RMSE'] = use_rmse
        return res
    
    def eval_valid(self):
        res = dict()
        use_mae = mae(self.valid_tru, self.valid_pred)
        use_rmse = rmse(self.valid_tru, self.valid_pred)
        res['MAE'] = use_mae
        res['RMSE'] = use_rmse
        print('#############################evaluation mae and rmse#######################################')
        print(res)

    
    def eval_test(self):
        res = dict()
        use_mae = mae(self.test_tru, self.test_pred)
        use_rmse = rmse(self.test_tru, self.test_pred)
        res['MAE'] = use_mae
        res['RMSE'] = use_rmse
        print('#############################evaluation mae and rmse#######################################')
        print(res)
    
    def plot_y(self, a, b, name):
        plt.figure()
        plt.plot(a, 'b', label='Prediction')
        plt.plot(b, 'g', label='True value')
        plt.title(name)
        plt.legend()
        plt.show()
    
    def plot_train(self):
        self.plot_y(self.train_pred, self.train_tru, 'Train set')
    
    def plot_valid(self):
        self.plot_y(self.valid_pred, self.valid_tru, 'Validation set')
    
    def plot_test(self):
        self.plot_y(self.test_pred, self.test_tru, 'Test set')
        
    
    def reset(self):
        self.data_model = 0
        self.train_pred = 0
        self.valid_pred = 0
        self.test_pred = 0
        self.train_tru = 0
        self.valid_tru = 0
        self.test_tru = 0
        self.train_par = 0
        self.train_x = 0
        self.valid_x = 0
        self.test_x = 0


# In[4]:


class BenchPredict(Model):
    def __init__(self, data_model, train_pred=0, valid_pred=0, test_pred=0, train_tru=0, valid_tru=0, test_tru=0, train_par=0, lb=0, la=0):
        Model.__init__(self, data_model, train_pred, valid_pred, test_pred, train_tru, valid_tru, test_tru, train_par)  
        self.la = la
        self.lb = lb
        self.data = self.data_model.prepared_data(self.lb, self.la)
        self.reset()
        self.valid_x = self.data['validation X']
        self.train_x = self.data['train X']
        self.valid_tru = self.data['validation Y']
        self.train_tru = self.data['train Y']
        self.norm_data()

    # Get the benchmark prediction (use average of lookback period) results for validation set
    def get_res1(self):
        nlen = int(self.valid_x.shape[1] / self.lb)
        if self.la == 1:
            self.valid_pred = np.mean(self.valid_x[:, 18::nlen], axis=1)
        else:
            self.valid_pred = np.tile(np.mean(self.valid_x[:, 18::nlen], axis=1), (self.la,1)).T
        result1 = self.eval_valid()
        print(self.valid_pred.shape[0])
        print(len(self.valid_tru))
        self.plot_valid()
        return result1

    
    # Get the benchmark prediction (use last value of lookback period) results for validation set
    def get_res2(self):
        if self.la == 1:
            self.valid_pred = self.valid_x[:, 18]
        else:
            self.valid_pred = np.tile(self.valid_x[:, 18], (self.la,1)).T
        result2 = self.eval_valid()
        return result2


# In[5]:


class NN(Model):
    def __init__(self, data_model, train_pred=0, valid_pred=0, test_pred=0, train_tru=0, valid_tru=0, test_tru=0, train_par=0, lb=0, la=0):
        Model.__init__(self, data_model, train_pred, valid_pred, test_pred, train_tru, valid_tru, test_tru, train_par)  
        self.la = la
        self.lb = lb
        self.data = self.data_model.prepared_data(self.lb, self.la)
        self.init_len_x = self.data_model.initialX.shape[1]
        self.reset()
        self.valid_x = self.data['validation X']
        self.train_x = self.data['train X']
        self.valid_tru = self.data['validation Y']
        self.train_tru = self.data['train Y']
        self.test_tru = self.data['test Y']
        self.test_x = self.data['test X']
        self.norm_data()
        
    def train_nn(self):
        m = Sequential()
        m.add(Dense(150, input_dim=self.lb*self.init_len_x, init='uniform', activation='relu'))
        m.add(Dense(75, init='uniform', activation='relu'))
        m.add(Dense(self.la, init='uniform', activation='linear'))
        m.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae'])
        history = m.fit(self.train_x, self.train_tru, epochs=100, batch_size=200, validation_data=(self.valid_x, self.valid_tru))
        self.train_par = m
        self.loss = history.history['loss']
        self.val_loss = history.history['val_loss']
        self.scores = m.evaluate(self.valid_x, self.valid_tru)
        self.valid_pred = m.predict(self.valid_x)
        return self.scores
    
    def get_res(self):
        epochs = range(len(self.loss))
        plt.figure()
        plt.plot(epochs, self.loss, 'b', label='Training loss')
        plt.plot(epochs, self.val_loss, 'g', label='Validation loss')
        plt.title('Simple Two Layer NN')
        plt.legend()
        plt.show()
        if self.la == 1:
            self.valid_tru = np.tile(self.valid_tru, (1,1)).T
        self.plot_valid()
        return self.eval_valid()
    
    def pred_test(self):
        if self.la == 1:
            self.test_tru = np.tile(self.test_tru, (1,1)).T
        self.test_pred = self.train_par.predict(self.test_x)
        self.plot_test()
        return self.eval_test()    


# In[6]:


class RNN(Model):
    def __init__(self, data_model, train_pred=0, valid_pred=0, test_pred=0, train_tru=0, valid_tru=0, test_tru=0, train_par=0, lb=0, la=0):
        Model.__init__(self, data_model, train_pred, valid_pred, test_pred, train_tru, valid_tru, test_tru, train_par)  
        self.la = la
        self.lb = lb
        self.data = self.data_model.prepared_data(self.lb, self.la)
        self.init_len_x = self.data_model.initialX.shape[1]
        self.reset()
        self.valid_x = self.data['validation X']
        self.train_x = self.data['train X']
        self.valid_tru = self.data['validation Y']
        self.train_tru = self.data['train Y']
        self.valid_tru_init = self.data['validation Y']
        self.train_tru_init = self.data['train Y']
        self.test_tru_init = self.data['test Y']
        self.test_tru = self.data['test Y']
        self.test_x = self.data['test X']
        self.norm_data()
        self.train_x = self.train_x.reshape(self.train_x.shape[0], self.lb, self.init_len_x)
        self.valid_x = self.valid_x.reshape(self.valid_x.shape[0], self.lb, self.init_len_x)
        self.test_x = self.test_x.reshape(self.test_x.shape[0], self.lb, self.init_len_x)
        self.train_par = dict()
    
    def cons_lstm(self):
        lstm_nodes = 5
        m = Sequential()
        m.add(LSTM(units=lstm_nodes, input_shape=(self.lb, self.init_len_x)))
        m.add(Dense(self.la, activation='linear'))
        print(m.summary())
        m.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mse'])
        history = m.fit(self.train_x, self.train_tru, validation_split=0.1, epochs=50, batch_size=64, validation_data=(self.valid_x, self.valid_tru))
        self.loss = history.history['loss']
        self.val_loss = history.history['val_loss']
        self.scores = m.evaluate(self.valid_x, self.valid_tru)
        self.valid_pred = m.predict(self.valid_x)
        self.train_par['lstm'] = m
    
    def plot_loss(self, name):
        epochs = range(len(self.loss))
        plt.figure()
        plt.plot(epochs, self.loss, 'b', label='Training loss')
        plt.plot(epochs, self.val_loss, 'g', label='Validation loss')
        plt.title(name)
        plt.legend()
        plt.show()
        if self.la == 1:
            self.valid_tru = np.tile(self.valid_tru, (1,1)).T
        self.loss = 0
        self.val_loss = 0
        res = self.eval_valid() 
        self.valid_tru = self.valid_tru_init
        return res   
    
    def cons_gru(self):
        gru_nodes = 10
        m = Sequential()
        m.add(GRU(units=gru_nodes, input_shape=(self.lb, self.init_len_x)))
        m.add(Dense(self.la, activation='linear'))
        print(m.summary())
        m.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mse'])
        history = m.fit(self.train_x, self.train_tru, validation_split=0.1, epochs=50, batch_size=100, validation_data=(self.valid_x, self.valid_tru))
        self.loss = history.history['loss']
        self.val_loss = history.history['val_loss']
        self.scores = m.evaluate(self.valid_x, self.valid_tru)
        self.valid_pred = m.predict(self.valid_x)
        self.train_par['gru'] = m
    
    def cons_gru_drop(self):
        gru_nodes = 10
        m = Sequential()
        m.add(GRU(units=gru_nodes, input_shape=(self.lb, self.init_len_x), recurrent_dropout=0.2))
        m.add(Dense(self.la, activation='linear'))
        print(m.summary())
        m.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mse'])
        history = m.fit(self.train_x, self.train_tru, validation_split=0.1, epochs=50, batch_size=100, validation_data=(self.valid_x, self.valid_tru))
        self.loss = history.history['loss']
        self.val_loss = history.history['val_loss']
        self.scores = m.evaluate(self.valid_x, self.valid_tru)
        self.valid_pred = m.predict(self.valid_x)  
        self.train_par['gru_dropout'] = m
    
    def cons_gru_drop_add(self):
        gru_nodes1 = 10
        gru_nodes2 = 20
        m = Sequential()
        m.add(GRU(units=gru_nodes1, input_shape=(self.lb, self.init_len_x), recurrent_dropout=0.2, return_sequences=True))
        m.add(GRU(units=gru_nodes2, recurrent_dropout=0.2, dropout=0.4))
        m.add(Dense(self.la, activation='linear'))
        print(m.summary())
        m.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mse'])
        history = m.fit(self.train_x, self.train_tru, validation_split=0.1, epochs=50, batch_size=100, validation_data=(self.valid_x, self.valid_tru))
        self.loss = history.history['loss']
        self.val_loss = history.history['val_loss']
        self.scores = m.evaluate(self.valid_x, self.valid_tru)
        self.train_pred = m.predict(self.train_x)
        self.valid_pred = m.predict(self.valid_x)  
        self.train_par['gru_dropout_add'] = m
        self.test_pred = self.train_par['gru_dropout_add'].predict(self.test_x)
    
    def train_lstm(self):
        self.cons_lstm()
        self.plot_loss('RNN with LSTM')
    
    def train_gru(self):
        self.cons_gru()
        self.plot_loss('RNN with GRU')
    
    def train_gru_drop(self):
        self.cons_gru_drop()
        self.plot_loss('RNN with GRU and recurrent dropout')
    
    def train_gru_drop_add(self):
        self.cons_gru_drop_add()
        self.plot_loss('RNN with GRU, recurrent dropout and another dropout added in')
    
    def test_lstm(self):
        if self.la == 1:
            self.test_tru = np.tile(self.test_tru, (1,1)).T
        self.test_pred = self.train_par['lstm'].predict(self.test_x)
        self.plot_test()
        self.eval_test()
        self.test_tru = self.test_tru_init
    
    def test_gru(self):
        if self.la == 1:
            self.test_tru = np.tile(self.test_tru, (1,1)).T
        self.test_pred = self.train_par['gru'].predict(self.test_x)
        self.plot_test()
        self.eval_test()
        self.test_tru = self.test_tru_init
    
    def test_gru_dropout(self):
        if self.la == 1:
            self.test_tru = np.tile(self.test_tru, (1,1)).T
        self.test_pred = self.train_par['gru_dropout'].predict(self.test_x)
        self.plot_test()
        self.eval_test()
        self.test_tru = self.test_tru_init

    def test_gru_dropout_add(self):
        if self.la == 1:
            self.test_tru = np.tile(self.test_tru, (1,1)).T
        self.test_pred = self.train_par['gru_dropout_add'].predict(self.test_x)
        self.plot_test()
        self.eval_test()
        self.test_tru = self.test_tru_init
    
        


# In[7]:


class Arma:
    def __init__(self, data_model):
        self.data = data_model.prepared_data(28, 1)
        self.valid_tru = self.data['validation Y']
        self.train_tru = self.data['train Y']
        self.test_tru = self.data['test Y']
    
    def test_stationary(self):
        res = adfuller(self.train_tru)
        print('ADF Statistic: %f' % res[0])
        print('p-value: %f' % res[1])
        if res[1] <= 0.01:
            print("stationary!")
        else:
            print("not stationary!")
    
    def plot_stats(self):
        lag_acf = acf(self.train_tru, nlags=20)
        lag_pacf = pacf(self.train_tru, nlags=20)
        fig, axes = plt.subplots(1,2, figsize=(20,5))
        plot_acf(self.train_tru, lags=50, ax=axes[0])
        plot_pacf(self.train_tru, lags=50, ax=axes[1])
        plt.title('ACF and PACF')
        plt.show()
    
    def order_select(self, arr, pmax,qmax): 
        aic = 0
        bestp = 0
        bestq = 0
        for p in range(pmax):
            for q in range(qmax):
                model = ARMA(arr, order=(p, q))
                results = model.fit(trend='c')
                print(p,q,results.aic)
                if results.aic < aic:
                    bestp = p
                    bestq = q
                    aic = results.aic
        return bestp,bestq
 
    def cons_arma(self):
        bestp, bestq = self.order_select(self.train_tru,5,2)
        print(bestp, bestq)
        model = ARMA(self.train_tru, order=(bestp,bestq))
        model_fit = model.fit(disp=0)
        model_fit.summary()
        self.train_pred = model_fit.fittedvalues
        self.par = model_fit.params
        self.valid_pred = self.par[0] + self.par[1] * np.r_[np.zeros(1), self.valid_tru][:-1] + self.par[2] * np.r_[np.zeros(2), self.valid_tru][:-2]
        + self.par[3] * np.r_[np.zeros(3), self.valid_tru][:-3] + self.par[4] * np.r_[np.zeros(4), self.valid_tru][:-4]
        self.test_pred = self.par[0] + self.par[1] * np.r_[np.zeros(1), self.test_tru][:-1] + self.par[2] * np.r_[np.zeros(2), self.test_tru][:-2]
        + self.par[3] * np.r_[np.zeros(3), self.test_tru][:-3] + self.par[4] * np.r_[np.zeros(4), self.test_tru][:-4]

    def eval(self, a1, a2):
        res = dict()
        use_mae = mae(a1, a2)
        use_rmse = rmse(a1, a2)
        res['MAE'] = use_mae
        res['RMSE'] = use_rmse
        print('#############################evaluation mae and rmse#######################################')
        print(res)
        
    def eval_train(self):
        self.cons_arma()
        self.eval(self.train_tru, self.train_pred)
    
    def eval_val(self):
        self.eval(self.valid_tru, self.valid_pred)
    
    def eval_test(self):
        self.eval(self.test_tru, self.test_pred)
        
    
    
        


# In[8]:


def decide_weight(rnn_pred, arma_pred, tru):
    w = pd.Series(index = np.linspace(0, 1, 11))
    for i in np.linspace(0, 1, 11):
        pred = rnn_pred * i + arma_pred * (1 - i)
        print('RNN weight: ' + str(i) + ' and ARMA weight: ' + str(1-i)+' -- MAE: '+str(mae(tru, pred)))
        w.loc[i] = mae(tru, pred)
    return w[w==w.max()].index[0]


# In[9]:


def combine_model(rnn_model, arma_model):
    m1 = rnn_model
    m1.cons_gru_drop_add()
    m2 = arma_model
    m2.cons_arma()
    rnn_train_pred = m1.train_pred
    arma_train_pred = m2.train_pred
    arma_val_pred = m2.valid_pred
    arma_test_pred = m2.test_pred
    df_list1 = [arma_train_pred]
    df_list2 = [arma_val_pred]
    df_list3 = [arma_test_pred]
    for i in range(1, 7):
        df_list1.append(np.r_[arma_train_pred, np.zeros(i)][i:])
        df_list2.append(np.r_[arma_val_pred, np.zeros(i)][i:])
        df_list3.append(np.r_[arma_test_pred, np.zeros(i)][i:])
    arma_train_pred= np.tile(np.mean(reduce(lambda  left,right: np.c_[left,right], df_list1), axis=1), (1,1)).T
    arma_val_pred = np.tile(np.mean(reduce(lambda  left,right: np.c_[left,right], df_list2), axis=1), (1,1)).T
    arma_test_pred= np.tile(np.mean(reduce(lambda  left,right: np.c_[left,right], df_list3), axis=1), (1,1)).T                                  
    train_tru = np.tile(m1.train_tru, (1,1)).T
    valid_tru = np.tile(m1.valid_tru, (1,1)).T
    test_tru = np.tile(m1.test_tru, (1,1)).T
    w = decide_weight(rnn_train_pred, arma_train_pred, train_tru)
    train_pred = rnn_train_pred * w + arma_train_pred * (1 - w)
    valid_pred = m1.valid_pred * w + arma_val_pred * (1 - w)  
    test_pred = m1.test_pred * w + arma_test_pred * (1 - w) 
    res = dict()
    res['train'] = train_pred
    res['valid'] = valid_pred
    res['test'] = test_pred
    res['Train MAE'] = mae(train_tru, train_pred)
    res['Valid MAE'] = mae(valid_tru, valid_pred)
    res['Test MAE'] = mae(test_tru, test_pred)
    res['Train RMSE'] = rmse(train_tru, train_pred)
    res['Valid RMSE'] = rmse(valid_tru, valid_pred)
    res['Test RMSE'] = rmse(test_tru, test_pred)   
    return res


# In[11]:


try:    
    get_ipython().system('jupyter nbconvert --to python model.ipynb')
except:
    pass


# In[ ]:




