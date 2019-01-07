# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 09:52:08 2018
0.1取全部数据24组
0.2转置,每一段数据长度24，共1078行
@author: morphy
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#from pmdarima.arima import ARIMA, auto_arima
from pmdarima.arima import auto_arima
#from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from keras.models import Sequential,load_model
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.regularizers import l1_l2
from keras import backend as K
#from keras.optimizers import rmsprop,adam
from keras.utils.generic_utils import get_custom_objects
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error,mean_absolute_error
import os

#数据准备
#PMdata_df=pd.read_csv('./PMdate_20130117.csv')
PMdata_df=pd.read_csv('./data/Shenyang.csv')
PMdata=PMdata_df['PM_Taiyuanjie']
#以时间窗（24*4=96）为间隔建立index_liet
WINDOW=48   #时间窗的选择？24/96?谁效果好选谁
LENGTH=len(PMdata)
M = 1
index_list = []
for _ in range(WINDOW):         
    indices = []
    for k in range(_, LENGTH,WINDOW):   #共732行数据，时间窗48
        indices.append(k)
    index_list.append(indices)         #(732,48)
    
data_matrix = []
for _ in range(WINDOW):
    strided = list(PMdata.loc[index_list[_]][:int(LENGTH/WINDOW)]).copy()  #每组有732条数值,
    '''舍弃了18条的数据-------'''
    data_matrix.append(strided)
                
data_matrix = np.transpose(data_matrix)  
data_df = pd.DataFrame(data_matrix)      #(732,48)

#划分训练集，验证集
MSE=0
MAE=0
tmp=[]
train_X = []; train_Y = []
dev_X = []; dev_Y = []
test1_X = []; test1_Y = []
test2_X = []; test2_Y = []
for i in range(data_df.shape[0]):
    print(i)
    #ARIMA Modeling
    arima = auto_arima(data_df.iloc[i], error_action='warn', trace=True, n_jobs=6,
                  seasonal=True, m=M, information_criterion='aic', scoring='mse')
    predictions = list(arima.predict_in_sample())
    plt.scatter(np.arange(WINDOW)+1,data_df.iloc[i].values,s=5,label='True')
    plt.legend()
    plt.plot(predictions,label='Predict',alpha=0.75)
    plt.legend()
    plt.title('SARIMA'+str(arima.order)+str(arima.seasonal_order))
    plt.xlabel('AIC={}'.format(round(arima.aic(),4)))
    plt.show()
    print(arima.order)
    print(arima.seasonal_order)
    #残差
    residual = pd.Series(np.array(data_df.iloc[i]) - np.array(predictions))
    print('mean_absolute_error:{}'.
          format(mean_absolute_error(np.array(data_df.iloc[i]),np.array(predictions))))
    MAE=MAE+mean_absolute_error(np.array(data_df.iloc[i]),np.array(predictions))
    print('mean_squared_error:{}'.
          format(mean_squared_error(np.array(data_df.iloc[i]),np.array(predictions))))
    MSE=MSE+mean_squared_error(np.array(data_df.iloc[i]),np.array(predictions))
    tmp.append(np.array(residual))

tmp=np.array(tmp)
train_X=tmp[:,:-4]  #(732,44)(特征，时间步)
train_Y=tmp[:,-3]    #(732,)
dev_X=tmp[:,1:-3]
dev_Y=tmp[:,-2]
test1_X=tmp[:,2:-2]
test1_Y=tmp[:,-1]
test2_X=tmp[:,3:-1]
test2_Y=tmp[:,(WINDOW-1)]

pd.DataFrame(train_X).to_csv('./Shenyangtrain_X.csv') 
pd.DataFrame(train_Y).to_csv('./Shenyangtrain_Y.csv')
pd.DataFrame(dev_X).to_csv('./Shenyangdev_X.csv')
pd.DataFrame(dev_Y).to_csv('./Shenyangdev_Y.csv')
pd.DataFrame(test1_X).to_csv('./Shenyangtest1_X.csv')
pd.DataFrame(test1_Y).to_csv('./Shenyangtest1_Y.csv')
pd.DataFrame(test2_X).to_csv('./Shenyangtest2_X.csv')
pd.DataFrame(test2_Y).to_csv('./Shenyangtest2_Y.csv')


#residual-LSTM数据准备
_train_X = np.asarray(train_X).reshape((train_X.shape[0], 1, train_X.shape[1]))  #(269,1,92)
_dev_X = np.asarray(dev_X).reshape((dev_X.shape[0], 1, dev_X.shape[1]))  
_test1_X = np.asarray(test1_X).reshape((test1_X.shape[0], 1, test1_X.shape[1]))  
_test2_X = np.asarray(test2_X).reshape((test2_X.shape[0], 1, test2_X.shape[1]))  

_train_Y = np.asarray(train_Y).reshape((train_Y.shape[0], 1))
_dev_Y = np.asarray(dev_Y).reshape((dev_Y.shape[0], 1))
_test1_Y = np.asarray(test1_Y).reshape((test1_Y.shape[0], 1))
_test2_Y = np.asarray(test2_Y).reshape((test2_Y.shape[0], 1))

#参数
time_step = _train_X.shape[1]
attr = _train_X.shape[2]
save_dir = os.path.join(os.getcwd(), 'Shenyang_saved_models')
#define custom activation
class Double_Tanh(Activation):
    def __init__(self, activation, **kwargs):
        super(Double_Tanh, self).__init__(activation, **kwargs)
        self.__name__ = 'double_tanh'

def double_tanh(x):
    return (K.tanh(x) * 2)        #过滤出更多非线性
get_custom_objects().update({'double_tanh':Double_Tanh(double_tanh)})

def model_generation():
    #opt = rmsprop(lr=0.0001, decay=1e-6)
    #opt = adam(lr=0.0001,decay=1e-6)
    model = Sequential()
    model.add(LSTM(30, input_shape=(time_step,attr), dropout=0.2, return_sequences=False, 
                  kernel_regularizer=l1_l2(0.05,0.05), activity_regularizer=l1_l2(0.04,0.07)))
    #model.add(LSTM(20, input_shape=(time_step,attr), return_sequences=False, 
                  #kernel_regularizer=l1_l2(0.05,0.05), activity_regularizer=l1_l2(0.04,0.07)))
    model.add(Dense(1))
    model.add(Activation('double_tanh'))
    model.compile(loss='mean_squared_error', optimizer= 'rmsprop', metrics=['mae', 'mse'])
    #model.compile(loss='mae', optimizer='adam,rmsprop')#opt, 
    print(model.summary())
    return model

model = model_generation()

#reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=4, mode='min')
filepath="model_{epoch:02d}-{val_loss:.2f}.hdf5" #'model_best.hdf5'
checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath), monitor='val_loss',
                             verbose=1, save_best_only=False, mode='min')
earlystop=EarlyStopping(monitor='loss', patience=10, verbose=0, mode='min')
history = model.fit(_train_X, _train_Y, batch_size=32, epochs=160, 
                    shuffle=True,
                    callbacks=[checkpoint], #earlystop],
                    validation_split=0.1)
                    #validation_data=(x_test, y_test), reduce_lr
                    
pd.DataFrame(history.history).to_csv('Shenyang_model_history.csv')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(8,8),dpi=90)
plt.figure(1)
plt.subplot(411)
plt.plot(history.history['loss'][:100],label='训练集MSE',alpha=0.75)
plt.legend()
plt.ylabel('Mean Squared Error')
plt.subplot(412)
plt.plot(history.history['val_loss'][:100],label='验证集MSE',alpha=0.75)
plt.legend()
plt.ylabel('Mean Squared Error')
plt.subplot(413)
plt.plot(history.history['mean_absolute_error'][:100],label='训练集MAE',alpha=0.75)
plt.legend()
plt.ylabel('Mean Absolute Error')
plt.subplot(414)
plt.plot(history.history['val_mean_absolute_error'][:100],label='验证集MAE',alpha=0.75)
plt.legend()
plt.xlabel('epochs')
plt.ylabel('Mean Absolute Error')

plt.show()

print('Complet Training!')