# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 20:25:19 2018

@author: morphy
"""
import numpy as np
import pandas as pd
from keras.models import load_model
import matplotlib.pyplot as plt

#import Moidel & Data
train_X = pd.read_csv('./train_X.csv',index_col=0) 
train_Y = pd.read_csv('./train_Y.csv',index_col=0)
dev_X = pd.read_csv('./dev_X.csv',index_col=0)
dev_Y = pd.read_csv('./dev_Y.csv',index_col=0)
test1_X = pd.read_csv('./test1_X.csv',index_col=0)
test1_Y = pd.read_csv('./test1_Y.csv',index_col=0)
test2_X = pd.read_csv('./test2_X.csv',index_col=0)
test2_Y = pd.read_csv('./test2_Y.csv',index_col=0)

#数据准备
_train_X = np.asarray(train_X)
_train_X = _train_X.reshape((_train_X.shape[0], _train_X.shape[1], 1))  #(269, 92, 1)
_dev_X = np.asarray(dev_X)
_dev_X =_dev_X.reshape((_dev_X.shape[0], _dev_X.shape[1], 1))
_test1_X = np.asarray(test1_X)
_test1_X=_test1_X.reshape((_test1_X.shape[0], _test1_X.shape[1], 1))
_test2_X = np.asarray(test2_X)
_test2_X=_test2_X.reshape((_test2_X.shape[0], _test2_X.shape[1], 1))
_train_Y = np.asarray(train_Y)
_train_Y=_train_Y.reshape((_train_Y.shape[0],  1))
_dev_Y = np.asarray(dev_Y)
_dev_Y=_dev_Y.reshape((_dev_Y.shape[0], 1))
_test1_Y = np.asarray(test1_Y)
_test1_Y=_test1_Y.reshape((_test1_Y.shape[0], 1))
_test2_Y = np.asarray(test2_Y)
_test2_Y=_test2_Y.reshape((_test2_Y.shape[0], 1))

model = load_model('./Shenyang_saved_models/model_19-498.38.hdf5')
# test the model
score_train = model.evaluate(_train_X, _train_Y)
score_dev = model.evaluate(_dev_X, _dev_Y)
score_test1 = model.evaluate(_test1_X, _test1_Y)
score_test2 = model.evaluate(_test2_X, _test2_Y)
pd.DataFrame([score_train,score_dev,score_test1,score_test2],
         index=['score_train','score_dev','score_test1','score_test2'],
         columns=['loss','mae','mse']).to_csv('./Shenyang_model_evaluate_score.csv')

#画图
dev_pred=model.predict(_dev_X)
test1_pred=model.predict(_test1_X)
test2_pred=model.predict(_test2_X)
train_pred=model.predict(_train_X)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(8,8),dpi=90)
plt.figure(1)
plt.subplot(411)
plt.scatter(np.arange(50),dev_pred[:50],label='dev预测值',alpha=0.75)
plt.legend()
plt.scatter(np.arange(50),dev_Y[:50],label='dev观测值',alpha=0.75)
plt.legend()
plt.title('dev数据集 ')
plt.subplot(412)
plt.scatter(np.arange(50),test1_pred[:50],label='test1预测值',alpha=0.75)
plt.legend()
plt.scatter(np.arange(50),test1_Y[:50],label='test1观测值',alpha=0.75)
plt.legend()
plt.title('test1数据集 ')
plt.subplot(413)
plt.scatter(np.arange(50),test2_pred[:50],label='test2预测值',alpha=0.75)
plt.legend()
plt.scatter(np.arange(50),test2_Y[:50],label='test2观测值',alpha=0.75)
plt.legend()
plt.title('test2数据集 ')
plt.subplot(414)
plt.scatter(np.arange(50),train_pred[:50],label='test2预测值',alpha=0.75)
plt.legend()
plt.scatter(np.arange(50),train_Y[:50],label='test2观测值',alpha=0.75)
plt.legend()
plt.title('test2数据集 ')
