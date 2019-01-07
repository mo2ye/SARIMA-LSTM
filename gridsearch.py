# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 20:25:19 2018

@author: morphy
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential,load_model
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.regularizers import l1_l2


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

SEQUENCE=_train_X.shape[1]

def model_generation(num_neurons = 20):
    #opt = rmsprop(lr=0.0001, decay=1e-6)
    #opt = adam(lr=0.0001,decay=1e-6)
    model = Sequential()
    model.add(LSTM(num_neurons, input_shape=(SEQUENCE,1), dropout=0.2, return_sequences=False, 
                  kernel_regularizer=l1_l2(0.1,0.1), activity_regularizer=l1_l2(0.04,0.07)))
    #model.add(LSTM(10, return_sequences=False))
    #model.add(LSTM(50))
    model.add(Dense(1))
    model.add(Activation('tanh'))
    model.compile(loss='mean_squared_error', optimizer= 'rmsprop', metrics=['mae', 'mse'])
    #model.compile(loss='mae', optimizer='adam,rmsprop')#opt, 
    #,
    print(model.summary())
    return model

sk_params={'batch_size':32, 'epochs':150, 
        }
#模型
model = KerasRegressor(build_fn=model_generation, **sk_params)

# 设置参数候选值
epochs = [20,50,100]
optimizer = ['rmsprop', 'adam', 'adagrad']
dropout = [0.2, 0.5]
num_neurons = [5, 10, 15, 20]
kernel_regularizer = [l1_l2(0.07,0.07),l1_l2(0.1,0.1),l1_l2(0.05,0.05),l1_l2(0.01,0.01)]
activity_regularizer = [l1_l2(0.04,0.07),l1_l2(0.05,0.1),l1_l2(0.1,0.05),l1_l2(0.05,0.05)]
# 创建GridSearchCV，并训练
param_grid = dict(num_neurons = num_neurons,
                  optimizer = optimizer,
                  kernel_regularizer = kernel_regularizer,
                  activity_regularizer = activity_regularizer)
grid = GridSearchCV(estimator=model, param_grid=param_grid, 
                    n_jobs=8, cv=3, verbose=1)
grid_result = grid.fit(_train_X, _train_Y, validation_split=0.1)

print('Best: {} using {}'.format(grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# =============================================================================
# model = load_model('./saved_models/model_15-775.02.hdf5')
# # test the model
# score_train = model.evaluate(_train_X, _train_Y)
# score_dev = model.evaluate(_dev_X, _dev_Y)
# score_test1 = model.evaluate(_test1_X, _test1_Y)
# score_test2 = model.evaluate(_test2_X, _test2_Y)
# pd.DataFrame([score_train,score_dev,score_test1,score_test2],
#          index=['score_train','score_dev','score_test1','score_test2'],
#          columns=['loss','mae','mse']).to_csv('./model_evaluate_score.csv')
# =============================================================================
