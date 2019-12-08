# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 16:57:13 2019

@author: Aadarsh Srivastava
"""
import numpy as np
from DataPreprocessing import read_mat,data_reshape
from SleepModel import model_single,model_dual
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
#LOAD TRAINING DATA
tr_fpz,tr_pz,tr_hyp = read_mat('SC4001E0')
tr_label,tr_fpz,tr_pz,tr_dual,tr_y = data_reshape(tr_fpz,tr_pz,tr_hyp)
#LOAD TEST DATA
ts_fpz,ts_pz,ts_hyp = read_mat('SC4002E0')
ts_label,ts_fpz,ts_pz,ts_dual,ts_y = data_reshape(ts_fpz,ts_pz,ts_hyp)

#MODEL TRAINED USING FPZ CHANNEL
model_fpz = model_single()
model_fpz.fit(x=tr_fpz,y=tr_y,epochs=25,verbose=1,callbacks=[EarlyStopping(patience=3,restore_best_weights=True)])
pred_fpz = model_fpz.predict(ts_fpz,verbose=1)
label_fpz = np.argmax(pred_fpz,axis=-1)
print('Accuracy : {} %'.format(accuracy_score(ts_label,label_fpz)*100))


#MODEL TRAINED USING PZ CHANNEL
model_pz = model_single()
model_pz.fit(x=tr_pz,y=tr_y,epochs=25,verbose=1,callbacks=[EarlyStopping(patience=3,restore_best_weights=True)])
pred_pz = model_fpz.predict(ts_pz,verbose=1)
label_pz = np.argmax(pred_pz,axis=-1)
print('Accuracy : {}%'.format(accuracy_score(ts_label,label_pz)*100))


#MODEL TRAINED USING FPZ AND PZ CHANNELS
model_two = model_dual()
model_two.fit(x=tr_dual,y=tr_y,epochs=25,verbose=1,callbacks=[EarlyStopping(patience=3,restore_best_weights=True)])
pred_two = model_fpz.predict(ts_dual,verbose=1)
label_two = np.argmax(pred_two,axis=-1)
print('Accuracy : {}%'.format(accuracy_score(ts_label,label_two)*100))


