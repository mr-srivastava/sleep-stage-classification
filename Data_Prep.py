# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 09:35:09 2019

@author: Aadarsh Srivastava
"""
import pandas as pd
import scipy.io
eeg_fpz  = scipy.io.loadmat('D:/Project/Github_dwn/SC4002E0/matlab/eeg_fpz.mat')
eeg_pz  = scipy.io.loadmat('D:/Project/Github_dwn/SC4002E0/matlab/eeg_pz.mat')
event  = scipy.io.loadmat('D:/Project/Github_dwn/SC4002E0/matlab/event_marker.mat')
hyp = scipy.io.loadmat('D:/Project/Github_dwn/SC4002E0/matlab/hypnogram.mat')

data_dic = {'eeg_fpz':eeg_fpz['signal'],'eeg_pz':eeg_pz['signal'],'event':event['signal'],'hyp':hyp['hypnogram']}
data = pd.DataFrame()
data['eeg_fpz']=list(map(float,eeg_fpz['signal'][:,0]))
data['eeg_pz']=list(map(float,eeg_pz['signal'][:,0]))
data['event']=list(map(float,event['signal'][:,0]))

#sampling frequency is 100 Hz
fs=100
epochs =30
data['hyp']=0
for i in range(0,len(hyp['hypnogram'])-1):
    a = hyp['hypnogram'][i]
    if a=='W':
        a=5
    if a=='R':
        a=6
    data['hyp'][3000*i]=float(a)
    data['hyp'][3000*i:2999+(3000*i)]=float(a)
    

data.to_csv(r'C:\Users\Aadarsh Srivastava\OneDrive\Desktop\test_data.csv', index=False) 