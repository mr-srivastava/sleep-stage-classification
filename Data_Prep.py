# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 09:35:09 2019

@author: Aadarsh Srivastava
"""
import numpy as np
import pandas as pd
import scipy.io

data = pd.DataFrame()

def read_mat(file):
    path= 'D:/Project/Github_dwn/SC4001E0/matlab/'+file+'.mat'
    temp = scipy.io.loadmat(path)
    data[file]=list(map(float,temp['signal'][:,0]))

'''eeg_fpz  = scipy.io.loadmat('D:/Project/Github_dwn/SC4002E0/matlab/eeg_fpz.mat')
eeg_pz  = scipy.io.loadmat('D:/Project/Github_dwn/SC4002E0/matlab/eeg_pz.mat')
event  = scipy.io.loadmat('D:/Project/Github_dwn/SC4002E0/matlab/event_marker.mat')
'''
hyp = scipy.io.loadmat('D:/Project/Github_dwn/SC4001E0/matlab/hypnogram.mat')
np.unique(hyp["hypnogram"])

read_mat('eeg_fpz')
read_mat('eeg_pz')
read_mat('event_marker')

#data_dic = {'eeg_fpz':eeg_fpz['signal'],'eeg_pz':eeg_pz['signal'],'event':event['signal'],'hyp':hyp['hypnogram']}

'''data['eeg_fpz']=list(map(float,eeg_fpz['signal'][:,0]))
data['eeg_pz']=list(map(float,eeg_pz['signal'][:,0]))
data['event']=list(map(float,event['signal'][:,0]))
'''
#sampling frequency is 100 Hz
fs=100
epochs =30
data['hyp']=0

for i in range(0,len(hyp['hypnogram'])-1):
    a = hyp['hypnogram'][i]
    if a=='W':
        a=0
    if a=='R':
        a=5
    data['hyp'][3000*i]=float(a)
    data['hyp'][3000*i:2999+(3000*i)]=float(a)
    
data.groupby('hyp').count()

    

data.to_csv(r'C:\Users\Aadarsh Srivastava\OneDrive\Desktop\Project GIthub\Data CSV\file_new.csv', index=False) 
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 09:35:09 2019

@author: Aadarsh Srivastava
"""
import numpy as np
import pandas as pd
import scipy.io

data = pd.DataFrame()

def read_mat(file):
    path= 'D:/Project/Github_dwn/SC4001E0/matlab/'+file+'.mat'
    temp = scipy.io.loadmat(path)
    data[file]=list(map(float,temp['signal'][:,0]))

'''eeg_fpz  = scipy.io.loadmat('D:/Project/Github_dwn/SC4002E0/matlab/eeg_fpz.mat')
eeg_pz  = scipy.io.loadmat('D:/Project/Github_dwn/SC4002E0/matlab/eeg_pz.mat')
event  = scipy.io.loadmat('D:/Project/Github_dwn/SC4002E0/matlab/event_marker.mat')
'''
hyp = scipy.io.loadmat('D:/Project/Github_dwn/SC4001E0/matlab/hypnogram.mat')
np.unique(hyp["hypnogram"])

read_mat('eeg_fpz')
read_mat('eeg_pz')
read_mat('event_marker')

#data_dic = {'eeg_fpz':eeg_fpz['signal'],'eeg_pz':eeg_pz['signal'],'event':event['signal'],'hyp':hyp['hypnogram']}

'''data['eeg_fpz']=list(map(float,eeg_fpz['signal'][:,0]))
data['eeg_pz']=list(map(float,eeg_pz['signal'][:,0]))
data['event']=list(map(float,event['signal'][:,0]))
'''
#sampling frequency is 100 Hz
fs=100
epochs =30
data['hyp']=0

for i in range(0,len(hyp['hypnogram'])-1):
    a = hyp['hypnogram'][i]
    if a=='W':
        a=0
    if a=='R':
        a=5
    data['hyp'][3000*i]=float(a)
    data['hyp'][3000*i:2999+(3000*i)]=float(a)
    
data.groupby('hyp').count()

    

data.to_csv(r'C:\Users\Aadarsh Srivastava\OneDrive\Desktop\Project GIthub\Data CSV\file_new.csv', index=False) 
