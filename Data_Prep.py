# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 09:35:09 2019

@author: Aadarsh Srivastava
"""
import numpy as np
from mne.io import read_raw_edf
from mne import read_annotations

fs = 100
epochs = 30
samples = fs*epochs

def hyp_reader(data):
    desc=data.description
    dur=data.duration
    dur=dur.astype(int)
    def rep(index):
        temp = np.array([desc[index]]*(int(dur[index]/30)))
        return temp
    hyp = rep(0)
    for i in range(1,len(dur)-1):
        hyp=np.concatenate((hyp,rep(i)),axis=0)
    return hyp
    

def read_EDF(psg,hyp):
    data  = read_raw_edf('D:/Project/Dataset/Sleep EDF Database Expanded/sleep-cassette/'+psg+'.edf')
    raw_data = data.get_data()
    data_hyp  = read_annotations('D:/Project/Dataset/Sleep EDF Database Expanded/sleep-cassette/'+hyp+'.edf')
    l1 = raw_data[0]
    l2 = raw_data[1]
    l3 = hyp_reader(data_hyp)
    return l1,l2,l3

def merge_data(fpz,pz,label,psg,hyp):
    temp1,temp2,temp3=read_EDF(psg,hyp)
    fpz=np.concatenate((fpz,temp1),axis=0)
    pz=np.concatenate((pz,temp2),axis=0)
    label=np.concatenate((label,temp3),axis=0)
    return fpz,pz,label

def data_reshape(fpz,pz,label):
    fpz= np.reshape(fpz,(label.shape[0],samples,1))
    pz= np.reshape(pz,(label.shape[0],samples,1))
    return fpz,pz