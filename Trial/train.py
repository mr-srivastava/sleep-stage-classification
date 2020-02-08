# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 08:12:08 2020

@author: Aadarsh Srivastava
"""

import numpy as np
from ntpath import basename

psg_files=['SC4001E0-PSG','SC4002E0-PSG','SC4062E0-PSG','SC4031E0-PSG','SC4041E0-PSG','SC4042E0-PSG','SC4051E0-PSG','SC4052E0-PSG']


def reader(i):
    filename = basename(psg_files[i]).replace("-PSG", ".npz")
    file=np.load('C:/Users/Aadarsh Srivastava/OneDrive/Desktop/Project Github/Data/'+filename)
    return file['x'],file['y']

def join_data(x,y,i):
    temp_x,temp_y=reader(i)
    x=np.concatenate((x,temp_x),axis=0)
    y=np.concatenate((y,temp_y),axis=0)
    return x,y

for i in range(len(psg_files)):
    if i==0:
        signal,stages=reader(i)
    else:
        signal,stages=join_data(signal,stages,i)



    
