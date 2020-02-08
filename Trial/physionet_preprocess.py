from mne import read_annotations
from mne.io import read_raw_edf
import numpy as np
import ntpath

#Label values
W=0
N1=1
N2=2
N3=3
REM=4
UNKNOWN=5
stage_dict={"W":W,"N1":N1,"N2":N2,"N3":N3,"REM":REM,"UNKNOWN":UNKNOWN}
class_dict={0:"W",1:"N1",2:"N2",3:"N3",4:"REM",5:"UNKNOWN"}
ann2label={"Sleep stage W":0,"Sleep stage 1":1,"Sleep stage 2":2,"Sleep stage 3":3,"Sleep stage 4":3,"Sleep stage R":4,"Sleep stage ?":5,"Movement time":5}
EPOCH_SEC_SIZE=30

#File Names
psg_files=['SC4001E0-PSG','SC4002E0-PSG','SC4062E0-PSG','SC4031E0-PSG','SC4041E0-PSG','SC4042E0-PSG','SC4051E0-PSG','SC4052E0-PSG']
ann_files=['SC4001EC-Hypnogram','SC4002EC-Hypnogram','SC4062EC-Hypnogram','SC4031EC-Hypnogram','SC4041EC-Hypnogram','SC4042EC-Hypnogram','SC4051EC-Hypnogram','SC4052EC-Hypnogram']
select_chs=['EEG Fpz-Cz','EEG Pz-Oz','EOG horizontal','EMG submental']

psg_fnames=[]
ann_fnames=[]

for i in range(len(psg_files)):
    psg_fnames.append('D:/Project/Dataset/Sleep EDF Database Expanded/sleep-cassette/'+psg_files[i]+'.edf')
    ann_fnames.append('D:/Project/Dataset/Sleep EDF Database Expanded/sleep-cassette/'+ann_files[i]+'.edf')

for i in range(len(psg_fnames)):
    raw=read_raw_edf(psg_fnames[i],preload=True,stim_channel=None)
    sampling_rate=raw.info['sfreq']
    raw_ch_df=raw.to_data_frame(scaling_time=sampling_rate)[select_chs]
    raw_ch_df.set_index(np.arange(len(raw_ch_df)))
    ann=read_annotations(ann_fnames[i])
    #Generate label and remove indices
    remove_idx=[]
    labels=[]
    label_idx=[]
    for x in range(len(ann.description)):
        onset_sec=ann.onset[x]
        duration_sec=ann.duration[x]
        ann_str=ann.description[x]
        label=ann2label[ann_str]
        if label != UNKNOWN:
            if duration_sec % EPOCH_SEC_SIZE != 0:
                raise Exception("Something wrong")
            duration_epoch = int(duration_sec / EPOCH_SEC_SIZE)
            label_epoch = np.ones(duration_epoch, dtype=np.int) * label
            labels.append(label_epoch)
            idx = int(onset_sec * sampling_rate) + np.arange(duration_sec * sampling_rate, dtype=np.int)
            label_idx.append(idx)
            
            print("Include onset:{}, duration:{}, label:{} ({})".format(onset_sec,duration_sec,label,ann_str))
        else:
            idx = int(onset_sec * sampling_rate) + np.arange(duration_sec * sampling_rate, dtype=np.int)
            remove_idx.append(idx)
            print("Remove onset:{}, duration:{}, label:{} ({})".format(onset_sec,duration_sec,label,ann_str))
    labels = np.hstack(labels)
    
    print("before remove unwanted: {}".format(np.arange(len(raw_ch_df)).shape))
    if len(remove_idx)>0:
        remove_idx=np.hstack(remove_idx)
        select_idx=np.setdiff1d(np.arange(len(raw_ch_df)),remove_idx)
    else:
        select_idx=np.arange(len(raw_ch_df))
    print("after remove unwanted: {}".format(select_idx.shape))
    
    #Select only the data with labels
    print("before intersect label: {}".format(select_idx.shape))
    label_idx=np.hstack(label_idx)
    select_idx=np.intersect1d(select_idx,label_idx)
    print("after intersect label: {}".format(select_idx.shape))
    
    #Remove extra index
    if len(label_idx)>len(select_idx):
        print("before remove extra labels: {}, {}".format(select_idx.shape,labels.shape))
        extra_idx=np.setdiff1d(label_idx,select_idx)
        #Trim the tail
        if np.all(extra_idx>select_idx[-1]):
            n_trims=len(select_idx)%int(EPOCH_SEC_SIZE*sampling_rate)
            n_label_trims=int(math.ceil(n_trims/(EPOCH_SEC_SIZE*sampling_rate)))
            select_idx=select_idx[:-n_trims]
            labels=labels[:-n_label_trims]
            print("after remove extra labels: {}, {}".format(select_idx.shape,labels.shape))
            
    #Remove movement and unknown stages if any
    raw_ch=raw_ch_df.values[select_idx]
    
    #Verify that we can split into 30-s epochs
    if len(raw_ch)%(EPOCH_SEC_SIZE*sampling_rate)!=0:
        raise Exception("Something wrong")
    n_epochs=len(raw_ch)/(EPOCH_SEC_SIZE*sampling_rate)
    
    #Get epochs and their corresponding labels
    x=np.asarray(np.split(raw_ch,n_epochs)).astype(np.float32)
    y=labels.astype(np.int32)
    assert len(x)==len(y)
    
    #Select on sleep periods
    w_edge_mins=30
    nw_idx=np.where(y!=stage_dict["W"])[0]
    start_idx=nw_idx[0]-(w_edge_mins*2)
    end_idx=nw_idx[-1]+(w_edge_mins*2)
    if start_idx<0:start_idx=0
    if end_idx>=len(y):end_idx=len(y)-1
    select_idx=np.arange(start_idx,end_idx+1)
    print("Data before selection: {}, {}".format(x.shape,y.shape))
    x=x[select_idx]
    y=y[select_idx]
    print("Data after selection: {}, {}".format(x.shape,y.shape))
    
    # Save
    filename = ntpath.basename(psg_files[i]).replace("-PSG", ".npz")
    save_dict={"x":x,"y":y,"fs":sampling_rate}
    np.savez('C:/Users/Aadarsh Srivastava/OneDrive/Desktop/Project Github/Data/'+filename, **save_dict)

    print("\n=======================================\n")
'''
raw.set_annotations(ann,emit_warning=False)
annotation_desc_2_event_id = {'Sleep stage W': 1,'Sleep stage 1': 2,'Sleep stage 2': 3,'Sleep stage 3': 4,'Sleep stage 4': 4,'Sleep stage R': 5}

events, _ = events_from_annotations(raw, event_id=annotation_desc_2_event_id, chunk_duration=30.)

# create a new event_id that unifies stages 3 and 4
event_id = {'Sleep stage W': 1,'Sleep stage 1': 2,'Sleep stage 2': 3,'Sleep stage 3/4': 4,'Sleep stage R': 5}
tmax = 30. - 1. / raw.info['sfreq']  # tmax in included

epochs= Epochs(raw=raw, events=events,event_id=event_id, tmin=0., tmax=tmax, baseline=None)
print(epochs)
data=epochs.get_data()
events_trial=epochs.events

ann.description
ann.duration
ann.onset
'''