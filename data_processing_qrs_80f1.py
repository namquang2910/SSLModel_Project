import pandas as pd
import numpy as np
import neurokit2 as nk
from csv import writer
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score,precision_score, recall_score
from tqdm import tqdm
from scipy.signal import medfilt, butter, filtfilt, iirnotch
import random

#==============Detect outlier ========================
def find_average_qrs(ecg_signal,left_ms, right_ms, sample_rate=100):

    # Set the outer side of QRS range
    qrs_dict = {
        'qrs': [],
        'label': [],
        'rpeak_indice': [],
        'outlier': [],
        'distance': [],
        'avg': None,
        'left_sample': int((left_ms / 1000) * sample_rate),
        'right_sample': int((right_ms / 1000) * sample_rate)
    }
    # Find the R peaks
    _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=sample_rate)
    rpeak_indices = rpeaks['ECG_R_Peaks']

    # Detect the QRS and ensure valid indexing
    for rpeak in rpeak_indices:
        start_idx = rpeak - qrs_dict['left_sample']
        end_idx = rpeak + qrs_dict['right_sample']
        
        # Ensure indices are within valid range
        if start_idx >= 0 and end_idx <= len(ecg_signal):
            qrs = ecg_signal[start_idx:end_idx]
            qrs_dict['qrs'].append(qrs)
            qrs_dict['rpeak_indice'].append(rpeak)
  
    # Convert to NumPy array to ensure consistent shape
    if qrs_dict['qrs']:
        qrs_dict['qrs'] = np.array(qrs_dict['qrs'])  # Ensure uniform shape
        qrs_dict['avg'] = np.mean(qrs_dict['qrs'], axis=0)
        qrs_dict['distance'] = [np.linalg.norm(qrs - qrs_dict['avg']) for qrs in qrs_dict['qrs']]
    return qrs_dict

    
def qrs_detect_impute(ecg_signal,left_ms, right_ms, sample_rate=100, up_percen = 99):
    signal = np.copy(ecg_signal)  
    #Find the aveage and qrs range
    qrs = find_average_qrs(signal,left_ms, right_ms, sample_rate=100)
    upperbound = np.percentile(qrs['distance'], up_percen)
    #Find the outliers
    for i, dis in enumerate(qrs['distance']):
        if dis > upperbound:
            #$Get the indice Rpeak of the outlier
            outlier_indice = qrs['rpeak_indice'][i]
            #Assign the outlier to average
            #signal[outlier_indice - qrs['left_sample']:outlier_indice + qrs['right_sample']] = qrs['qrs'][i-1]
            qrs['outlier'].append(outlier_indice)
    return signal, qrs

#================ Data Scaling =======================
    
def rescale_label(label,ecg_data,input_sample, output_sample):
    length_ecg = len(ecg_data)
    #Find the position that start to change
    change_indices = np.where(label[1:] != label[:-1])[0]  + 1
    change_indices = np.concatenate(([0], change_indices, [len(ecg_data)]))
    
    out_indices = np.round((change_indices/input_sample)*output_sample)
    indices_label = [label[i-1] for i in change_indices]
    
    label_new = []
    for i, value in enumerate(indices_label):
        if i < 0:
            temp_list = [value]* int((out_indices[i+1] - out_indices[i]))
        else:
            temp_list = [value]* int((out_indices[i] - out_indices[i-1]))
        label_new.extend(temp_list)
    return np.array(label_new)
    
#==================== ECG Processing ================
def moving_average(signal, window_size=10):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')
    
def ecg_preprocessing(df, sample_rate):
    #ecg_baseline_corrected = butter_highpass(df, cutoff=0.5, fs=sample_rate, order=4)
    # Step 2: Apply a notch filter to remove powerline interference (50 Hz)
    #notched_ecg = notch_filter(df, freq = 50, fs = 700, quality=30) 
    # Step 3: Apply a band-pass filter to isolate the QRS complex (8-20 Hz)
    print('5-15')
    band_passed_ecg =  nk.signal_filter(df, sampling_rate=sample_rate, lowcut=0.5, highcut=30, method='butterworth_zi', order = 2)
    emg = moving_average(band_passed_ecg, window_size=10)
    # Step 4: Downsample the filtered ECG signal 
    downsampled_ecg = nk.signal_resample(emg, sampling_rate=sample_rate, desired_sampling_rate=100)
    return downsampled_ecg


#=====================================================
#                    SWELL-KW
#=====================================================

#===============  Get label  =========================
def slice_per_label_swell(labelseq,ecg,flabel,time_window,step, qrs_filter,is_train):
    labelseq +=1
    count_nonstress = 0
    pts_window=time_window*flabel
    taken_indices=[]
    conv=np.array([1 for i in range(0,pts_window)])
    condition = False  # Initialize condition to a default value
    for i in range(0,len(labelseq)-pts_window + 1,flabel*step):   #Sliding 5s window, step 1s
        extr=labelseq[i:i+pts_window]
        res=np.sum(np.multiply(extr,conv))
        l=labelseq[i]
        if l in [1,2]:
            condition=l*pts_window==res
            if is_train:
                for index in qrs_filter['outlier']:
                    if index > i and index < ( i + pts_window) and l == 1:
                        condition = False
                        count_nonstress += 1
            if condition==True:
                taken_indices.append((i,l-1))
                    
    if qrs_filter:
        print(f"remove nonstress: {count_nonstress}")
    return taken_indices

#============== Extract data ============================
def get_ecg_dataframe_swell(data,sample_rate, segment_size, stride,train = False):
    ecg_ls = []
    label_ls = []
    pts_window = segment_size*100
    encoder = StandardScaler()
    label = np.array(data['label']) 
    ecg_data = np.array(data["ECG"])
    #Filter the ecg and downsample
    ecg_val = ecg_preprocessing(ecg_data, sample_rate)
    impute_val, qrs_ls = qrs_detect_impute(ecg_val,200, 220, sample_rate=100)
    scale_data = encoder.fit_transform(ecg_val.reshape(-1,1))
    label_new = rescale_label(label,ecg_data,sample_rate,100)
    #Get the starting window
    taken_indices = slice_per_label_swell(label_new, ecg_val,100, segment_size,stride,qrs_ls,train)
    #Create a ecg and label window
    ecg_ls = np.array([scale_data[start[0]:start[0] + pts_window] for start in taken_indices])
    label_ls = np.array([start[1] for start in taken_indices])
    return ecg_ls, label_ls

#============== Load dataset from folder ==================
def load_process_swell(root_dir, sample_rate, win_size, strides, is_train):
    ecg_data = []
    label_data = []
    # Second pass: Apply the same scaler to each subject
    for file in os.listdir(root_dir):
        if file.endswith('.pkl'):
            file_path = os.path.join(root_dir, file)
            print(f"Processing: {file_path}")
            data = pd.read_pickle(file_path)
            ecg_temp, label_temp = get_ecg_dataframe_swell(data, sample_rate, win_size, strides,is_train)
            
            ecg_data.extend(ecg_temp)
            label_data.extend(label_temp)
    return np.array(ecg_data), np.array(label_data)

#============== Load dataset from list folder ==================
def load_process_extract_ls_swell(root_dir,folder_ls, sample_rate,win_size, strides,is_train = False):
    ecg_data = []
    label_data = []
    subject = []
    df = pd.DataFrame()

    for file in folder_ls:
        if file.endswith('.pkl'):
            file_path = os.path.join(root_dir, file)
            #Read data and store
            data = pd.read_pickle(file_path)
            ecg_temp, label_temp = get_ecg_dataframe_swell(data, sample_rate, win_size, strides, is_train)
            print(f"Working with: {file_path}")
            ecg_data.extend(ecg_temp)
            label_data.extend(label_temp)
    
    return np.array(ecg_data), np.array(label_data)


#=====================================================
#                       WESAD
#=====================================================

#=================  Get label  =========================
def slice_per_label(labelseq,ecg,flabel,time_window,step, qrs_filter,is_train):
    count_nonstress = 0
    pts_window=time_window*flabel
    taken_indices=[]
    conv=np.array([1 for i in range(0,pts_window)])
    condition = False  # Initialize condition to a default value
    for i in range(0,len(labelseq)-pts_window + 1,flabel*step):   #Sliding 5s window, step 1s
        extr=labelseq[i:i+pts_window]
        res=np.sum(np.multiply(extr,conv))
        l=labelseq[i]
        if l in [1,2,3,4]:
            condition=l*pts_window==res
            if is_train:
                for index in qrs_filter['outlier']:
                    if index > i and index < ( i + pts_window) and l != 2:
                        condition = False
                        count_nonstress += 1
            if condition==True:
                l_temp = 1 if l == 2 else 0
                taken_indices.append((i,l_temp))
    if qrs_filter:
        print(f"remove nonstress: {count_nonstress}")
    return taken_indices

#============== Extract data ============================
def get_ecg_dataframe(data,sample_rate, segment_size, stride, is_train):
    ecg_ls = []
    label_ls = []
    pts_window = segment_size*100
    
    encoder = StandardScaler()
    
    label = data['label']
    ecg_data = np.array(data["signal"]["chest"]["ECG"][:,0])
    
    #Filter the ecg and downsample
    ecg_val = ecg_preprocessing(ecg_data, sample_rate)
    
    impute_val, qrs_ls = qrs_detect_impute(ecg_val,200, 220, sample_rate=100)
    scale_data = encoder.fit_transform(ecg_val.reshape(-1,1))
    label_new = rescale_label(label,ecg_data,sample_rate,100)
    
    #Get the starting window
    taken_indices = slice_per_label(label_new, ecg_val,100, segment_size,stride, qrs_ls,is_train)
    
    #Create a ecg and label window
    ecg_ls = np.array([scale_data[start[0]:start[0] + pts_window] for start in taken_indices])
    label_ls = np.array([start[1] for start in taken_indices])
    return ecg_ls, label_ls

#============== Load dataset from folder ==================
def load_process(root_dir, sample_rate, win_size, strides,is_train = False):
    ecg_data = None
    label_data = []
    
    # Second pass: Apply the same scaler to each subject
    for folder in os.listdir(root_dir):
        if folder == ".DS_Store" or folder == ".ipynb_checkpoints":
            continue
        file_folder = os.path.join(root_dir, folder)
        for file in os.listdir(file_folder):
            if file.endswith('.pkl'):
                file_path = os.path.join(file_folder, file)
                print(file_path)
                data = pd.read_pickle(file_path)
                ecg_temp, label_temp = get_ecg_dataframe(data, sample_rate, win_size, strides,is_train)
                print(f"Working with: {file_path}")
                ecg_new = np.expand_dims(ecg_temp, axis=-1)  # Add a single channel (original signal)

                # Stack results dynamically
                if ecg_data is None:
                    ecg_data = ecg_new
                else:
                    ecg_data = np.vstack((ecg_data, ecg_new))  # Stack along samples (axis=0)
                label_data.extend(label_temp)  # Keep labels as a list

    return ecg_data, np.array(label_data)
        

#============== Load dataset from list folder ==================
def load_process_extract_ls(root_dir,folder_ls, sample_rate,win_size, strides,is_train = False):
    ecg_data = []
    label_data = []
    df = pd.DataFrame()

    for folder in folder_ls:
        if folder == ".DS_Store" or folder == ".ipynb_checkpoints":
            continue
        else:
            file_folder = os.path.join(root_dir, folder)
            folder = os.listdir(file_folder)
            for file in folder:
                if file.endswith('.pkl'):
                    file_path = os.path.join(file_folder, file)
                    #Read data and store
                    data = pd.read_pickle(file_path)
                    ecg_temp, label_temp = get_ecg_dataframe(data, sample_rate, win_size, strides,is_train)
        
                    print(f"Working with: {file_path}")
                    ecg_data.extend(ecg_temp)
                    label_data.extend(label_temp)
    
    return np.array(ecg_data), np.array(label_data)