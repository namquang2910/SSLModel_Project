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
import pywt
from collections import Counter

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
            
#============== Extract data ============================
def get_ecg_dataframe(data,sample_rate, segment_size, stride, is_train):
    ecg_ls = []
    label_ls = []
    pts_window = segment_size*100
    major_rate = 1
    ecg_data = np.array(data["signal"]["chest"]["ECG"][:,0])
    
    #Preprocessing the ECG data
    band_passed_ecg =  nk.signal_filter(ecg_data, sampling_rate=sample_rate, lowcut=0.1, highcut=100, method='butterworth_zi', order = 2)
    emg = moving_average(band_passed_ecg, window_size=10)
    ecg_val = nk.signal_resample(emg, sampling_rate=sample_rate, desired_sampling_rate=100)
    
    #Scale the ecg
    encoder =  MinMaxScaler(feature_range=(-0.5, 0.5))
    ecg_scale = encoder.fit_transform(ecg_val.reshape(-1,1))

    #Downsample the label
    label_resampled = rescale_label(data['label'],ecg_data,sample_rate,100)

    #Extract the label and ecg window
    for i in range(0,len(label_resampled)-pts_window + 1,stride*100):
        label_counts = Counter(label_resampled[i: i +pts_window])
        max_item, max_count = label_counts.most_common(1)[0]
        if (max_count/ pts_window) >= major_rate: 
            if max_item in [1,2,3,4]:
                l_temp = 1 if max_item == 2 else 0
                ecg_ls.append(ecg_scale[i: i +pts_window])
                label_ls.append(l_temp)

    return np.array(ecg_ls), np.array(label_ls)
        

#============== Load dataset from list folder ==================
def load_process_extract_ls(root_dir,folder_ls, sample_rate,win_size, strides,is_train = False):
    ecg_data = []
    label_data = []

    for folder in folder_ls:
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