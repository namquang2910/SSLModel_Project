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
    print('5-15')
    band_passed_ecg =  nk.signal_filter(df, sampling_rate=sample_rate, lowcut=0.5, highcut=40, method='butterworth_zi', order = 2)
    emg = moving_average(band_passed_ecg, window_size=10)
    # Step 4: Downsample the filtered ECG signal 
    downsampled_ecg = nk.signal_resample(emg, sampling_rate=sample_rate, desired_sampling_rate=100)
    return downsampled_ecg


#=================  Get label  =========================
def slice_per_label(labelseq,ecg,flabel,time_window,step):
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
            if condition==True:
                l_temp = 1 if l == 2 else 0
                taken_indices.append((i,l_temp))

    return taken_indices

#============== Extract data ============================
def get_ecg_dataframe(data,sample_rate, segment_size, stride, is_train):
    ecg_ls = []
    label_ls = []
    pts_window = segment_size*100
    print("minmax")
    encoder = StandardScaler()
    
    label = data['label']
    ecg_data = np.array(data["signal"]["chest"]["ECG"][:,0])
    
    #Filter the ecg and downsample
    ecg_val = ecg_preprocessing(ecg_data, sample_rate)
    
    scale_data = encoder.fit_transform(ecg_val.reshape(-1,1))
    label_new = rescale_label(label,ecg_data,sample_rate,100)
    
    #Get the starting window
    taken_indices = slice_per_label(label_new, ecg_val,100, segment_size,stride)
    
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