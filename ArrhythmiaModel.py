import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import wfdb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import neurokit2 as nk
from scipy.signal import resample, medfilt
import pywt
import pickle
import time
import os
import csv 
import mne
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from zoneinfo import ZoneInfo

#import mne to read edf files
#Change model to take 30 seconds, try single bit masking, if successful move on to multiple R-peaks

def downsample(data, original_rate, target_rate):
    num_samples = int(len(data) * target_rate / original_rate)
    return resample(data, num_samples)

def calculate_l1(predictions, targets):
    l1_norm = torch.norm(predictions-targets, p=1)
    l1_sum = torch.sum(torch.abs(targets))
    accuracy = 100 * (1 - (l1_norm / l1_sum))
    return accuracy

def calculate_l2(predictions, targets):
    l2_norm = torch.norm(predictions - targets, p=2)
    l2_sum = torch.norm(targets, p=2)
    accuracy = 100 * (1 - (l2_norm / l2_sum))
    return accuracy

def calculate_MAPE(predictions, targets):
    mape = torch.mean(torch.abs((targets - predictions) / targets))
    return mape

class MaskingLayer(nn.Module):
    def __init__(self):
        super(MaskingLayer, self).__init__()
        
    def forward(self, seq, mask):
        seq = seq.clone()   # avoid in-place ops on computation graph
        seq[mask] = 0
        return seq
        
class Conv_Block(nn.Module):
    def __init__(self, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(1, embed_dim, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(embed_dim, embed_dim * 2, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(embed_dim * 2, hidden_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Input: (B, L) → (B, 1, L)
        x = x.unsqueeze(1)
        
        x = F.leaky_relu(self.conv1(x), negative_slope=0.01)
        x = self.pool1(x)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.01)
        x = self.pool2(x)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.01)
        return x

class Encoder1D_Mask(nn.Module):
    def __init__(self, embed_dim=32, hidden_dim=64, lstm_hidden_dim=128):
        super().__init__()
        self.mask_peaks = MaskingLayer()
        self.QRS_Conv_Block = Conv_Block(embed_dim, hidden_dim)
        self.P_T_Conv_Block = Conv_Block(embed_dim, hidden_dim)
        self.bilstm = nn.LSTM(hidden_dim * 2, lstm_hidden_dim,
                              bidirectional=True, batch_first=True)

    def forward(self, x, mask_qrs, mask_pt):
        # Apply masking
        qrs_x = self.mask_peaks(x, mask_qrs)
        pt_x = self.mask_peaks(x, mask_pt)

        # Encode each branch
        qrs_x = self.QRS_Conv_Block(qrs_x)
        pt_x = self.P_T_Conv_Block(pt_x)

        # Concatenate features along channels
        x = torch.cat([qrs_x, pt_x], dim=1)  # (B, hidden_dim*2, L/4)

        # BiLSTM expects (B, L, C)
        x = x.permute(0, 2, 1)
        x, _ = self.bilstm(x)
        x = x.permute(0, 2, 1)  # (B, 2*lstm_hidden_dim, L/4)
        
        return x


class Decoder1D(nn.Module):
    def __init__(self, embed_dim=32, hidden_dim=64, lstm_hidden_dim=128):
        super().__init__()
        self.deconv1 = nn.Conv1d(lstm_hidden_dim * 2, embed_dim * 2,
                                 kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.Conv1d(embed_dim * 2, embed_dim,
                                 kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.Conv1d(embed_dim, 1,
                                 kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.deconv1(x), negative_slope=0.01)
        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)

        x = F.leaky_relu(self.deconv2(x), negative_slope=0.01)
        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)

        x = self.deconv3(x)  # (B, 1, L)
        return x  # keep (B, 1, L)


class MAE1D_Mask(nn.Module):
    def __init__(self, embed_dim=32, hidden_dim=64, lstm_hidden_dim=128):
        super().__init__()
        self.encoder = Encoder1D_Mask(embed_dim, hidden_dim, lstm_hidden_dim)
        self.decoder = Decoder1D(embed_dim, hidden_dim, lstm_hidden_dim)

    def forward(self, x, mask_qrs, mask_pt):
        x = self.encoder(x, mask_qrs, mask_pt)
        x = self.decoder(x)
        return x.squeeze(1)

def random_mask_peaks(rpeaks, seq, mask_length, num_peaks_to_mask):
    mask = np.zeros_like(seq, dtype=bool)  # Create a mask with the same length as seq, initially all False

    if len(rpeaks) > 0:  # If there are detected peaks
        # Randomly select 'num_peaks_to_mask' peaks from rpeaks
        num_peaks_to_mask = min(num_peaks_to_mask, len(rpeaks))
        selected_peaks = random.sample(list(rpeaks), min(num_peaks_to_mask, len(rpeaks)))

        for peak in selected_peaks:
            mask_half = mask_length // 2
            start = int(max(0, peak - mask_half))
            end = int(min(len(seq), peak + mask_half))
            mask[start:end] = True
    
    return mask

def load_record(record_id, seq_len=1000, stride=1000):
    record_name = str(record_id)
    if not os.path.exists(f"mit_bih/{record_name}.dat"):
        wfdb.dl_database('mitdb', './', records=[record_name])
    record = wfdb.rdrecord('mit_bih/' + record_name)
    
    #Extract each channel
    signal_data = record.p_signal
    ch1 = preprocessing_data(signal_data[:, 0]).reshape(-1, 1)
    ch2 = preprocessing_data(signal_data[:, 1]).reshape(-1, 1)

    #Scale the signal
    scaler1 = MinMaxScaler(feature_range=(-0.5, 0.5))
    scaler2 = MinMaxScaler(feature_range=(-0.5, 0.5))
    ch1_scale = scaler1.fit_transform(ch1)
    ch2_scale = scaler2.fit_transform(ch2)

    
    # Create sequences using sliding window
    seq_ch1 = [ch1_scale[i:i+seq_len] for i in range(0, len(ch1_scale)-seq_len+1, stride)]
    seq_ch2 = [ch2_scale[i:i+seq_len] for i in range(0, len(ch2_scale)-seq_len+1, stride)]
    seq_ch1.extend(seq_ch2)
    return seq_ch1

def load_multiple_records(record_ranges, seq_len=1000, stride_len=100):
    all_data = []

    for start, end in tqdm(record_ranges):
        for record_id in range(start, end + 1):
            record_data = load_record(record_id, seq_len, stride_len)
            all_data.extend(record_data)
    # Save to pkl
    with open('database.pkl', 'wb') as f:
        pickle.dump(all_data, f)
    
    return all_data

def fixed_position_mask_peaks(rpeaks, seq, mask_length, num_peaks_to_mask):
    mask = np.zeros_like(seq, dtype=bool)  # Create a mask with the same length as seq, initially all False

    if len(rpeaks) > 0:  # If there are detected peaks
        # Randomly select 'num_peaks_to_mask' peaks from rpeaks
        selected_peaks = rpeaks[1:2*num_peaks_to_mask + 1: 2]

        for peak in selected_peaks:
            mask_half = mask_length // 2
            start = int(max(0, peak - mask_half))
            end = int(min(len(seq), peak + mask_half))
            mask[start:end] = True
    
    return mask

def fixed_position_mask_P_T(rpeaks, seq, mask_length, num_peaks_to_mask):
    mask = np.zeros_like(seq, dtype=bool)  # Create a mask with the same length as seq, initially all False

    if len(rpeaks) > 0:  # If there are detected peaks
        # Randomly select 'num_peaks_to_mask' peaks from rpeaks
        selected_peaks = rpeaks[1:2*num_peaks_to_mask + 1: 2]

        for peak in selected_peaks:
            P_start = int(max(0, peak - 15 - mask_length))
            P_end = int(max(0, peak + 15))
            T_start = int(max(0, peak + 15))
            T_end = int(min(0, peak + 15 + mask_length))            
            mask[P_start:P_end] = True
            mask[T_start:T_end] = True
    return mask

def fixed_position_mask_peaks_int(rpeaks, seq, mask_length, num_peaks_to_mask):
    mask = np.ones_like(seq, dtype=float)  # Create a mask with the same length as seq, initially all False

    if len(rpeaks) > 0:  # If there are detected peaks
        # Randomly select 'num_peaks_to_mask' peaks from rpeaks
        selected_peaks = rpeaks[1:2*num_peaks_to_mask + 1: 2]

        for peak in selected_peaks:
            mask_half = mask_length // 2
            start = int(max(0, peak - mask_half))
            end = int(min(len(seq), peak + mask_half))
            mask[start:end] = 0
    
    return mask

class ECGDataset(Dataset):
    def __init__(self, data, seq_len=1000, mask_length=30, target_rate=100, num_peaks_to_mask=1):
        """
        Args:
            data (np.array): Full ECG signal data.
            seq_len (int): Length of each sequence before downsampling.
            mask_length (int): Length of masking window around the first R-peak.
            target_rate (int): Target sampling rate after downsampling.
            augment (bool): Whether to apply data augmentation.
        """
        self.data = data
        self.seq_len = seq_len
        self.mask_length = mask_length
        self.target_rate = target_rate
        self.num_peaks_to_mask = num_peaks_to_mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx].flatten()
        # Detect R-peaks
        rpeaks = nk.ecg_findpeaks(seq, sampling_rate=self.target_rate)['ECG_R_Peaks']

        # Create masking
        mask_r_peak = fixed_position_mask_peaks(rpeaks, seq, self.mask_length, self.num_peaks_to_mask)
        mask_p_t_peak = fixed_position_mask_P_T(rpeaks, seq, 15, self.num_peaks_to_mask)

      #  masked_seq = seq.copy()
        # masked_seq[mask] = 0  # Apply masking

        return (
            torch.tensor(seq, dtype=torch.float32),
            torch.tensor(mask_r_peak, dtype=torch.bool),
            torch.tensor(mask_p_t_peak, dtype=torch.bool)
        )


# Data preparation
def prepare_data(data, seq_len, num_rpeaks, mask_len, batch_size=128):
    dataset = ECGDataset(data, seq_len, mask_length = mask_len, num_peaks_to_mask = num_rpeaks)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

class Trainer:
    def __init__(self, model, criterion, optimizer, seq_len=5000, num_rpeak=1, log_dir=None, masking_length = 30,scale = 'minmax_0_5',test_case = True):
        self.seq_len = seq_len
        self.num_rpeak = num_rpeak
        self.mask_len = masking_length
        self.scale = scale
        self.device = ( "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        self.log_dir = log_dir
        self.test_case = test_case
        self.cur_time = datetime.now(ZoneInfo("Australia/Sydney")).strftime("%H%M_%d%m%Y")
        if self.log_dir is None:
            self.log_dir = f'runs/seq{seq_len}_rpeak{num_rpeak}_{self.cur_time}'
        if self.test_case == False:
            self.writer = SummaryWriter(log_dir=self.log_dir)

    def run(self, train_loader, val_loader, test_loader, epochs=100, patience = 10):
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for original_seq, mask_qrs, mask_pt in train_loader:
                original_seq, mask_qrs,mask_pt = original_seq.to(self.device), mask_qrs.to(self.device), mask_pt.to(self.device)

                self.optimizer.zero_grad()
                reconstructed = self.model(original_seq, mask_qrs, mask_pt)

                loss_masked = self.criterion(reconstructed[mask_qrs + mask_pt], original_seq[mask_qrs + mask_pt])
                # full reconstruction loss to preserve unmasked regions (small weight)
                loss_full = self.criterion(reconstructed, original_seq)
                
                # combine: favor masked loss but still penalize drifting on unmasked
                alpha = 1.0   # weight for masked-target objective
                beta  = 0.1  # small weight for full-signal reconstruction
                loss = alpha * loss_masked + beta * loss_full


                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)

            avg_val_loss, avg_l1, avg_l2, avg_MAPE = self.evaluate(epoch=epoch, data_loader = val_loader)
            self.val_losses.append(avg_val_loss)

            print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                  f"L1 Accuracy: {avg_l1:.2f}, L2 Accuracy: {avg_l2:.2f}, MAPE Accuracy: {avg_MAPE:.2f}")
            if self.test_case == False:
                self.writer.add_scalar("Loss/Train", avg_train_loss, epoch)
                self.writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
                self.writer.add_scalar("Accuracy/L1", avg_l1, epoch)
                self.writer.add_scalar("Accuracy/L2", avg_l2, epoch)
                self.writer.add_scalar("Accuracy/MAPE", avg_MAPE, epoch)

            if epoch % 10 == 0:
                print(f"{self.log_dir}/model_{self.seq_len}_{self.num_rpeak}_{self.cur_time}.pth")
                if self.test_case == False:
                    print("Saving the best model checkpoint...")
                    torch.save(self.model.state_dict(), f"{self.log_dir}/model_{self.seq_len}_{self.num_rpeak}_{self.cur_time}.pth")

        self.writer.close()
        if test_loader is None:
            self.final_evaluation(val_loader)
        else: 
            self.final_evaluation(test_loader)
            
    def evaluate(self, data_loader, epoch= None):
        self.model.eval()
        val_loss = 0.0
        l1_accuracy = 0.0
        l2_accuracy = 0.0
        MAPE_accuracy = 0.0

        with torch.no_grad():
            for original_seq, mask_qrs, mask_pt in data_loader:
                original_seq, mask_qrs,mask_pt = original_seq.to(self.device), mask_qrs.to(self.device), mask_pt.to(self.device)
                reconstructed = self.model(original_seq, mask_qrs, mask_pt)
                
                loss_qrs = self.criterion(reconstructed[mask_qrs], original_seq[mask_qrs])
                loss_pt = self.criterion(reconstructed[mask_pt], original_seq[mask_pt])
                loss_masked = 0.7 * loss_qrs + 0.3 * loss_pt
                # full reconstruction loss to preserve unmasked regions (small weight)
                loss_full = self.criterion(reconstructed, original_seq)
                
                # combine: favor masked loss but still penalize drifting on unmasked
                alpha = 1.0   # weight for masked-target objective
                beta  = 0.1  # small weight for full-signal reconstruction
                loss = alpha * loss_masked + beta * loss_full
                
                val_loss += loss.item()
                l1_accuracy += calculate_l1(reconstructed[mask_qrs+mask_pt], original_seq[mask_qrs+mask_pt])
                l2_accuracy += calculate_l2(reconstructed[mask_qrs+mask_pt], original_seq[mask_qrs+mask_pt])
                MAPE_accuracy += calculate_MAPE(reconstructed[mask_qrs+mask_pt], original_seq[mask_qrs+mask_pt])

        avg_val_loss = val_loss / len(data_loader)
        avg_l1 = l1_accuracy / len(data_loader)
        avg_l2 = l2_accuracy / len(data_loader)
        avg_MAPE = MAPE_accuracy / len(data_loader)

        if epoch is None:
            print(f"\nValidation Loss: {avg_val_loss:.4f}")
            print(f"L1 Accuracy: {avg_l1:.4f}")
            print(f"L2 Accuracy: {avg_l2:.4f}")
            print(f"MAPE Accuracy: {avg_MAPE:.2f}%")
        return avg_val_loss, avg_l1, avg_l2, avg_MAPE
        
    def plot_predictions_with_residuals(self, data_loader, sample_len = None, num_samples=2 ):
        self.model.eval()
        all_actual, all_predicted, all_masks = [], [], []
        if sample_len is None:
            sample_len = self.seq_len
        
        with torch.no_grad():
            for original_seq, mask_qrs, mask_pt in data_loader:
                original_seq, mask_qrs,mask_pt = original_seq.to(self.device), mask_qrs.to(self.device), mask_pt.to(self.device)
                reconstructed = self.model(original_seq, mask_qrs, mask_pt)
                mask = mask_qrs + mask_pt
                all_actual.extend(original_seq.cpu().numpy())  # original (unmasked) ECG
                all_predicted.extend(reconstructed.cpu().numpy())  # predictions
                all_masks.extend(mask.cpu().numpy()) # Masked regions
     
                break  # Take only one batch for plotting
        
        # Flatten the lists for metric calculation
        all_actual_flat = np.concatenate(all_actual)
        all_predicted_flat = np.concatenate(all_predicted)
        
        # Calculate metrics
        mse = mean_squared_error(all_actual_flat, all_predicted_flat)
        mae = mean_absolute_error(all_actual_flat, all_predicted_flat)
        r2 = r2_score(all_actual_flat, all_predicted_flat)
    
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R-squared (R²): {r2:.4f}")
        
        # Visualize predictions vs. actual data with residuals
        plt.figure(figsize=(15, num_samples * 5))
        for i in range(min(num_samples, len(all_actual))):
            original = all_actual[i]
            predicted = all_predicted[i]
            mask = all_masks[i]
    
            # Plot predictions vs actual data
            plt.subplot(num_samples, 2, i * 2 + 1)
            plt.plot(original[:sample_len], label="Original", color="blue", alpha=0.7)
            plt.plot(predicted[:sample_len], label="Predicted", color="orange", alpha=0.7)
    
            # Highlight masked regions
            masked_indices = np.where(mask)[0]
            plt.scatter(masked_indices, original[masked_indices], color="red", label="Masked Regions", alpha=0.7)
    
            plt.title(f"Sample {i + 1} - Predictions")
            plt.legend()
    
            # Compute and plot residuals
            residuals = np.array(original) - np.array(predicted)
            plt.subplot(num_samples, 2, i * 2 + 2)
            plt.plot(residuals[:sample_len], label="Residuals", color="green")
            plt.axhline(0, color="black", linestyle="--")
            plt.title(f"Sample {i + 1} - Residuals")
            plt.legend()
    
        plt.tight_layout()
        plt.show()

    def plot_loss_curves(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label="Train Loss", marker='o')
        plt.plot(self.val_losses, label="Validation Loss", marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curves")
        plt.legend()
        plt.grid()
        plt.show()
        
    def final_evaluation(self, data_loader):
        avg_val_loss, avg_l1, avg_l2, avg_MAPE = self.evaluate(data_loader, epoch= None)
        with open('runs/Model_Running.csv', 'a') as f_object:
            writer_object = csv.writer(f_object)
            writer_object.writerow([f'model_{self.seq_len}_{self.num_rpeak}_{self.cur_time}.pth', avg_l1.cpu().numpy(), avg_l2.cpu().numpy(), avg_MAPE.cpu().numpy(), self.seq_len, self.num_rpeak, self.mask_len, self.scale])
            f_object.close()  
        self.plot_predictions_with_residuals(data_loader, sample_len = 1000)
        self.plot_loss_curves()

def plot_some_sample(data_loader, device, num_samples=5):
    all_actual, all_masks = [], []
    with torch.no_grad():
        for masked_seq, original_seq, mask in data_loader:
            all_actual.extend(original_seq)  # original (unmasked) ECG
            all_masks.extend(mask) # Masked regions
            break
            
    plt.figure(figsize=(15, num_samples * 5))
    for i in range(min(num_samples, len(all_actual))):
        original = all_actual[i]
        mask = all_masks[i]

        # Plot predictions vs actual data
        plt.subplot(num_samples, 2, i * 2 + 1)
        plt.plot(original, label="Original", color="blue", alpha=0.7)

        # Highlight masked regions
        masked_indices = np.where(mask)[0]
        plt.scatter(masked_indices, original[masked_indices], color="red", label="Masked Regions", alpha=0.7)

        plt.title(f"Sample {i + 1} - Predictions")
        plt.legend()    

def plot_ecg_sample_no_mask(data, num_samples=2):
    # Visualize predictions vs. actual data with residuals
    fig, ax = plt.subplots(1, num_samples, figsize = (16,9))
    ax = ax.flatten()
    for i in range(0, num_samples):
        # Plot predictions vs actual data
        ax[i].plot(data[i], label="Original", color="blue", alpha=0.7)
    plt.tight_layout()
    plt.show()