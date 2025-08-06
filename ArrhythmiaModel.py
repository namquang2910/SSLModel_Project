import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import wfdb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import neurokit2 as nk
from scipy.signal import resample, medfilt
import pywt
import time
import os

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

def load_multiple_records(record_ranges):
    all_data = []
    for start, end in record_ranges:
        for record_id in range(start, end + 1):
            record_data = load_record(record_id)
            # record_data = record_data.iloc[:, 0].to_frame()
            #print(record_data.head())
            all_data.append(record_data)
    
    combined_data = pd.concat(all_data, ignore_index=True)
    #print(combined_data.head())
    
    combined_data.to_csv(f'database.csv', index=False)
    return combined_data

def load_record(record_id):
    record_name = str(record_id)
    if not os.path.exists(f"mit_bih/{record_name}.dat"):
        wfdb.dl_database('mitdb', './', records=[record_name])
    record = wfdb.rdrecord('mit_bih/' + record_name)
    signal_data = record.p_signal
    flattened_signal = signal_data.flatten(order='F')
    
    data = pd.DataFrame(flattened_signal)
    
    return data

# Custom Dataset
class ECGDataset(Dataset):
    def __init__(self, data, seq_len=1000, mask_length=60, target_rate=100):
        """
        Args:
            data (np.array): Full ECG signal data.
            seq_len (int): Length of each sequence before downsampling.
            mask_length (int): Length of masking window around the first R-peak.
            original_rate (int): Original sampling rate of the data (default: 360 Hz).
            target_rate (int): Target sampling rate after downsampling (default: 100 Hz).
        """
        self.data = data
        self.seq_len = seq_len
        self.mask_length = mask_length
        self.target_rate = target_rate

    def __len__(self):
        # Calculate the number of sequences that can be extracted
        return (len(self.data) - self.seq_len) // self.target_rate

    def __getitem__(self, idx):
        # Adjust index for downsampled sequences
        start_idx = idx * self.target_rate
        seq = self.data[start_idx:start_idx + self.seq_len]

        # Detect R-peaks in the sequence
        rpeaks = nk.ecg_findpeaks(seq, sampling_rate=self.target_rate)['ECG_R_Peaks']

        # Create masking array
        mask = np.zeros_like(seq, dtype=bool)
        mask = random_mask_peaks(rpeaks, seq, mask_length=60, num_peaks_to_mask=10)

        masked_seq = seq.copy()
        masked_seq[mask] = 0  # Apply masking

        mask = torch.tensor(mask, dtype=torch.bool)
        return torch.tensor(masked_seq, dtype=torch.float32), torch.tensor(seq, dtype=torch.float32), mask


# Masked Autoencoder
class MAE1D(nn.Module):
    def __init__(self, seq_len, embed_dim=32, hidden_dim=64, lstm_hidden_dim=128):
        super(MAE1D, self).__init__()
        self.seq_len = seq_len
        self.encoder = nn.Sequential(
            nn.Conv1d(1, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(embed_dim, embed_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(embed_dim * 2, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # BiLSTM layer, processes sequence forwards and backwars
        self.bilstm = nn.LSTM(hidden_dim, lstm_hidden_dim, bidirectional=True, batch_first=True)

        self.decoder = nn.Sequential(
            nn.Conv1d(lstm_hidden_dim * 2, embed_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),

            nn.Conv1d(embed_dim * 2, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),

            nn.Conv1d(embed_dim, 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x, mask):
        # print(f"Input shape: {x.shape}")

        x = x.unsqueeze(1)
        # print(f"Shape after unsqueeze (adding channel): {x.shape}")

        encoded = self.encoder(x)
        # print(f"Shape after encoding: {encoded.shape}")

        encoded = encoded.permute(0, 2, 1)
        lstm_out, _ = self.bilstm(encoded)

        lstm_out = lstm_out.permute(0, 2, 1)
        decoded = self.decoder(lstm_out)

        x_reconstructed = decoded.squeeze(1)
        # print(f"Shape after squeeze: {x_reconstructed.shape}")

        return x_reconstructed
    

# Data preparation
def prepare_data(data, seq_len, batch_size=32, target_rate=100, original_rate=360, median_window_size=5):

    data = data.iloc[:, 0]
    data = data.to_numpy()

    # data = (data - np.min(data)) / (np.max(data) - np.min(data))

    coeffs = pywt.wavedec(data, 'db4', level=5) #Wavelet decomposition, reduces noise, enhances key elements
    data = pywt.waverec(coeffs, 'db4')

    data = medfilt(data, kernel_size=median_window_size) #Reduces noise further

    data = downsample(data, original_rate=original_rate, target_rate=target_rate) #Downsample form 360Hz to 100Hz, then feed sequences of 500Hz

    dataset = ECGDataset(data, seq_len)
    train_size = int(0.75 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# Training
def train_model_with_plot(model, train_loader, val_loader, epochs, patience=10, max_hours=2, lr=0.001, device='mps'):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []

    # last best validation loss variable, set to 0.
    best_val_loss = float('inf')
    # loss patience counter
    patience_counter = 0

    start_time = time.time()
    max_time = max_hours * 3600

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for masked_seq, original_seq, mask in train_loader:
            masked_seq, original_seq, mask = masked_seq.to(device), original_seq.to(device), mask.to(device)

            optimizer.zero_grad()
            reconstructed = model(masked_seq, mask)

            loss_global = criterion(reconstructed, original_seq)
            loss_masked = criterion(reconstructed[mask], original_seq[mask])
            loss = 0.9 * loss_masked + 0.1 * loss_global

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))

        val_loss = 0.0
        l1_accuracy = 0
        l2_accuracy = 0
        MAPE_accuracy = 0
        model.eval()
        with torch.no_grad():
            for masked_seq, original_seq, mask in val_loader:
                masked_seq, original_seq, mask = masked_seq.to(device), original_seq.to(device), mask.to(device)
                reconstructed = model(masked_seq, mask)
                masked_reconstructed = reconstructed[mask]
                masked_original = original_seq[mask]
                loss = criterion(masked_reconstructed, masked_original)
                val_loss += loss.item()
                l1_accuracy += calculate_l1(masked_reconstructed, masked_original)
                l2_accuracy += calculate_l2(masked_reconstructed, masked_original)
                MAPE_accuracy += calculate_MAPE(masked_reconstructed, masked_original)

        val_losses.append(val_loss / len(val_loader))
        avg_l1 = l1_accuracy / len(val_loader)
        avg_l2 = l2_accuracy / len(val_loader)
        avg_MAPE = MAPE_accuracy / len(val_loader)

        print(f"Epoch {epoch}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}, , L1 Accuracy: {avg_l1:.2f}, L2 Accuracy: {avg_l2:.2f}, MAPE Accuracy: {avg_MAPE:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

    # Visualize predictions vs. actual data
    plot_predictions_with_residuals(model, val_loader, device)
    plot_loss_curves(train_losses, val_losses)


def plot_predictions_with_residuals(model, data_loader, device, num_samples=5):
    model.eval()
    all_actual, all_predicted, all_masks = [], [], []
    
    with torch.no_grad():
        for masked_seq, original_seq, mask in data_loader:
            masked_seq, original_seq = masked_seq.to(device), original_seq.to(device)
            reconstructed = model(masked_seq, mask)
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
        plt.plot(original, label="Original", color="blue", alpha=0.7)
        plt.plot(predicted, label="Predicted", color="orange", alpha=0.7)

        # Highlight masked regions
        masked_indices = np.where(mask)[0]
        plt.scatter(masked_indices, original[masked_indices], color="red", label="Masked Regions", alpha=0.7)

        plt.title(f"Sample {i + 1} - Predictions")
        plt.legend()

        # Compute and plot residuals
        residuals = np.array(original) - np.array(predicted)
        plt.subplot(num_samples, 2, i * 2 + 2)
        plt.plot(residuals, label="Residuals", color="green")
        plt.axhline(0, color="black", linestyle="--")
        plt.title(f"Sample {i + 1} - Residuals")
        plt.legend()

    plt.tight_layout()
    plt.show()

def plot_loss_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid()
    plt.show()



record_ranges = [(100,109),(111, 119),(121,124),(200,203),(205,205),(207,210),(212,215),(217,217),(219,223),(228,228),(230,232)]
#record_ranges = [(100, 101)]
print("Loading the data")
data = load_multiple_records(record_ranges)
# data = pd.read_csv('100.csv')
# Instantiate and train the model
seq_len = 5000
print("Preparing data")
train_loader, val_loader = prepare_data(data, seq_len=seq_len)
mae_model = MAE1D(seq_len=seq_len)
print("cuda")
train_model_with_plot(mae_model, train_loader, val_loader, epochs=50, device='cuda')