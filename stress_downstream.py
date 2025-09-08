from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from utlis.wesad_processing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd
import torch.optim as optim
import pickle
import os
import csv 
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from utlis.finetune_wesad_data import load_pre_wesad_dataset, process_wesad_dataset

class DownstreamClassifier(nn.Module):
    def __init__(self, encoder, num_classes=2):
        super(DownstreamClassifier, self).__init__()
        self.encoder = encoder

        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Classifier head after GAP
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),  # Adjust 256 to match your encoder's output channels
            nn.LeakyReLU(negative_slope = 0.01),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.squeeze(1)         # (B, L)

        # Pass through encoder
        z = self.encoder(x)  # -> (B, C, L)

        # Global Average Pooling over sequence dimension
        z = F.adaptive_avg_pool1d(z, 1)  # -> (B, C, 1)
        z = z.squeeze(-1)                # -> (B, C)

        logits = self.classifier(z)      # -> (B, num_classes)
        return logits
    
class Conv_Block(nn.Module):
    def __init__(self, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(1, embed_dim, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(embed_dim, embed_dim * 2, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

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
        self.ConvBlock = Conv_Block(embed_dim, hidden_dim)
        self.bilstm = nn.LSTM(hidden_dim, lstm_hidden_dim,
                              bidirectional=True, batch_first=True)

    def forward(self, x):
        # Encode each branch
        x = self.ConvBlock(x)
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

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x.squeeze(1)


class ECGClassificationDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X  # shape: [N, L]
        self.Y = Y  # shape: [N]
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        seq = self.X[idx].flatten()
        return torch.tensor(seq, dtype=torch.float32).unsqueeze(0), torch.tensor(self.Y[idx], dtype=torch.long)


def evaluate(model,criterion, optimizer, dataloader, device):
 
    # --- Validation ---
    model.eval()
    
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for x_val, y_val in dataloader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            outputs = model(x_val)
            loss = criterion(outputs, y_val)
            
            preds = torch.argmax(outputs, dim=1)
            all_val_preds.extend(preds.cpu().numpy())
            all_val_labels.extend(y_val.cpu().numpy())
    
    prec_temp = precision_score(all_val_labels, all_val_preds)
    rec_temp = recall_score(all_val_labels, all_val_preds)
    f1_temp = f1_score(all_val_labels, all_val_preds,average='macro')
    acc_temp = accuracy_score(all_val_labels, all_val_preds)
    
    return prec_temp, rec_temp, f1_temp, acc_temp


def train_evaluate_downstream_classifier(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    test_loader,
    num_epochs=100,
    device = 'cuda'
):

    model.to(device)
    val_losses = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        all_train_preds = []
        all_train_labels = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(y.cpu().numpy())

        train_accuracy = accuracy_score(all_train_labels, all_train_preds)
        train_f1_macro = f1_score(all_train_labels, all_train_preds, average='macro')

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                outputs = model(x_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(y_val.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_val_labels, all_val_preds)
        val_f1_macro = f1_score(all_val_labels, all_val_preds, average='macro')
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch}: "
              f"Train Loss = {total_loss/len(train_loader):.4f}, "
              f"Train Acc = {train_accuracy:.4f}, "
              f"Train F1 = {train_f1_macro:.4f} | "
              f"Val Loss = {avg_val_loss:.4f}, "
              f"Val Acc = {val_accuracy:.4f}, "
              f"Val F1 = {val_f1_macro:.4f}")
        
    prec, rec, f1, acc = evaluate( model = model, criterion = criterion,  optimizer = optimizer, dataloader = test_loader, device = 'cuda')
    print(f"Train F1 = {f1:.4f} - Average = {acc:.4f}")
    return prec, rec, f1, acc




def loso_training(root_dir,model_path, test_size, filename):
    prec, rec, acc, f1 = [], [], [], []
    folder_ls = [f for f in os.listdir(root_dir) if f not in (".ipynb_checkpoints", ".DS_Store")]
    start_index = np.arange(0, len(folder_ls) - test_size + 1, test_size)

    for start in start_index:
        subj = folder_ls[start:start + test_size]
        print(f"***** Loop {start}: {subj} *****")

        # Load dataset
        X_train, X_test, y_train, y_test = load_pre_wesad_dataset(root_dir, subj)
        train_dataset = ECGClassificationDataset(X_train, y_train)
        test_dataset = ECGClassificationDataset(X_test, y_test)

        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        loaders = {
            'train': DataLoader(train_dataset, batch_size=128, shuffle=True),
            'val': DataLoader(val_dataset, batch_size=128, shuffle=False),
            'test': DataLoader(test_dataset, batch_size=128, shuffle=False)
        }

        # Train and evaluate downstream classifier
            # Load pretrained model once
        auto_model = MAE1D_Mask()
        auto_model.load_state_dict(torch.load(
            model_path, 
            weights_only=True
        ))
        encoder = auto_model.encoder
        model = DownstreamClassifier(encoder)
                # Split train/val
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
        prec_, rec_, f1_, acc_ = train_evaluate_downstream_classifier(
            model,criterion, optimizer, loaders['train'], loaders['val'], loaders['test'], num_epochs=100, device='cuda'
        )
        prec.append(prec_); rec.append(rec_); f1.append(f1_); acc.append(acc_)
        print([start, acc_, f1_, rec_, prec_])

        # Save iteration results
        with open(filename, 'a', newline='') as f:
            csv.writer(f).writerow([start, acc_, f1_, rec_, prec_, subj])

    # Save mean results
    with open(filename, 'a', newline='') as f:
        csv.writer(f).writerow([np.mean(acc), np.mean(f1), np.mean(prec), np.mean(rec)])

    print(f"Accuracy: {np.mean(acc)}, F1: {np.mean(f1)}, Precision: {np.mean(prec)}, Recall: {np.mean(rec)}")
    return acc, f1, rec, prec
