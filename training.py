# import os
# import re
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler
# from torch.nn.utils.parametrizations import weight_norm
# from torch.nn.utils.rnn import pad_sequence
#
# # --- 1. Robust Natural Sorting ---
# # This ensures 1.csv, 2.csv, 10.csv stay in order even if numbers are missing
# def natural_sort_key(s):
#     return [int(text) if text.isdigit() else text.lower()
#             for text in re.split('([0-9]+)', s)]
#
# # --- 2. Model: TCN with Global Adaptive Pooling ---
# class TCNBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, dilation):
#         super(TCNBlock, self).__init__()
#         # Padding is calculated to keep the sequence length consistent through layers
#         padding = (kernel_size - 1) * dilation
#         self.conv = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
#                                          stride=1, padding=padding, dilation=dilation))
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.2)
#         # Chomp keeps the convolution "causal" (output doesn't depend on future data)
#         self.chomp = lambda x: x[:, :, :-padding] if padding > 0 else x
#         self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
#         self.net = nn.Sequential(self.conv, self.relu, self.dropout)
#
#     def forward(self, x):
#         res = x if self.downsample is None else self.downsample(x)
#         return self.relu(self.chomp(self.net(x)) + res)
#
# class TCNModel(nn.Module):
#     def __init__(self, input_size, num_channels, kernel_size=3, output_size=1):
#         super(TCNModel, self).__init__()
#         layers = []
#         for i in range(len(num_channels)):
#             dilation_size = 2 ** i
#             in_ch = input_size if i == 0 else num_channels[i-1]
#             out_ch = num_channels[i]
#             layers += [TCNBlock(in_ch, out_ch, kernel_size, dilation_size)]
#
#         self.network = nn.Sequential(*layers)
#
#         # KEY: Adaptive Pooling handles the "different number of rows" issue
#         # It squashes any sequence length down to a fixed size of 1
#         self.global_pool = nn.AdaptiveMaxPool1d(1)
#         self.fc = nn.Linear(num_channels[-1], output_size)
#
#     def forward(self, x):
#         # x: [Batch, Time, Features] -> Transpose for Conv1d: [Batch, Features, Time]
#         x = x.transpose(1, 2)
#         y = self.network(x)
#         y = self.global_pool(y).squeeze(-1) # Output is now [Batch, Features]
#         return self.fc(y)
#
# # --- 3. Data Loading Logic ---
# class ADCDataset(Dataset):
#     def __init__(self, data_dir, annotation_file):
#         self.data_dir = data_dir
#
#         # Load labels from text file
#         with open(annotation_file, 'r') as f:
#             self.labels = [float(line.strip()) for line in f.readlines() if line.strip()]
#
#         # Get and sort files naturally
#         all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
#         self.csv_files = sorted(all_files, key=natural_sort_key)
#
#         # Sync counts
#         self.count = min(len(self.csv_files), len(self.labels))
#
#         # Scale ADC and Freq to be between 0 and 1 (highly recommended for TCNs)
#         self.scaler = StandardScaler()
#         # Fit on first 5 files to get a good range
#         combined_samples = pd.concat([pd.read_csv(os.path.join(data_dir, f)) for f in self.csv_files[:5]])
#         self.scaler.fit(combined_samples[['freq_hz', 'adc']].values)
#
#     def __len__(self):
#         return self.count
#
#     def __getitem__(self, idx):
#         df = pd.read_csv(os.path.join(self.data_dir, self.csv_files[idx]))
#         # Features: freq_hz and adc
#         features = self.scaler.transform(df[['freq_hz', 'adc']].values)
#         label = self.labels[idx]
#         return torch.FloatTensor(features), torch.FloatTensor([label])
#
# # --- 4. Collate Function to handle different row counts ---
# def pad_collate(batch):
#     (xx, yy) = zip(*batch)
#     # Automatically adds zeros to the end of shorter sequences
#     xx_pad = pad_sequence(xx, batch_first=True, padding_value=0.0)
#     yy_stack = torch.stack(yy)
#     return xx_pad, yy_stack
#
# # --- 5. Training Loop ---
# def train():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     dataset = ADCDataset('data', 'annotation/data.txt')
#     train_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=pad_collate)
#
#     # input_size=2 (freq_hz, adc)
#     model = TCNModel(input_size=2, num_channels=[32, 64, 128]).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     criterion = nn.MSELoss()
#
#     print(f"--- Dataset: {len(dataset)} samples ---")
#     print(f"Training on {device}...")
#
#     for epoch in range(100):
#         model.train()
#         total_loss = 0
#         for seq, label in train_loader:
#             seq, label = seq.to(device), label.to(device)
#
#             optimizer.zero_grad()
#             output = model(seq)
#             loss = criterion(output, label)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#
#         if (epoch + 1) % 10 == 0:
#             print(f"Epoch {epoch+1:03} | Loss: {total_loss/len(train_loader):.6f}")
#
#     torch.save(model.state_dict(), "best_capstone_model.pth")
#     print("Training finished! Model saved.")
#
# if __name__ == "__main__":
#     train()

# Add this import at the top
import joblib
import os
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.rnn import pad_sequence

# --------------------------------------------------
# 1. Robust Natural Sorting
# --------------------------------------------------
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

# --------------------------------------------------
# 2. TCN Model with Global Adaptive Pooling
# --------------------------------------------------
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation

        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation
            )
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.chomp = lambda x: x[:, :, :-padding] if padding > 0 else x
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )

        self.net = nn.Sequential(
            self.conv,
            self.relu,
            self.dropout
        )

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        out = self.net(x)
        out = self.chomp(out)
        return self.relu(out + res)


class TCNModel(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=3, output_size=1):
        super(TCNModel, self).__init__()

        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_ch = input_size if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation))

        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x: [Batch, Time, 1]
        x = x.transpose(1, 2)       # -> [Batch, 1, Time]
        x = self.network(x)
        x = self.global_pool(x)     # -> [Batch, Channels, 1]
        x = x.squeeze(-1)           # -> [Batch, Channels]
        return self.fc(x)

# --------------------------------------------------
# 3. Dataset (ONLY ADC)
# --------------------------------------------------
class ADCDataset(Dataset):
    def __init__(self, data_dir, annotation_file):
        self.data_dir = data_dir

        # Load labels
        with open(annotation_file, 'r') as f:
            self.labels = [
                float(line.strip())
                for line in f.readlines()
                if line.strip()
            ]

        # Sort CSV files naturally
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        self.csv_files = sorted(all_files, key=natural_sort_key)

        self.count = min(len(self.csv_files), len(self.labels))

        # Scale ADC values
        self.scaler = StandardScaler()

        # Fit scaler using first few files
        sample_dfs = [
            pd.read_csv(os.path.join(data_dir, f))
            for f in self.csv_files[:5]
        ]
        combined = pd.concat(sample_dfs)
        self.scaler.fit(combined[['adc']].values)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        df = pd.read_csv(os.path.join(self.data_dir, self.csv_files[idx]))
        adc_values = df[['adc']].values
        adc_scaled = self.scaler.transform(adc_values)

        label = self.labels[idx]

        return (
            torch.FloatTensor(adc_scaled),
            torch.FloatTensor([label])
        )

# --------------------------------------------------
# 4. Collate Function (Variable Length Sequences)
# --------------------------------------------------
def pad_collate(batch):
    sequences, labels = zip(*batch)
    sequences_padded = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0.0
    )
    labels = torch.stack(labels)
    return sequences_padded, labels

# --------------------------------------------------
# 5. Training Loop
# --------------------------------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ADCDataset(
        data_dir="data",
        annotation_file="annotation/data.txt"
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=pad_collate
    )

    model = TCNModel(
        input_size=1,                # ONLY ADC
        num_channels=[32, 64, 128]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print(f"Dataset size: {len(dataset)}")
    print(f"Training on: {device}")

    for epoch in range(100):
        model.train()
        total_loss = 0.0

        for seq, label in loader:
            seq = seq.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), "best_capstone_model.pth")
    print("Training complete. Model saved.")

    # ✅ Save scaler (FIX)
    joblib.dump(dataset.scaler, "scaler.pkl")
    print("Scaler saved as scaler.pkl")


# --------------------------------------------------
if __name__ == "__main__":
    train()
