# dataset.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class FallDataset(Dataset):
    def __init__(self, csv_file, window_size=30, step_size=5):
        self.df = pd.read_csv(csv_file)
        self.window_size = window_size
        self.step_size = step_size
        self.features, self.labels = self.create_windows()

    def create_windows(self):
        X, y = [], []
        activities = self.df['activity'].unique()
        
        for act in activities:
            temp_df = self.df[self.df['activity'] == act]
            kp_data = temp_df.iloc[:, :20].values
            frame_labels = temp_df['label'].values 
            
            if len(kp_data) < self.window_size:
                continue

            for i in range(0, len(kp_data) - self.window_size, self.step_size):
                window = kp_data[i : i + self.window_size]
                # Oynaning oxirgi kadri labelini olish
                label = 1 if frame_labels[i + self.window_size - 1] == 1 else 0
                
                X.append(window)
                y.append(label)
                
                # Yiqilishni 5 marta ko'paytiramiz (Oversampling)
                if label == 1:
                    for _ in range(5):
                        X.append(window)
                        y.append(1)
        
        return np.array(X), np.array(y)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)