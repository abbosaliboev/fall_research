# models_full_kp.py
import torch
import torch.nn as nn


class TCN_Attention_Model_Full(nn.Module):
    def __init__(self, input_size=34, num_classes=2):
        super(TCN_Attention_Model_Full, self).__init__()
        # input_size = 34 (17 keypoints * 2)
        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Multi-Head Self-Attention
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: [Batch, Window, Features] -> [B, 30, 34]
        x = x.transpose(1, 2) # [B, 34, 30]
        x = self.tcn(x)
        x = x.transpose(1, 2) # [B, 30, 128]
        
        attn_out, _ = self.attention(x, x, x)
        
        # Global temporal pooling (oxirgi kadr orqali)
        out = self.fc(attn_out[:, -1, :]) 
        return out