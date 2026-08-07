import torch
import torch.nn as nn

class TCN_Attention_Model(nn.Module):
    def __init__(self, input_size=20, num_classes=2, num_channels=[64, 128, 256], kernel_size=3, dropout=0.2):
        super(TCN_Attention_Model, self).__init__()
        layers = []
        in_ch = input_size
        for out_ch in num_channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=(kernel_size-1)//2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        
        # Paperdagi Transformer o'rniga yengil Attention
        self.attention = nn.MultiheadAttention(embed_dim=num_channels[-1], num_heads=4, batch_first=True)
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        # x: [batch, 30, 20] -> [batch, 20, 30]
        x = x.transpose(1, 2)
        x = self.tcn(x)
        # x: [batch, 256, 30] -> [batch, 30, 256]
        x = x.transpose(1, 2)
        
        attn_out, _ = self.attention(x, x, x)
        out = attn_out[:, -1, :] # Oxirgi kadr xulosasi
        return self.fc(out)