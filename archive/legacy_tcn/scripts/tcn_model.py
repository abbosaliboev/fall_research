# scripts/tcn_model.py
import torch
import torch.nn as nn

class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, dil=1, drop=0.2):
        super().__init__()
        pad = ((k - 1) * dil) // 2  # same-length padding
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=k, dilation=dil, padding=pad),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv1d(c_out, c_out, kernel_size=k, dilation=dil, padding=pad),
            nn.ReLU(),
            nn.Dropout(drop),
        )
        self.proj = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.proj(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size=3, num_channels=[64, 128, 256, 256], drop=0.2):
        super().__init__()
        layers = []
        c = input_size
        dilations = [1, 2, 4, 8]
        for out_c, d in zip(num_channels, dilations):
            layers.append(TCNBlock(c, out_c, k=3, dil=d, drop=drop))
            c = out_c
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(c, output_size)
        )

    def forward(self, x):  # x: [B, C, L]
        y = self.tcn(x)
        y = self.head(y)
        return y
