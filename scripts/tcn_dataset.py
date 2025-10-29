# -*- coding: utf-8 -*-
# scripts/tcn_dataset.py
import re
import pandas as pd
import torch
from torch.utils.data import Dataset

class SequenceCSVDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

        # faqat f0, f1, f2, ... ustunlarini oling (feat_dim kabi boshqa 'f...' larni emas)
        feat_cols = [c for c in self.df.columns if re.fullmatch(r"f\d+", c)]
        if not feat_cols:
            raise ValueError("Feature ustunlar topilmadi (f0, f1, ...). CSV formatini tekshiring.")
        feat_cols = sorted(feat_cols, key=lambda x: int(x[1:]))  # f0,f1,...

        self.feat_cols = feat_cols
        self.X = self.df[self.feat_cols].values.astype("float32")

        # label_id ustuni bo'lishi kerak (0=no_fall,1=pre_fall,2=fall)
        if "label_id" not in self.df.columns:
            raise ValueError("CSVda 'label_id' ustuni yo'q. generate_sequences.py chiqishini tekshiring.")
        self.y = self.df["label_id"].values.astype("int64")

        # seq_len/feat_dim ni CSVdan olish (har qatorda bir xil bo'lishi kerak)
        if "seq_len" not in self.df.columns or "feat_dim" not in self.df.columns:
            raise ValueError("CSVda 'seq_len' va 'feat_dim' ustunlari bo‘lishi shart.")

        self.seq_len = int(self.df["seq_len"].iloc[0])
        self.feat_dim = int(self.df["feat_dim"].iloc[0])

        # xavfsizlik: f-ustunlar soni seq_len*feat_dim ga tengligini tekshiring
        expected = self.seq_len * self.feat_dim
        if len(self.feat_cols) != expected:
            raise ValueError(
                f"Feature ustunlar soni mos emas: {len(self.feat_cols)} != seq_len*feat_dim ({expected}). "
                "generate_sequences.py parametrlarini tekshiring."
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.X[idx]  # [seq_len*feat_dim]
        # [seq_len, feat_dim] -> [C=feat_dim, L=seq_len] (TCN uchun)
        x = x.reshape(self.seq_len, self.feat_dim).T
        x = torch.from_numpy(x)  # float32 tensor
        y = torch.tensor(self.y[idx])  # int64 tensor
        return x, y
