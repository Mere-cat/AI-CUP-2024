try:
    from Pre_data import BaseDataReader, BaseGeoDataReader
except ImportError:
    from .Pre_data import BaseDataReader, BaseGeoDataReader

from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.loader import DataLoader

import torch
import numpy as np
import pandas as pd
import os

dataPath = __file__.split(os.sep)[:-2]
dataPath = os.path.join(os.sep, *dataPath, "data")
fin_Path = "TrainingData_fin"
eps = 1e-8

edge_index = BaseGeoDataReader().edge_index

def normalizeScale(df, scale=None):
    if scale is None:
        scale = (1.0 / (df.abs().max() + eps)) * 0.999999
    else:
        scale = (1.0 / (scale + eps)) * 0.999999
    return df * scale, scale


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, day=True, idx=range(1, 18)):
        self.day = day
        self.idx = idx
        self._ready_flow()

    def _ready_flow(self):
        self._read_csv()
        self._full_augmentation()
        self._build_full_data()
        self._len()

    def _len(self):
        self.index = self.data["Date"].dropna().unique() if self.day else self.data.index
        self.len =  len(self.index)

    def _read_csv(self):
        Ls = {i: self._read_sigle_csv(i) for i in self.idx}
        self.X = {key: x for key, (x, y) in Ls.items()}
        self.Y = {key: y for key, (x, y) in Ls.items()}

    def _build_full_data(self):
        _x = list(self.X.values())
        _y = list(self.Y.values())

        for idx, i in enumerate(_x):
            i.rename(columns={col: f"{idx + 1}_{col}" for col in i.columns}, inplace=True)

        for idx, i in enumerate(_y):
            i.rename(columns={col: f"{idx + 1}" for col in i.columns}, inplace=True)

        x = _x[0]
        self.dim = x.shape[1] - 2

        for i in _x[1:]:
            x = pd.merge(x, i, left_index=True, right_index=True, how="outer")

        y = _y[0]
        for i in _y[1:]:
            y = pd.merge(y, i, left_index=True, right_index=True, how="outer")

        x = self._integrate_date(x, "DateTime")
        x = self._integrate_date(x, "Date")

        mask = y.isna().sum(axis=1) < 15

        self.data = x.loc[mask]
        self.Y = y.loc[mask]

        new_columns = ["DateTime", "Date", *self.data.drop(columns=["DateTime", "Date"]).columns]
        self.data = self.data[new_columns]

    def _full_augmentation(self):
        full = pd.concat(self.X.values())
        x_max = full.iloc[:, 2:-5].abs().max()
        y_max = full["Power"].abs().max()

        for key in self.X.keys():
            self.X[key].iloc[:, 2:-5], _ = normalizeScale(self.X[key].iloc[:, 2:-5], x_max)

            self.X[key]["Power"], self.M = normalizeScale(self.X[key]["Power"], y_max)
            self.Y[key]["Power"], _ = normalizeScale(self.Y[key]["Power"], y_max)

    def __getitem__(self, item):
        ration = torch.rand((1,)) * 0.5
        if self.day:
            itemIdx = self.data["Date"] == self.index[item]
            data = self.data[itemIdx]
            Y = self.Y[itemIdx]
            offline = Y.iloc[15:].isna().sum(axis=0) > 40

        else:
            data = self.data.iloc[[item]]
            Y = self.Y.iloc[item]
            offline = Y.isna()

        online_idx = list(map(int, offline.index[~offline.values]))

        data = data.drop(columns=["DateTime", "Date"])

        mask_idx = np.random.choice(online_idx,
                                    size=max(int(len(online_idx) * ration), 1),
                                    replace=False) - 1
        if self.day:
            data = data.fillna(0).values.reshape(data.shape[0], -1, self.dim)
            Y = Y.fillna(0).values.reshape(data.shape[0], -1)
            out = [self.getitem_step(x, y, mask_idx, offline) for x, y in zip(data, Y)]
            return out

        data = data.fillna(0).values.reshape(-1, self.dim)
        Y = Y.fillna(0).values
        return self.getitem_step(data, Y, mask_idx, offline)

    def __len__(self):
        return self.len

    @staticmethod
    def _read_sigle_csv(csv_idx):
        df = pd.read_csv(os.path.join(dataPath, fin_Path, f"L{csv_idx}_Train_avg.csv"))
        return df, df[["Power"]]

    @staticmethod
    def _integrate_date(df, name):
        DateTimes = [i for i in df.columns if f"_{name}" in i]
        df[name] = df[DateTimes].bfill(axis=1).iloc[:, 0]
        df.drop(columns=DateTimes, inplace=True)
        return df

    @staticmethod
    def getitem_step(x, y, mask_idx, offline):
        mask = torch.zeros(len(y), dtype=torch.bool)
        mask[mask_idx] = True


        x[mask] = x[~mask & ~offline.values].mean(axis=0)

        return Data(x=torch.tensor(x, dtype=torch.float32),
                    y=torch.tensor(y, dtype=torch.float32),
                    edge_index=edge_index,
                    mask=mask
                    )

    def random_mask(self, data, ratio):
        mask = torch.rand((data.shape[0], 1)) < ratio
        return mask

# 创建模型
if __name__ == "__main__":
    from tqdm import tqdm
    data = BaseDataset(day=True)
    loader = DataLoader(data, batch_size=1, shuffle=True)

    for seq_data in tqdm(data):
        output = model(seq_data)

