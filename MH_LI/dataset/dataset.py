try:
    from read_upload import upload2datatime
    from base import BaseGeoDataReader
except ImportError:
    from .read_upload import upload2datatime
    from .base import BaseGeoDataReader


from torch_geometric.data import Data

import torch
import numpy as np
import pandas as pd
import os

dataPath = __file__.split(os.sep)[:-2]
dataPath = os.path.join(os.sep, *dataPath, "data")
fin_Path = "TrainingData_fin"
eps = 1e-6

def normalizeScale(df):
    scale = (1.0 / (df.abs().max() + eps)) * 0.999999
    return df * scale, scale

class BaseData(torch.utils.data.Dataset):
    def __init__(self):
        self._ready_flow()

    def _ready_flow(self, idx=range(1, 18)):
        self._read_csv(idx)
        self._find_OrgPoint()
        self._DataTime2DataIdx()
        self._Dropnan()
        self._count_len()
        self._Data2Index()

    def _read_csv(self, idx):
        self.Ls = {i: self._read_sigle_csv(i) for i in idx}
        self.Ms = {key: m for key, (x, y, m) in self.Ls.items()}
        self.upload = upload2datatime()

    def _find_OrgPoint(self):
        self.OrgPoint = pd.to_datetime(min([x["DateTime"].min() for x, y, m in self.Ls.values()]))

    def _DataTime2DataIdx(self):
        self.Ls = {key: (self._DateOffset(x), y, m) for key, (x, y, m) in self.Ls.items()}
        self.upload = self._DateOffset(self.upload)

    def _Dropnan(self):
        self.Ls = {keys: self._dropsingle(keys) for keys in self.Ls.keys()}

    def _count_len(self):
        _len = []
        for x, y in self.Ls.values():
            _len.extend(x["DateTime"].to_list())
        self.len = len(set(_len))

    def _Data2Index(self):
        self.L, self.Y = {}, {}
        for key, (x, y) in self.Ls.items():
            x.set_index('DateTime', inplace=True)
            y.index = x.index
            self.L[key] = x
            self.Y[key] = y

    def _dropsingle(self, idx):
        Lx, Ly, m = self.Ls[idx]
        upload = self.upload[self.upload["id"] == idx]
        nan_rows = Lx.drop(columns=["DateTime"]).eq(0).all(axis=1)
        upload_row = Lx["DateTime"].isin(upload["DateTime"])
        filter_rows = ~nan_rows | upload_row
        return Lx.loc[filter_rows], Ly.loc[filter_rows]

    def _DateOffset(self, df):
        DateTime = pd.to_datetime(df["DateTime"])
        df["DateTime"] = ((DateTime - self.OrgPoint).dt.total_seconds() // 600).astype(int)
        return df

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        pass

    def baseline(self):
        upload = upload2datatime()
        values = [self.Y[idx].mean() / self.Ms[idx] for idx in self.Y.keys()]
        for idx in self.Y.keys():
            upload.loc[upload["id"]==idx, "答案"] = values[idx-1]

        upload[["序號", "答案"]].to_csv("baseline.csv", index=False)

    @staticmethod
    def _read_sigle_csv(csv_idx):
        df = pd.read_csv(os.path.join(dataPath, fin_Path, f"L{csv_idx}_Train_avg.csv"))
        y, m = normalizeScale(df["Power"])
        df["Power"] = y
        return df, df[["Power"]], m


class GraphDataset(BaseData):
    def __init__(self):
        super().__init__()
        self._read_geo()
        self._add_geo()

        self._build_full_data()
        self._random_()


    def _random_(self):
        self.num_mask = torch.rand((self.len, )) * 0.5

    def _add_geo(self):
        geo = self.geo.node_geo
        for keys in self.L.keys():
            g = geo[geo["ID"] == keys]
            self.L[keys]["面朝"] = float(np.sin(g["面朝"] - np.pi / 100))
            self.L[keys]["高度"] = float(g["模擬高度"])

    def _build_full_data(self):
        _l = list(self.L.values())
        _y = list(self.Y.values())

        for idx, i in enumerate(_l):
            i.rename(columns={col: f"{idx + 1}_{col}" for col in i.columns}, inplace=True)

        for idx, i in enumerate(_y):
            i.rename(columns={col: f"{idx + 1}" for col in i.columns}, inplace=True)

        l = _l[0]
        self.dim = l.shape[1]
        for i in _l[1:]:
            l = pd.merge(l, i, left_index=True, right_index=True, how="outer")

        y = _y[0]
        for i in _y[1:]:
            y = pd.merge(y, i, left_index=True, right_index=True, how="outer")

        mask = y.isna().sum(axis=1) < 15

        self.data = l.loc[mask]
        self.Y = y.loc[mask]

        self.len = len(self.data)

    def _read_geo(self):
        self.geo = BaseGeoDataReader()
        self.edge_index = self.geo.edge_index

    def __getitem__(self, item):
        data = self.data.iloc[item]
        y = self.Y.iloc[item]
        ration = self.num_mask[item]
        x, y, edge_index, mask =  self._getitem(data, y, ration)
        return Data(x=x, y=y, edge_index=edge_index, mask=mask)

    def _getitemold(self, data, y, ration=0.5):
        data = data.dropna().values
        data = torch.tensor(data, dtype=torch.float).reshape(-1, self.dim)

        y = y[~y.isna()]
        _idx = np.random.choice(range(len(y)), size = int(len(y) * ration), replace=False)

        mask = torch.zeros(len(y), dtype=torch.bool)
        mask[_idx] = True

        data[mask] = data[~mask].mean(axis=0)

        return data, torch.tensor(y), self._get_edge(y), mask

    def _get_edge(self, y):
        idx = torch.tensor(list(map(int, y.index)), dtype=torch.int) - 1
        mapping_size = max(idx) + 1
        global_to_local = torch.full((mapping_size,), -1, dtype=torch.long)
        for local_idx, node in enumerate(idx):
            global_to_local[node] = local_idx

        mask_0 = torch.isin(self.edge_index[0], idx)
        mask_1 = torch.isin(self.edge_index[1], idx)
        mask = mask_0 & mask_1

        edge_index = self.edge_index[:, mask]
        edge_index = global_to_local[edge_index]
        return torch.cat([edge_index, edge_index.flip(0)], dim=1)

    def _getitem(self, data, y, ration=0.5):
        offline = y.isna()
        onlineidx = list(map(int, y[~y.isna()].index))

        data = data.fillna(0).values
        data = torch.tensor(data, dtype=torch.float32).reshape(-1, self.dim)
        y = y.fillna(0).values

        _idx = np.random.choice(onlineidx, size=max(int(len(onlineidx) * ration), 1), replace=False)

        mask = torch.zeros(len(y), dtype=torch.bool)
        mask[_idx - 1] = True

        data[mask] = data[~mask & ~offline.values].mean(axis=0)
        return data, torch.tensor(y, dtype=torch.float32), self.edge_index, mask


if __name__ == "__main__":
    from torch_geometric.loader import DataLoader
    a = GraphDataset()
    loader = DataLoader(a, batch_size=1)
    for i in loader:
        print(i)
        break
