from collections.abc import Iterable
import os

import pandas as pd
import numpy as np
from tqdm import tqdm

pd.options.mode.chained_assignment = None
dataPath = __file__.split(os.sep)[:-2]
dataPath = os.path.join(os.sep, *dataPath, "data")

Train_Path = "36_TrainingData"
Additional_Path = "36_TrainingData_Additional_V2"
fin_Path = "TrainingData_fin"

class BaseDataReader(object):
    @classmethod
    def run(cls, idx = range(1, 18), freq=10):
        if isinstance(idx, Iterable):
            qbar = tqdm(idx, desc="Reading data")
            for i in qbar:
                cls()._work_flow(i, freq, False)
        else:
            cls()._work_flow(idx, freq)

    def init(self, id, freq, verbose):
        self.name = id
        self.freq = freq
        self.verbose = verbose

    def _work_flow(self, id, freq=10, verbose=True):
        self.init(id, freq, verbose)

        self._read_data()
        self._col_augmentation()
        self._avg_df()

        aug_col = self.df.columns != "DateTime"
        self.df.iloc[:, aug_col] = self.augmentation(self.df.iloc[:, aug_col])

        self.save_csv()

    def augmentation(self, df):
        pass

    def save_csv(self):
        self.df.to_csv(os.path.join(dataPath, fin_Path, f"L{self.name}_Train_avg.csv"), index=False)

    def _read_data(self):
        name = self.name
        name = str(name) if not isinstance(name, str) else name
        df_org = pd.read_csv(os.path.join(dataPath, Train_Path, f"L{name}_Train.csv"))
        try:
            df_add = pd.read_csv(os.path.join(dataPath, Additional_Path, f"L{name}_Train_2.csv"))
            df_final = pd.concat([df_org, df_add], ignore_index=True)

        except FileNotFoundError:
            df_final = df_org

        df_final = df_final.drop(columns=['LocationCode'])
        df_final.columns = df_final.columns.str.replace(r'\(.*\)', '', regex=True).str.strip()
        self.df = df_final

    def _col_augmentation(self):
        for col in self.df.columns:
            method = getattr(self, col.lower(), False)
            if method:
                if self.verbose:
                    print(f"Data augmentation: {col}")
                self.df[col] = method(self.df[col])

    def _avg_df(self):
        freq = str(self.freq) if not isinstance(self.freq, str) else self.freq

        self.df['DateTime'] = pd.to_datetime(self.df['DateTime'])
        self.df.set_index('DateTime', inplace=True)

        df_resampled = self.df.resample(f'{freq}T').sum()#.apply(lambda x: np.sum(x.values))
        df_resampled.reset_index(inplace=True)
        self.df = df_resampled

    def __getattr__(self, item):
        pass