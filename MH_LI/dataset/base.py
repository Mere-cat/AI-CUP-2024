from collections.abc import Iterable
import os

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

pd.options.mode.chained_assignment = None
dataPath = __file__.split(os.sep)[:-2]
dataPath = os.path.join(os.sep, *dataPath, "data")

Train_Path = "36_TrainingData"
Additional_Path = "36_TrainingData_Additional_V2"
Geo_Path = "geo_data"

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

        df_resampled = self.df.resample(f'{freq}min').sum()#.apply(lambda x: np.sum(x.values))
        df_resampled.reset_index(inplace=True)
        self.df = df_resampled

    def __getattr__(self, item):
        pass

class BaseGeoDataReader(object):
    def __init__(self, threshold_km = 0.6):
        self.threshold_km = threshold_km
        self._work_flow()

    def _work_flow(self):
        self._read()
        self._DMS2decimal()
        self._build_graph()
        self._Angle2Radians()

    def _read(self):
        self.node_geo = pd.read_csv(os.path.join(dataPath, Geo_Path, f"node_geographical.csv"))
        self.sum_ele = pd.read_csv(os.path.join(dataPath, Geo_Path, f"sun_elevation.csv"))
        self.sum_pos = pd.read_csv(os.path.join(dataPath, Geo_Path, f"sun_position.csv"))


    def _DMS2decimal(self):
        for coord_str in self.node_geo["座標"]:
            lat_str, lon_str = coord_str.split()
            self.node_geo["lat"] = self.dms_to_decimal(lat_str)
            self.node_geo["lon"] = self.dms_to_decimal(lon_str)

    def _build_graph(self):
        def haversine(lat1, lon1, lat2, lon2):
            import math
            R = 6371.0  # 地球半徑/KM
            phi1, phi2 = math.radians(lat1), math.radians(lat2)
            delta_phi = math.radians(lat2 - lat1)
            delta_lambda = math.radians(lon2 - lon1)

            a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            return R * c


        coordinates = [[lat, lon] for lat, lon in zip(self.node_geo["lat"], self.node_geo["lon"])]

        edge_index = []
        num_nodes = len(coordinates)

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                coord_i = coordinates[i]
                coord_j = coordinates[j]
                distance_km = haversine(coord_i[0], coord_i[1], coord_j[0], coord_j[1])
                if distance_km < self.threshold_km:
                    edge_index.append([i, j])

        edge_index = torch.tensor(edge_index).T

        self.edge_index = edge_index[:, edge_index[0].argsort()]

    def _Angle2Radians(self):
        self.node_geo["ele"] = self._angle_to_radians(self.node_geo["面朝"])
        self.sum_ele = self._angle_to_radians(self.sum_ele)
        self.sum_pos = self._angle_to_radians(self.sum_pos)

    @staticmethod
    def _angle_to_radians(angle):
        return np.sin(angle * np.pi / 180)

    @staticmethod
    def dms_to_decimal(dms_str):
        import re
        dms_regex = r"(\d+)\u00b0(\d+)'(\d+)(?:\.\d+)?\"([NSEW])"
        match = re.match(dms_regex, dms_str)
        if not match:
            raise ValueError(f"Invalid DMS format: {dms_str}")
        degrees, minutes, seconds, direction = match.groups()
        decimal = int(degrees) + (int(minutes) / 60.0) + (int(seconds) / 3600.0)
        if direction in ['S', 'W']:
            decimal *= -1
        return decimal