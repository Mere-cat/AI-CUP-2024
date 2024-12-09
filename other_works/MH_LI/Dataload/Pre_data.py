from collections.abc import Iterable
import os
from operator import index

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
start_date = "2024-01-01 06:30:00"
between_time = ["06:30", "17:00"]

df_Geo = None

class BaseDataReader(object):
    @classmethod
    def run(cls, idx = range(1, 18), freq=10):
        if isinstance(idx, Iterable):
            qbar = tqdm(idx, desc="Reading data")
            for i in qbar:
                cls()._work_flow(i, freq, False)
        else:
            cls()._work_flow(idx, freq)

    def init(self, verbose):
        self.verbose = verbose

    def _work_flow(self, idx, freq=10, verbose=True):
        self.init(verbose)
        self._read_data(idx)
        self._time_augmentation(freq)
        self._geo_augmentation(idx)

        self.df.iloc[:, :-5] = self.augmentation(self.df.iloc[:, :-5])
        self.save_csv(idx)

    def _geo_augmentation(self, idx):
        node_geo = df_Geo.node_geo[df_Geo.node_geo["ID"] == idx]
        self.df["high"] = node_geo["模擬高度"].values[0]
        self.df["ele"] = node_geo["ele"].values[0]

        self.df["sum_ele"] = df_Geo.sum_ele.loc[self.df["DateTime"]].values
        self.df["sum_pos"] = df_Geo.sum_pos.loc[self.df["DateTime"]].values

        column = ["DateTime", "Date",
                  "WindSpeed", "Pressure", "Temperature", "Humidity", "Sunlight",
                  "sum_ele", "sum_pos", "high", "ele", "Power"]

        self.df = self.df[column]

    def _read_data(self, idx):
        idx = str(idx) if not isinstance(idx, str) else idx
        df_org = pd.read_csv(os.path.join(dataPath, Train_Path, f"L{idx}_Train.csv"))
        try:
            df_add = pd.read_csv(os.path.join(dataPath, Additional_Path, f"L{idx}_Train_2.csv"))
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

    def augmentation(self, df):
        return df

    def _time_augmentation(self, freq):
        self.df['DateTime'] = pd.to_datetime(self.df['DateTime'])
        self._resample_time(freq)
        self._build_DateIdx()

    def _resample_time(self, freq):
        freq = str(freq) if not isinstance(freq, str) else freq
        self.df["Date"] = self.df["DateTime"].dt.date - pd.to_datetime(start_date).date()
        self.df["Date"] = self.df["Date"].dt.days.astype(int)
        self.df.set_index('DateTime', inplace=True)
        self.df = self.df.resample(f'{freq}min').mean()
        self.df = self.df.between_time(*between_time)
        self.df.reset_index(inplace=True)

    def _build_DateIdx(self):
        self.df["DateIdx"] = self.df["DateTime"] - pd.to_datetime(start_date)
        self.df["DateIdx"] = ((self.df["DateIdx"]).dt.total_seconds() // 600).astype(int)
        self.df.set_index('DateIdx', inplace=True)

    def save_csv(self, idx):
        self.df.to_csv(os.path.join(dataPath, fin_Path, f"L{idx}_Train_avg.csv"), index=False)


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

        self.sum_ele = self._angle_to_radians(self._interpolate(self.sum_ele))
        self.sum_pos = self._angle_to_radians(self._interpolate(self.sum_pos))

    def _interpolate(self, df):
        df.reset_index(inplace=True)
        solar_df = pd.melt(df, id_vars=['index'], var_name='Hour', value_name='SolarData')
        solar_df["index"] = (solar_df["index"] + 1)
        SolarData = solar_df["SolarData"].copy()
        solar_df = solar_df.astype(int)

        solar_df['TimeIndex'] = solar_df.apply(
            lambda row: pd.Timestamp(year=2024, month=row['index'], day=1, hour=row['Hour']), axis=1)

        solar_df["SolarData"] = SolarData
        solar_df = solar_df.set_index('TimeIndex').sort_index()
        solar_df = solar_df.resample('10T').interpolate("time")
        return solar_df[["SolarData"]]

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

df_Geo = BaseGeoDataReader()

if __name__ == "__main__":
    BaseDataReader.run()
