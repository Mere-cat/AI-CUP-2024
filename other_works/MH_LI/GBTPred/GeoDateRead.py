import numpy as np
import pandas as pd
import torch
import os

dataPath = __file__.split(os.sep)[:-2]
dataPath = os.path.join(os.sep, *dataPath, "data")

Geo_Path = "geo_data"

def haversine(lat1, lon1, lat2, lon2):
    import math
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

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
        self.node_geo["ele"] = self.node_geo["面朝"]

        self.sum_ele_s = self._interpolate(self.sum_ele)
        self.sum_pos_s =self._interpolate(self.sum_pos)

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
        solar_df = solar_df.resample('10min').interpolate("time")
        return solar_df[["SolarData"]]

    @staticmethod
    def _angle_to_radians_s(angle):
        return np.sin(angle * np.pi / 180)

    @staticmethod
    def _angle_to_radians_s(angle):
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