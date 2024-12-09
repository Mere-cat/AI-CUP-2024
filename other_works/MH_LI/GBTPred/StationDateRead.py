import os
import pandas as pd
import numpy as np


pd.options.mode.chained_assignment = None
dataPath = __file__.split(os.sep)[:-2]
dataPath = os.path.join(os.sep, *dataPath, "data")

stat_Path = "stat_data"
start_date = "2024-01-01 06:30:00"
between_time = ["06:30", "17:00"]


class StationDataReader:
    def __init__(self, idx=1, name="GlobalSolarRadiation"):
        col_name = name
        df = pd.DataFrame()
        for i in range(1, 12):
            _df = pd.read_csv(os.path.join(dataPath,
                                           f"{stat_Path}_{idx}",
                                           f"2024-{i:02d}-{name}-hour.csv")).iloc[:-1]

            _df = _df.iloc[1:, 6:-8].replace(["--", "X"], np.nan).astype(float)
            _df = _df.fillna(_df.mean(axis=1))
            _df = _df.reset_index().melt(id_vars=['index'], var_name='Hour', value_name=name)
            _df["Month"] = i
            df = pd.concat([df, _df])

        df = df.astype(float)
        df.rename(columns={"index": "Day"}, inplace=True)
        df['DateTime'] = df.apply(
            lambda row: pd.Timestamp(year=2024, month=row['Month'].astype(int), day=row["Day"].astype(int),
                                     hour=row['Hour'].astype(int)), axis=1)

        df = df.set_index('DateTime').sort_index()
        name = df[name].resample('10min').interpolate("time")
        df = df.resample('10min').interpolate("pad")
        #df["name"] = value * 126000 / 60

        df.reset_index(inplace=True)
        df["Hour"] = df["DateTime"].dt.hour
        df["minute"] = df["DateTime"].dt.minute
        self.df = df
        self.name = col_name