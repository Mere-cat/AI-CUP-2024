from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np

from lightgbm import LGBMRegressor

import optuna
try:
    import optunahub
except ImportError:
    optunahub = False

try:
    from GeoDateRead import BaseGeoDataReader
    from StationDateRead import StationDataReader

except ImportError:
    from .GeoDateRead import BaseGeoDataReader
    from .StationDateRead import StationDataReader

n_trials = 15
timeout = 2000
early_stop = 50
sampler = None
#loss = "reg:squarederror"


class BasePredict:
    def __init__(self, col, pred, station_name=3, sampler=None, test_size=0.4, device="cpu", alpha=0.6, beta=0.4, parms=None):
        self.col = col
        self.pred = pred
        self.station_name = station_name
        self.Sampler = sampler

        self.test_size = test_size
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.parms = parms
        self._read_flow()

    def _read_flow(self):
        self.full_df = self._read_all()
        self.geo_df = BaseGeoDataReader().node_geo

        station = StationDataReader(self.station_name)
        self.station_df = station.df
        value_name = station.name

        self.full_df["DateTime"] = pd.to_datetime(self.full_df["DateTime"])
        self.station_df["DateTime"] = pd.to_datetime(self.station_df["DateTime"])

        self.full_df = self.full_df[(self.full_df["DateTime"].dt.hour >= 9) & (self.full_df["DateTime"].dt.hour < 17)]

        train_df = pd.merge(self.full_df, self.station_df[["DateTime", value_name]], on=["DateTime"], how='left')
        train_df = pd.merge(train_df, self.geo_df[["ID", "lat", "lon"]], on=["ID"], how='left')
        train_df.dropna(inplace=True)
        print(train_df.columns)

        self.x = train_df[self.col]
        self.y = train_df[self.pred]

    def _train_flow(self):
        if self.parms is None:
            if self.Sampler is None:
                Sampler = None if optunahub is False else optunahub.load_module("samplers/auto_sampler").AutoSampler()
            elif not sampler:
                Sampler = None
            else:
                Sampler = self.Sampler

            Sampler = Sampler() if callable(Sampler) else Sampler
            study = optuna.create_study(sampler=Sampler, direction="minimize")
            study.optimize(self.optuna_objective,
                           timeout=timeout,
                           n_trials=n_trials,
                           show_progress_bar=True)

            self.parms = study.best_params

            print("Best trial parameters:", self.parms)

        self._train()
        self._predict_flow()

    def fit(self):
        assert hasattr(self, "full_df"), "Please init first."
        self._train_flow()

    def train(self, train_x, train_y, valid_x, valid_y, parms):
        raise NotImplementedError
        #model = xgb.XGBRegressor(early_stopping_rounds=early_stop, **parms)
        #model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)])
        #return model

    def _train_step(self, train_x, train_y, valid_x, valid_y, parms, final=True):
        model = self.train(train_x, train_y, valid_x, valid_y, parms)
        if final:
           self.model = model
        return model.predict(valid_x)

    def _train(self, verbose=True):
        train_x, valid_x, train_y, valid_y = self.split_data(self.x, self.y)

        best_parm = self.parms if self.parms is not None else {}
        if not best_parm.get("tree_method"):
            best_parm["tree_method"] = "exact" if self.device == "cpu" else "approx"

        pred = self._train_step(train_x, train_y, valid_x, valid_y, best_parm)
        if verbose:
            print("MAE:", mean_absolute_error(valid_y, pred))
            print("R2 :", r2_score(valid_y, pred))
        return pred

    def _predict_flow(self):
        hat_name = [f"{hat}_hat" for hat in self.pred]

        self.full_df[hat_name] = self.model.predict(self.full_df[self.col])

        for hat in hat_name:
            self.full_df[hat] = np.clip(self.full_df[hat], a_min=0, a_max=None)

            self.full_df[hat] = self.full_df[hat].replace(0.0, np.nan)
            self.full_df[hat] = self.full_df.groupby("ID")[hat].ffill().bfill()
        self.grouby_save(self.full_df)


    def read(self, idx):
        raise NotImplementedError
        #return pd.read_csv(f"data/TrainingData_fin/L{i + 1}_Train_avg.csv")

    def save(self, x):
        raise NotImplementedError
        #lambda x: x.to_csv(f"data/TrainingData_hat/L{x['ID'].iloc[0]}_Train_hat.csv", index=False)

    def _read_all(self):
        df = pd.DataFrame()
        for i in range(1, 18):
            _df = self.read(i)
            _df["ID"] = i
            df = pd.concat([df, _df])
        return df

    @staticmethod
    def nor(df):
        return (df - df.min()) / (df.max() - df.min())

    def split_data(self, x, y):
        return train_test_split(x, y, test_size=self.test_size)

    def optuna_kwargs(self, trial):
        raise NotImplementedError

    def optuna_objective(self, trial):
        train_x, valid_x, train_y, valid_y = self.split_data(self.x, self.y)
        parms = self.optuna_kwargs(trial)
        pred = self._train_step(train_x, train_y, valid_x, valid_y, parms, final=False)

        r2 = r2_score(valid_y, pred)
        mae = mean_absolute_error(valid_y, pred)
        mae_baseline = mean_absolute_error(valid_y, [valid_y.mean()] * len(valid_y))
        return self.alpha * (1 - r2) + self.beta * (mae / mae_baseline)


    def grouby_save(self, df):
        return df.groupby("ID").apply(self.save, include_groups=True)


