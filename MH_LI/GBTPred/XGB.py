import xgboost as xgb
from base import BasePredict

import pandas as pd
import os

dataPath = __file__.split(os.sep)[:-2]
dataPath = os.path.join(os.sep, *dataPath, "data")

class XGBPredict(BasePredict):
    def __init__(self, col, pred, station_name=3, sampler=None, test_size=0.4, device="cpu", alpha=0.6, beta=0.4, parms=None):
        super().__init__(col, pred, station_name, sampler, test_size, device, alpha, beta, parms)
        self.multi_strategy = "multi_output_tree" if isinstance(pred, list) and len(pred) > 1 else "single_output"

    def train(self, train_x, train_y, valid_x, valid_y, parms):
        model = xgb.XGBRegressor(early_stopping_rounds=early_stop, **parms)
        model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], verbose=False)
        return model

    def read(self, idx):
        return pd.read_csv(os.path.join(dataPath, read_path,f"L{idx}{read_file}.csv"))

    def save(self, group):
        group.to_csv(os.path.join(dataPath, save_path, f"L{group['ID'].iloc[0]}{save_file}.csv"), index=False)

    def optuna_kwargs(self, trial):
        param = {
            "device": self.device,
            "multi_strategy": self.multi_strategy,
            # use exact for small dataset.
            "tree_method": "exact" if self.device == "cpu" else "approx",

            "objective": "reg:squarederror",
            "n_estimators": trial.suggest_int('n_estimators', 500, 2000, step=100),

            # defines booster, gblinear for linear functions.
            "booster": trial.suggest_categorical("booster", ["gbtree"]),
            # L2 regularization weight.
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            # L1 regularization weight.
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            # sampling ratio for training data.
            "subsample": trial.suggest_float("subsample", 0.2, 0.5),
            # sampling according to each tree.
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        }

        if param["booster"] in ["gbtree", "dart"]:
            # maximum depth of the tree, signifies complexity of the tree.
            param["max_depth"] = trial.suggest_int("max_depth", 3, 15, step=3)
            # minimum child weight, larger the term more conservative the tree.
            param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10, step=1)
            param["eta"] = trial.suggest_float("eta", 0.05, 1.0, log=True)
            # defines how selective algorithm is.
            param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            param["rate_drop"] = trial.suggest_float("rate_drop", 0.01, 1.0, log=True)
            param["skip_drop"] = trial.suggest_float("skip_drop", 0.01, 1.0, log=True)

        if param["multi_strategy"] == "multi_output_tree":
            param["tree_method"] = "hist"

        return param


if __name__ == "__main__":
    col = ["Date", "Month", "Day", "Hour", "high", "ele", "sum_ele", "sum_pos"]
    pred = ["Sunlight", "Power"]
    stat_name = "GlobalSolarRadiation"

    read_path = "TrainingData_fin"
    read_file = "_Train_avg"

    save_path = "TrainingData_hat"
    save_file = "_Train_hat"

    early_stop = 50
    model = XGBPredict(col, pred)
    model.fit()

