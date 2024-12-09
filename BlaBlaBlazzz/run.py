import os
import json
import warnings
import optuna
import numpy as np
import pandas as pd
from io import StringIO
from pathlib import Path
from datetime import datetime

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, sum_models

from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import StackingRegressor

warnings.filterwarnings("ignore")


class RunModel:
    def __init__(self, beta = 0.5):
        super(RunModel, self).__init__()
        self.dir = os.getcwd()
        self.beta = beta
    
    # generate processed datasets
    def processing_original_dataset(self):
        for i in range(1, 18):
            dataset_dir = self.dir + f"/36_TrainingData/L{i}_Train.csv"
            print(dataset_dir)
            df = pd.read_csv(dataset_dir)

            # concat additional data
            if i in [2, 4, 7, 8, 9, 10, 12]:
                addition_dir = os.getcwd() + f"/36_TrainingData_Additional_V2/L{i}_Train_2.csv"
                df2 = pd.read_csv(addition_dir)
                df = pd.concat([df, df2], axis = False)

            df['DateTime'] = pd.to_datetime(df['DateTime'])
            col = df.pop("Power(mW)")
            df["Power(mW)"] = col

            df.set_index('DateTime', inplace=True)
            df = df.resample('10T').mean()
            df = df.round(2)

            df.reset_index(inplace=True)
            # drop unrecorded time data
            df = df.dropna()

            df.to_csv(self.dir + f"/TrainingData_avg/L{i}_Train.csv", index=False)
    
    def make_date_device_dict(self, data_ls):
        target_dir = self.dir + f"/upload.csv"
        target = pd.read_csv(target_dir, encoding='utf8')['序號'].values

        # selected 200 days
        date_device = {}
        for item in target:
            date = str(item)[:8]
            device = str(item)[12:]
            dd = date + device

            if dd not in date_device.keys():
                date_device[dd] = [int(device)]
        # data_ls = load_data()

        for date in list(date_device.keys()):
            year = date[:4]
            month = date[4:6]
            day = date[6:-2]

            start_time = pd.Timestamp(f"{year}-{month}-{day} 09:00:00")
            end_time = pd.Timestamp(f"{year}-{month}-{day} 16:50:00")
            # traverse all dataset to find target days
            for i in range(1, 18):
                data = data_ls[i-1]

                # filter device with enough target time
                filtered_df = data[(data['DateTime'] >= start_time) & (data['DateTime'] <= end_time)]
                if len(filtered_df)>40:  # with missing data less than 8
                    date_device[date].append(i)
        
        # write file        
        with open("target_match_device.txt", 'w') as file:
            file.write(json.dumps(date_device))

        return date_device
    
    def load_data(self):
        data_ls = []
        for i in range(1, 18):
            data_dir = self.dir + f"/data/L{i}_Train.csv"
            data = pd.read_csv(data_dir)
            data['DateTime'] = pd.to_datetime(data['DateTime'])
            data.dropna(axis=0, how='any', inplace=True)
            data_ls.append(data)  # store data
        
        return data_ls
    
    def load_date_device_dict(self, data_ls):
        if os.path.exists(self.dir + "/target_match_device.txt"):
            with open(self.dir + "/target_match_device.txt") as f:
                date_dict = f.read()
            date_dict = json.loads(date_dict)
        else:
            date_dict = self.make_date_device_dict(data_ls)

        return date_dict

    def feature_engineering(self, df):
        df = df.copy()
        # df['hour'] = df['DateTime'].dt.hour
        # df['minute'] = df['DateTime'].dt.minute
        # baseline_date = pd.to_datetime('2024-01-01')
        # # distance to the baseline date(2024-01-01)
        # df['days'] = (df['DateTime'] - baseline_date).dt.days
        # df['month'] = df['DateTime'].dt.month


        # df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        # df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 24)
        # df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 24)

        df['sun_power_inter'] = df['Sunlight(Lux)'] * df['Power(mW)']  # interaction effect 

        df['lag_sunlight'] = df['Sunlight(Lux)'].shift(1)
        df['lag_power'] = df['Power(mW)'].shift(1)

        # df.drop(columns=['DateTime', 'LocationCode'], inplace=True)
        df.drop(columns=['idx', 'DateTime', 'LocationCode', 'height', 'angle', 'is_upper', 'Sunlight(Lux)'], inplace=True)
        return df
    
    # get interaction date of two devices
    def get_intersection_date(self, data, id1, id2, date_id):
        data1 = data[id1-1]
        data2 = data[id2-1]

        # assign id1 divice to be the target power(label)
        target = data1[data1['DateTime'].isin(data2['DateTime'])].reset_index(drop=True)
        target.rename(columns={'Power(mW)': 'Label'}, inplace=True)
        target = target['Label']

        # get id2 device's data as features
        match2 = data2[data2['DateTime'].isin(data1['DateTime'])].reset_index(drop=True)

        # target test data
        start_time = pd.to_datetime(f"{date_id}0900", format='%Y%m%d%H%M')
        end_time = pd.to_datetime(f"{date_id}1650", format='%Y%m%d%H%M')
        test_data = data2[(data2['DateTime'] >= start_time) & (data2['DateTime'] <= end_time)].reset_index(drop=True)

        match2 = self.feature_engineering(match2)

        test_data = self.feature_engineering(test_data)
        # if 
        if test_data.shape[0] < 48:
            test_data = self.fillna(test_data)

        return match2, target, test_data

    def split_train_val_test(self, df, label):
        X_train, X_val, y_train, y_val = train_test_split(df, label, test_size=0.2, random_state=42)
        return X_train, X_val, y_train, y_val

    def normalize_features(self, X_train, X_val, X_test):
        min_max_scaler = preprocessing.MinMaxScaler()

        X_train_scaled = min_max_scaler.fit_transform(X_train)
        X_val_scaled = min_max_scaler.transform(X_val)
        X_test_scaled = min_max_scaler.transform(X_test)

        A = pd.DataFrame(X_train_scaled, columns=X_train.columns).astype(float)
        B = pd.DataFrame(X_val_scaled, columns=X_val.columns).astype(float)
        C = pd.DataFrame(X_test_scaled, columns=X_test.columns).astype(float)

        return A, B, C
    
    # get all time pairs
    def complete_target_time(self):
        target_time = []
        for hour in range(9, 17):
            for minute in range(0, 60, 10):
                if hour == 16 and minute > 50:
                    break
                target_time.append((hour, minute))
        
        return target_time
    
    # fill missing testing time 
    def fillna(self, X_test):
        target_time = self.complete_target_time()
        time_df = pd.DataFrame(target_time, columns=["hour", "minute"])

        df = time_df.merge(
            X_test,
            on=["hour", "minute"],
            how="left"
        )
        df.fillna(method="ffill", inplace=True)
        df.fillna(method="bfill", inplace=True) 
        df = df[X_test.columns]

        return df

    def get_avg_of_closest4(self, data, date_id, result):
        df = self.feature_engineering(data)
        target_time = self.complete_target_time()

        # days from 2024-01-01
        date = pd.to_datetime(f"{date_id[:4]}-{date_id[4:6]}-{date_id[6:]}")
        baseline_date = pd.to_datetime('2024-01-01')
        target_day = (date - baseline_date).days

        for (hour, minute) in target_time:
            time_data = df[(df['hour'] == hour) & (df['minute'] == minute)]
            date_diff = np.abs(time_data['day'] - target_day)
            # choose the closest 4 days of power
            closest4_days = time_data.iloc[np.argsort(date_diff)[:4]]
            
            average_power = closest4_days['Power(mW)'].mean()
            result.append(np.round(average_power, 2))
        
        return result
    
    def train_model(self, X_train, y_train, X_val, y_val):
        model = XGBRegressor(n_estimators=100, random_state=42, verbose=0)
        model.fit(X_train, y_train)
        mse, mae = self.evaluate_metrics(model, X_val, y_val)

        return model, mse, mae
    
    def evaluate_metrics(self, model, X_val, y_val):
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)

        return mse, mae
    
    def run(self):
        data_ls = self.load_data()
        date_device = self.load_date_device_dict(data_ls)
        total_mse = 0
        total_mae = 0
        
        result = []  # results of testing set
        model_cache = {}  # store models

        for idx, (date_id, device_ls) in enumerate(date_device.items()):
            date_id = date_id[:-2]

            best_mse = float('inf')
            best_mae = float('inf')
            best_device = None
            
            model_weights = []
            model_pred = []

            # no target day found in other device, take average of 4 closest days
            if len(device_ls) == 1:
                data = data_ls[device_ls[0]]  # data of target device
                result = self.get_avg_of_closest4(data, date_id, result)

                print("current problem: {}/200 | No device, fill with average power in 4 days".format(idx+1))

            else:
                for dev_idx, dev in enumerate(device_ls[1:]):
                    train_data, label_data, X_test = self.get_intersection_date(data_ls, device_ls[0], dev, date_id)
                    X_train, X_val, y_train, y_val = self.split_train_val_test(train_data, label_data)
                    
                    if (device_ls[0], dev) not in list(model_cache.keys()):
                        epoch_model, epoch_mse, epoch_mae = self.train_model(X_train, y_train, X_val, y_val)
                        model_cache[(device_ls[0], dev)] = [epoch_model, epoch_mse, epoch_mae]  # store pair model in cache
                    else:
                        epoch_model, epoch_mse, epoch_mae = model_cache[(device_ls[0], dev)]
                    
                    # model_weights.append(1 / epoch_mae)
                    model_weights.append(epoch_mae)

                    if epoch_mae < best_mae:
                        best_model = epoch_model
                        best_mse = np.round(epoch_mse, 4)
                        best_mae = np.round(epoch_mae, 4)
                        best_device = dev
                    
                    # aggregate test data
                    predictions = epoch_model.predict(X_test)
                    model_pred.append(predictions)
                
                total_mse += best_mse
                total_mae += best_mae
                
                print("current problem: {}/200 | Best MSE: {} | Best MAE: {} in Device {}".format(idx+1, best_mse, best_mae, best_device))
                
                # print((1/np.array(model_weights)) / sum((1/np.array(model_weights))))
                
                # softmax
                model_weights = np.exp(-self.beta * np.array(model_weights))
                model_weights = model_weights / np.sum(model_weights)

                epoch_result = np.sum(np.array(model_pred).T * model_weights, axis=1)
                epoch_result = np.round(epoch_result, 2)
                result.extend(epoch_result)
            # display(X_test)

        self.save_results(result)
        print("\nAll questions finished, Total MSE: {} | Total MAE: {}".format(total_mse, total_mae))
        print("Save results at answer.csv")
        

    def save_results(self, result):
        upload = pd.read_csv(self.dir + "/results/upload.csv")
        upload['答案'] = result
        upload.to_csv(self.dir + "answer.csv", index=False)


if __name__=='__main__':
    RunModel().run()