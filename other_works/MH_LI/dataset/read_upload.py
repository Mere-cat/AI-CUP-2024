import pandas as pd
import os

dataPath = __file__.split(os.sep)[:-2]
upload_path = "data/36_TestSet_SubmissionTemplate/36_TestSet_SubmissionTemplate/upload(no answer).csv"

def upload2datatime():
    df = pd.read_csv(os.path.join(os.sep, *dataPath, upload_path))
    df_day = df["序號"].astype(str)

    df['year'] = df_day.str[:4]
    df['month'] = df_day.str[4:6]
    df['day'] = df_day.str[6:8]
    df['hour'] = df_day.str[8:10]
    df['minute'] = df_day.str[10:12]
    df['id'] = df_day.str[12:].astype(int)

    df['DateTime'] = pd.to_datetime(df['year'] + df['month'] + df['day'] + df['hour'] + df['minute'],
                                    format='%Y%m%d%H%M')

    df.drop(columns=['year', 'month', 'day', 'hour', 'minute'], inplace=True)
    return df

if __name__ == "__main__":
    print(upload2datatime())


