import pandas as pd

def upload2datatime():
    df = pd.read_csv(r'36_TestSet_SubmissionTemplate/36_TestSet_SubmissionTemplate/upload(no answer).csv')
    df_day = df["序號"].astype(str)

    df['year'] = df_day.str[:4]
    df['month'] = df_day.str[4:6]
    df['day'] = df_day.str[6:8]
    df['hour'] = df_day.str[8:10]
    df['minute'] = df_day.str[10:12]
    df['id'] = df_day.str[12:]

    # 創建一個新的 datetime 欄位
    df['datetime'] = pd.to_datetime(df['year'] + df['month'] + df['day'] + df['hour'] + df['minute'],
                                    format='%Y%m%d%H%M')

    # 刪除不需要的分拆欄位 (可選)
    df.drop(columns=['year', 'month', 'day', 'hour', 'minute'], inplace=True)
    print(df[["datetime", "id"]])

if __name__ == "__main__":
    upload2datatime()


