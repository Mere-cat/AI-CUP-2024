# 競賽日誌
紀錄討論內容、寫好的功能 & TODOs

## TODO
- [ ]  ...


## 🛠️ Functions
### DataReader(by MH_LI)
* DataReader 暫時就是把每分鐘 壓到每十分鐘平均 之後還有全局歸一化

* 輸出檔會在/MH_LI/data/TrainingData_fin/中

* 要是你們需要更多的數據增強 你們可以自行建立你們的Reader 可以考慮`dataset/Reader.py`

#### Intro.
```python
class DataReader(BaseDataReader):
    """
    - 'Pressure' augmentation:
    def pressure(self, df_col):
        somethings
        return df_co

    - Global augmentation:
    def augmentation(self, df):
        df = somethings
        return df
    """
    def windspeed(self, df_col):
        return df_col.astype(float)

    def augmentation(self, df):
        normalizer = normalizeScale
        df.iloc[:, df.columns != "Power"] = normalizer(df.iloc[:, df.columns != "Power"])
        return df
```
簡單來說就是建立一個class 之後繼承BaseDataReader
* `def augmentaion(self, df)`: 進行全局級別的數據增加(例如是歸一化)

而且如果你需要針對個別的欄位進行獨立的增強
例如是 `WindSpeed`
* `def windspeed(self, df_col)`: function名字是小寫, df_col是指單獨欄位的series

最後就是 `run()`
``` python
run(idx, freq)
```
idx 就是你要讀取的機器ID
* `idx: (str, int) ` 時處理單一csv
* `idx: (Iterable)` 例如是 [1,2,3,4] 那會處理 L1, L2, L3, L4

freq就是轉換頻率/分鐘

## 👁️‍🗨️ Observation & Assumptions
1. 缺乏當天下午 5:30 ~ 隔天早上 6:00 左右的資料
2. 17 號私人發電站在美崙校區旁
3. test set 中的天數在其他機器有被偵測到
4. 地科知識：海拔越高濕度越低
5. 發電機位置：

    * 1-12: 東華大學理工學院二館
    * 13, 14: 管理學院
    * 17: 猜是在美崙校園外的倉庫群
6. 風速應該可以不納入討論
7. **樓層非常重要**
8. baseline 可以說明資料時序性很強

## Data Argumentation
### 地理資料處理
1. **模擬真實高度**算法：找出海拔高度 + 樓高* 4m
2. ⭐️ **面朝方向**：

    - 面朝資料直接取sin
    - 外增加"太陽面朝"(時間變動)的特徵 取sin

## 🚏 Strategies
1. 手動查當天發電量
2. 主辦方：用「當天 9:00 前資料」 & 「前一天整天資料」去做當天預測
3. 拿其他台同時間的資料做回歸，當做對目標發電機的預測

    → 發現其實不同機台同時間的發電量差滿多的

    → 可能是因為遮蔽或是其他因素 **面朝和高度**

4. 基於地理位置的圖 搭配單一時間建立子圖 算出embedding後丟lstm嗎

## ⏳ History
| Times        | Rank          | Description        |
| ------------ | ------------- | ------------- |
| 1 | 83  |  對17個站點單獨找平均, 之後upload 各自的平均   |
| 2 | 42  |  直接取單獨機台 相同時間 的歷史平均   |
| 3 |     |     |
