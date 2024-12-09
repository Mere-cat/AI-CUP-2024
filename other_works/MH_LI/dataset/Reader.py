import numpy as np

try:
    from base import BaseDataReader
except ImportError:
    from .base import BaseDataReader

eps = 1e-6

def normalizeScale(df):
    scale = (1.0 / (df.abs().max() + eps)) * 0.999999
    return df * scale

def col_normalizeScale(df):
    for col in df.columns:
        scale = (1.0 / (df[col].abs().max() + eps)) * 0.999999
        df.loc[: col] *=  scale
    return df

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

if __name__ == "__main__":
    DataReader.run()