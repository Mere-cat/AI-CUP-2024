import numpy as np

try:
    from base import BaseDataReader
except ImportError:
    from .base import BaseDataReader

class DataReader(BaseDataReader):
    def windspeed(self, df_col):
        return df_col

    def augmentation(self, df):
        normalizeScale = self.normalizeScale
        df = normalizeScale(df)
        return df

    def normalizeScale(self, df):
        scale = (1.0 / df.abs().max()) * 0.999999
        return df * scale

    def col_normalizeScale(self, df):
        for col in df.columns:
            scale = (1.0 / df[col].abs().max()) * 0.999999
            df.loc[: col] *=  scale
        return df

if __name__ == "__main__":
    DataReader.run()