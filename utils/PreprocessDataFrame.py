import typing
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 


class PreprocessDataFrame:
    def __init__(self):
        pass

    def outlier_thresholds(self, df: pd.DataFrame, col_name: str, q1: float = 0.25, q3: float = 0.75) -> typing.Tuple[float, float]:
        quartile1 = df[col_name].quantile(q1)
        quartile3 = df[col_name].quantile(q3)

        interquantile_range = quartile3 - quartile1

        up_limit = quartile3 + 1.5 * interquantile_range
        low_limit = quartile1 - 1.5 * interquantile_range
         
        return low_limit, up_limit
    

    def chech_outlier(self, df: pd.DataFrame, col_name: str):
        low_limit, up_limit = self.outlier_thresholds(df, col_name)

        return df[(df[col_name] > up_limit) | (df[col_name] < low_limit)].any(axis=None)
    

    def grab_outliers(self, df: pd.DataFrame, col_name: str, index: bool = False):
        low, up = self.outlier_thresholds(df, col_name)

        if df[(df[col_name] > up) | (df[col_name] < low)].shape[0] > 10:
            print(df[(df[col_name] > up) | (df[col_name] < low)].head())
        
        else:
            print(df[(df[col_name] > up) | (df[col_name] < low)])
        
        if index:
            outlier_index = df[(df[col_name] > up) | (df[col_name] < low)].index 
            return outlier_index
        
    
    def remove_outlier(self, df: pd.DataFrame, col_name: str):
        low, up = self.outlier_thresholds(df, col_name)
        df_without_outliers = df[~((df[col_name] > up) | (df[col_name] < low))]

        return df_without_outliers
    

    def replace_with_thresholds (self, df: pd.DataFrame, variable):
        low, up = self.outlier_thresholds(df, variable)
        df. loc[(df[variable] < low), variable] = low
        df. loc[(df[variable] > up), variable] = up
    

    def one_hot_encoder(self, df: pd.DataFrame, categorical_cols: list[str], drop_first: bool = False):
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first)
        return df 
