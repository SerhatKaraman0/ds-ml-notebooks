import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import os 
import typing
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class AnalyzeDataFrame:
    def __init__(self):
        pass

    def analyze_df(self, df):
        print(20 * "-", "Columns", 20 * "-")
        print(df.columns)
        print(20 * "-", "First 5 Data in the Dataframe", 20 * "-")
        print(df.head(5))
        print(20 * "-", "DF INFO", 20 * "-")
        print(df.info())
        print(20 * "-", "DF DESCRIBE", 20 * "-")
        print(df.describe())
        print(20 * "-", "NULL COUNt", 20 * "-")
        print(df.isna().sum())
        print(20 * "-", "DF SHAPE", 20 * "-")
        print(df.shape)
        print(20 * "-", "DF DUPLICATES", 20 * "-")
        print(df.duplicated().sum())
        print(20 * "-", "DF UNIQUE VALUES", 20 * "-")
        print(df.nunique())
        print(20 * "-", "DF VALUE COUNTS", 20 * "-")
        print(df.value_counts())
        print(20 * "-", "UNIQUE VALUES EACH COLUMN", 20 * "-")
        for col in df.columns:
            print(20 * "-", f"{col} UNIQUE VALUES", 20 * "-")
            print(df[col].unique())
         
    def grab_col_names(self, df: pd.DataFrame, cat_th: int = 10, car_th: int = 20) -> tuple[list, list, list]:
        """
        Identifies and categorizes columns in a dataframe as categorical, numerical, or categorical but cardinal.
        
        Parameters:
        -----------
        df: pd.DataFrame
            The input dataframe
        cat_th: int, default=10
            Threshold for numerical columns to be considered categorical
        car_th: int, default=20
            Threshold for categorical columns to be considered cardinal
            
        Returns:
        --------
        tuple[list, list, list]:
            cat_cols: Categorical columns
            num_cols: Numerical columns
            categorical_but_car: Categorical but cardinal columns
        """
        # 1. Fix: Changed "0" to "O" for object dtype check
        categorical_cols = [col for col in df.columns if df[col].dtypes == "O"]
        
        # 2. Fix: Added proper dtype checks for numerical columns
        numerical_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and 
                            df[col].dtypes in ['int64', 'float64']]
        
        # 3. Fix: Check categorical columns for cardinality
        categorical_but_car = [col for col in categorical_cols if df[col].nunique() > car_th]
        
        # Combine categorical columns
        cat_cols = categorical_cols + numerical_but_cat
        cat_cols = [col for col in cat_cols if col not in categorical_but_car]
        
        # 4. Fix: Simplified numerical column identification
        num_cols = [col for col in df.columns if 
                    df[col].dtypes in ['int64', 'float64'] and 
                    col not in numerical_but_cat]
        
        # Print summary
        print(f"Observations: {df.shape[0]}")
        print(f"Variables: {df.shape[1]}")    
        print(f"cat_cols: {len(cat_cols)}")    
        print(f"num_cols: {len(num_cols)}")    
        print(f"cat_but_car: {len(categorical_but_car)}")
        print(f"num_but_cat: {len(numerical_but_cat)}")
        
        print("\nCategorical Cols:", cat_cols)
        print("\nNumerical Cols:", num_cols)
        print("\nCategorical but cardinal Cols:", categorical_but_car)
        
        return cat_cols, num_cols, categorical_but_car
    
    def correlation_for_drop(self, df: pd.DataFrame, threshold: float = 0.85):
        columns_to_drop = set()
        correlations = df.corr()
        col_len = len(correlations.columns)
        
        for i in range(col_len):
            for j in range(i):
                if abs(correlations.iloc[i,j]) > threshold: # type: ignore
                    columns_to_drop.add(correlations.columns[i])

        return columns_to_drop