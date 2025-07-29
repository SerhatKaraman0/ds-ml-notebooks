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
        print("{:*^90}".format(" " + "COLUMNS" + " "))
        print(df.columns)
        
        print("{:*^90}".format(" " + "DF INFO" + " "))
        print(df.info())
        
        print("{:*^90}".format(" " + "DF DESCRIBE" + " "))
        print(df.describe())

        print("{:*^90}".format(" " + "DF DUPLICATES" + " "))
        print(df.duplicated().sum())

        print("{:*^90}".format(" " + "DF UNIQUE VALUES" + " "))
        print(df.nunique())

        print("{:*^90}".format(" " + "DF VALUE COUNTS" + " "))
        print(df.value_counts())
        
        print("{:*^90}".format(" " + "UNIQUE VALUES EACH COLUMN" + " "))
        for col in df.columns:
            print("{:*^90}".format(" " + f"UNIQUE VALUES IN COLUMN {col}" + " "))
            print(df[col].unique())

    
    def check_df(self, df: pd.DataFrame, head: int = 5):
       print("{:*^90}".format(" " + "SHAPE" + " ")) 
       print(df.shape)
       print("{:*^90}".format(" " + "TYPES" + " ")) 
       print(df.dtypes)
       print("{:*^90}".format(" " + "HEAD" + " ")) 
       print(df.head(head))
       print("{:*^90}".format(" " + "TAIL" + " ")) 
       print(df.tail(head))
       print("{:*^90}".format(" " + "NA" + " ")) 
       print(df.isnull().sum())
       print("{:*^90}".format(" " + "QUANTILES" + " ")) 
       print(df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1], numeric_only=True).T)
    

    def missing_values_table(self, df: pd.DataFrame, na_name: bool = False):
        na_columns = [col for col in df.columns if df[col].isnull().sum() > 0]
        n_miss = df[na_columns].isnull().sum().sort_values(ascending=False)
        ratio = (df[na_columns].isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)
        missing_df = pd.concat([n_miss, pd.Series(np.round(ratio, 2), index=n_miss.index)], axis=1, keys=["n_miss", "ratio"])
        print(missing_df, end="\n")

        if na_name:
            return na_columns
        

    def missing_vs_target(self, df: pd.DataFrame, target: str, na_columns: list[str]):
        temp_df = df.copy()

        for col in na_columns:
            temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
        
        na_flags = temp_df.loc[:, temp_df.columns.str.contains('_NA_')].columns

        for col in na_flags:
            print(pd.DataFrame({'TARGET_MEAN': temp_df.groupby(col)[target].mean(),
                                'Count': temp_df.groupby(col)[target].count()}), end="\n\n\n")         


    def target_summary_with_num(self, df: pd.DataFrame, target: str, numerical_col: str):
        print(df.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


    def target_summary_with_cat(self, df: pd.DataFrame, target: str, categorical_col: str):
        print(pd.DataFrame({"TARGET_MEAN": df.groupby(categorical_col)[target].mean()}), end="\n\n\n")


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
    