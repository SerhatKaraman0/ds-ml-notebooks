import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px 
import seaborn as sns
import os 
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class VisualizeDataFrame:
    def __init__(self):
        pass
    
    def plot_3d(self, df:pd.DataFrame, data_x: str, data_y: str, data_z: str, color: str):
        missing_cols = [col for col in [data_x, data_y, data_z, color] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Column(s) {', '.join(missing_cols)} not found in the dataframe") 
        
        fig = px.scatter_3d(df, data_x, data_y, data_z, color)
        fig.show()

    def barplot_maker(self, df: pd.DataFrame, cat_x: str, cat_y: str, title: str):
        if not cat_x in df.columns or not cat_y in df.columns:
            raise ValueError(f"Column {cat_x} or {cat_y} not found in dataframe")
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=cat_x, y=cat_y, data=df)
        plt.title(title, fontsize=14)
        plt.xlabel(cat_x, fontsize=12)
        plt.ylabel(cat_y, fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

    def boxplot_maker(self, df: pd.DataFrame, cat_x: str, cat_y: str):
        sns.boxplot(data=df, x=cat_x, y=cat_y)
        plt.show()

    def subplot_maker(self, df: pd.DataFrame, num_cols):
        plt.figure(figsize=(20, 25))
        for i in range(0, len(num_cols)):
            plt.subplot(5, 3, i+1)
            sns.kdeplot(x=df[num_cols[i]], color="b", fill=True)
            plt.xlabel(num_cols[i], fontsize=12)
            plt.ylabel('Density', fontsize=12)
            plt.title(num_cols[i], fontsize=14)
            plt.tight_layout()
        plt.show()
    

    def scatterplot_maker(self, df: pd.DataFrame, data_x: str, data_y: str, data_hue: str):
        missing_cols = [col for col in [data_x, data_y, data_hue] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Column(s) {', '.join(missing_cols)} not found in the dataframe")
        
        sns.scatterplot(x=df[data_x], y=df[data_y], hue=df[data_hue])
        plt.show()