import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 


class SetupDataFrame:
    def __init__(self):
        self.DS_BASE_DIR = "/Users/user/Desktop/Projects/data-science/data-ds"
        self.ML_BASE_DIR = "/Users/user/Desktop/Projects/data-science/ml-data"

    def setup_ml(self, file_dir: str) -> pd.DataFrame:
        path = os.path.join(self.ML_BASE_DIR, file_dir)
        df = pd.read_csv(path)
        return df
    
    def setup_ds(self, file_dir: str) -> pd.DataFrame:
        path = os.path.join(self.DS_BASE_DIR, file_dir)
        df = pd.read_csv(path)
        return df
    