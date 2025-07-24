from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV

class HyperParameterTuning:
    def __init__(self):
        pass

    def create_params_dict(self, keys: list, values: list) -> dict:
        params = dict(zip(keys, values))
        return params
    
    
    def grid_search( self, model,
                     X_train: pd.DataFrame, y_train: pd.Series, 
                     params_dict: dict, scoring: str = "accuracy", 
                     n_jobs: int = -1) -> Tuple[float, dict]:
        
        cv = StratifiedKFold()
        grid = GridSearchCV(estimator=model, param_grid=params_dict, cv=cv, scoring=scoring, n_jobs=n_jobs)
        
        grid.fit(X_train, y_train)

        return grid.best_score_, grid.best_params_
    

    def randomized_search(self, model, 
                          params_dict: dict, 
                          X_train: pd.DataFrame, y_train: pd.Series, 
                          cv: int = 5, n_iter: int = 10, 
                          scoring: str = "accuracy") -> Tuple[float, dict]:
        
        random_cv = RandomizedSearchCV(estimator=model, param_distributions=params_dict, n_iter=n_iter, cv=cv, scoring=scoring)
        random_cv.fit(X_train, y_train)

        return random_cv.best_score_, random_cv.best_params_




