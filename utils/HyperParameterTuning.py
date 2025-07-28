from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

class HyperParameterTuning:
    def __init__(self):
        self.knn_params = {"n_neighbors": range(2, 50)}
        self.cart_params = {
            "max_depth": range(1, 20),
            "min_samples_split": range(2, 30)
        }
        self.rf_params = {
            "max_depth": [8, 15, None],
            "max_features": [5, 7, "auto"],
            "min_samples_split": [15, 20],
            "n_estimators": [200, 300]
        }
        self.xgboost_params = {
            "learning_rate": [0.1, 0.01],
            "max_depth": [5, 8],
            "n_estimators": [100, 200],
            "colsample_bytree": [0.5, 1]
        }
        self.lightgbm_params = {
            "learning_rate": [0.01, 0.1],
            "n_estimators": [300, 500],
            "colsample_bytree": [0.7, 1]
        }

        self.classifiers = [
            ("KNN", KNeighborsClassifier(), self.knn_params),
            ("CART", DecisionTreeClassifier(), self.cart_params),
            ("RF", RandomForestClassifier(), self.rf_params),
            ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='logloss'), self.xgboost_params),
            ("LightGBM", LGBMClassifier(), self.lightgbm_params)
        ]

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
        """
        Perform randomized search cross-validation for hyperparameter tuning.
        """
        cv_strategy = StratifiedKFold(n_splits=cv)
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=params_dict,
            n_iter=n_iter,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=-1,
            random_state=42
        )
        random_search.fit(X_train, y_train)
        return random_search.best_score_, random_search.best_params_

    def hyperparameter_optimization(self, X, y, cv=3, scoring="roc_auc"):
        print("Hyperparameter Optimization....")
        best_models = {}

        for name, classifier, params in self.classifiers:
            print(f"########## {name} ##########")
            cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
            print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")
            
            gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=0).fit(X, y)
            final_model = classifier.set_params(**gs_best.best_params_)
            
            cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
            print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
            print(f" {name} best params: {gs_best.best_params_}", end="\n\n")
            best_models[name] = final_model

        return best_models
    

    def voting_classifier(self, best_models, X, y):
        print("Voting Classifier...")
        voting_clf = VotingClassifier(
            estimators=[
                ('KNN', best_models["KNN"]),
                ('RF', best_models["RF"]),
                ('LightGBM', best_models["LightGBM"])
            ],
            voting='soft'
        ).fit(X, y)
        cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
        print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
        print(f"F1 Score: {cv_results['test_f1'].mean()}")
        print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
        return voting_clf




