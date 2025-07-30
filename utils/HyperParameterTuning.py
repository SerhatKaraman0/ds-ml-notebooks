from typing import Tuple, Dict, Any, List
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
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier

class HyperParameterTuning:
    """
    A class for hyperparameter optimization of machine learning models using
    grid search and randomized search techniques.
    """
    def __init__(self):
        """
        Initialize the HyperParameterTuning class with predefined parameter grids
        for various machine learning models.
        """
        self.knn_params = {"n_neighbors": range(2, 50)}
        self.cart_params = {
            "max_depth": range(1, 20),
            "min_samples_split": range(2, 30),
            "criterion": ["gini", "entropy", "log_loss"],
            "splitter": ["best", "random"],
            "max_features": ["sqrt", "log2", None]
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

        self.adaboost_params = {
            "n_estimators": [50, 70, 100, 120, 150, 200],
            "learning_rate": [0.001, 0.01, 0.1, 1, 10]
        }

        self.classifiers = [
            ("KNN", KNeighborsClassifier(), self.knn_params),
            ("CART", DecisionTreeClassifier(), self.cart_params),
            ("RF", RandomForestClassifier(), self.rf_params),
            ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='logloss'), self.xgboost_params),
            ("LightGBM", LGBMClassifier(), self.lightgbm_params),
            ("AdaBoost", AdaBoostClassifier(), self.adaboost_params)
        ]

    def create_params_dict(self, keys: list, values: list) -> dict:
        """
        Creates a parameter dictionary for tuning from separate lists of keys and values.
        
        Parameters:
        -----------
        keys : list
            List of parameter names
        values : list
            List of parameter values
            
        Returns:
        --------
        dict
            Dictionary mapping parameter names to values
        """
        params = dict(zip(keys, values))
        return params
    
    def grid_search(self, model: Any,
                   X_train: pd.DataFrame, y_train: pd.Series, 
                   params_dict: Dict[str, Any], scoring: str = "accuracy", 
                   n_jobs: int = -1) -> Tuple[float, Dict[str, Any]]:
        """
        Performs grid search for hyperparameter tuning.
        
        Parameters:
        -----------
        model : Any
            Machine learning model to tune
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target values
        params_dict : Dict[str, Any]
            Dictionary of parameters to tune
        scoring : str, default="accuracy"
            Scoring metric to use for evaluation
        n_jobs : int, default=-1
            Number of CPU cores to use (-1 means all available cores)
            
        Returns:
        --------
        Tuple[float, Dict[str, Any]]
            Best score and best parameters
        """
        
        cv = StratifiedKFold()
        grid = GridSearchCV(estimator=model, param_grid=params_dict, cv=cv, scoring=scoring, n_jobs=n_jobs)
        
        grid.fit(X_train, y_train)

        return grid.best_score_, grid.best_params_
    
    def randomized_search(self, model: Any, 
                        params_dict: Dict[str, Any], 
                        X_train: pd.DataFrame, y_train: pd.Series, 
                        cv: int = 5, n_iter: int = 10, 
                        scoring: str = "accuracy") -> Tuple[float, Dict[str, Any]]:
        """
        Performs randomized search cross-validation for hyperparameter tuning.
        
        Parameters:
        -----------
        model : Any
            Machine learning model to tune
        params_dict : Dict[str, Any]
            Dictionary of parameters to sample from
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target values
        cv : int, default=5
            Number of cross-validation folds
        n_iter : int, default=10
            Number of parameter settings to sample
        scoring : str, default="accuracy"
            Scoring metric to use for evaluation
            
        Returns:
        --------
        Tuple[float, Dict[str, Any]]
            Best score and best parameters
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

    def hyperparameter_optimization(self, X: pd.DataFrame, y: pd.Series, 
                                  cv: int = 3, scoring: str = "roc_auc") -> Dict[str, Any]:
        """
        Optimizes hyperparameters for multiple models and returns the best models.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features dataframe
        y : pd.Series
            Target variable
        cv : int, default=3
            Number of cross-validation folds
        scoring : str, default="roc_auc"
            Scoring metric to use for evaluation
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary mapping model names to best models with optimized parameters
            
        Notes:
        ------
        Automatically encodes categorical target variables and adjusts scoring for multiclass problems.
        """
        from sklearn.preprocessing import LabelEncoder
        print("Hyperparameter Optimization....")
        best_models = {}

        # Encode y if it is not numeric
        if y.dtype.kind not in 'iufc':  # i: integer, u: unsigned integer, f: float, c: complex
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
        else:
            y_encoded = y

        n_classes = len(np.unique(y_encoded))
        if scoring == "roc_auc" and n_classes > 2:
            scoring = "roc_auc_ovr"

        for name, classifier, params in self.classifiers:
            print(f"########## {name} ##########")
            cv_results = cross_validate(classifier, X, y_encoded, cv=cv, scoring=scoring)
            print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")
            
            gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=0, scoring=scoring).fit(X, y_encoded)
            final_model = classifier.set_params(**gs_best.best_params_)
            
            cv_results = cross_validate(final_model, X, y_encoded, cv=cv, scoring=scoring)
            print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
            print(f" {name} best params: {gs_best.best_params_}", end="\n\n")
            best_models[name] = final_model

        return best_models
    

    def voting_classifier(self, best_models: Dict[str, Any], X: pd.DataFrame, y: pd.Series) -> VotingClassifier:
        """
        Creates a voting classifier from the best models.
        
        Parameters:
        -----------
        best_models : Dict[str, Any]
            Dictionary mapping model names to best models
        X : pd.DataFrame
            Features dataframe
        y : pd.Series
            Target variable
            
        Returns:
        --------
        VotingClassifier
            Fitted voting classifier combining KNN, RF, and LightGBM models
            
        Notes:
        ------
        Prints accuracy, F1 score, and ROC_AUC for the voting classifier.
        Automatically selects the appropriate ROC_AUC metric for binary or multiclass problems.
        """
        print("Voting Classifier...")
        voting_clf = VotingClassifier(
            estimators=[
                ('KNN', best_models["KNN"]),
                ('RF', best_models["RF"]),
                ('LightGBM', best_models["LightGBM"])
            ],
            voting='soft'
        ).fit(X, y)
        # Determine the correct roc_auc scoring based on the number of classes
        n_classes = len(np.unique(y))
        roc_auc_score_type = "roc_auc" if n_classes == 2 else "roc_auc_ovr"
        cv_results = cross_validate(
            voting_clf, X, y, cv=3, scoring={"accuracy": "accuracy", "f1": "f1", "roc_auc": roc_auc_score_type}
        )
        print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
        print(f"F1 Score: {cv_results['test_f1'].mean()}")
        print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
        return voting_clf




