import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, fbeta_score, classification_report, confusion_matrix
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Tuple, Any
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_validate
import xgboost
import warnings
warnings.filterwarnings(action='ignore')


class EvalModel:
    """
    A class for evaluating machine learning models, providing methods for assessing
    both regression and classification models with various metrics.
    """
    def __init__(self):
        pass

    def eval_regression_model(self, X_test: pd.DataFrame, y_test: pd.Series, model: Any) -> Tuple[float, float, float]:
        """
        Evaluates regression models using R², MAE, and MSE metrics.
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test target values
        model : Any
            Trained regression model with predict method
            
        Returns:
        --------
        Tuple[float, float, float]
            R² score, Mean Absolute Error, Mean Squared Error
        """
        y_pred = model.predict(X_test)
        
        score = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        print("R2: ", score, "MSE: ", mse, "MAE: ", mae)

        return score, mae, mse

    def eval_class_model(self, X_test: pd.DataFrame, y_test: pd.Series, model: Any, 
                    visualize: bool = False, beta_param: float = 0.5) -> Tuple[float, float, float, float, float]:
        """
        Evaluates classification models using accuracy, recall, precision, F1, and F-beta scores.
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test target values
        model : Any
            Trained classification model with predict method
        visualize : bool, default=False
            If True, creates a heatmap of the confusion matrix
        beta_param : float, default=0.5
            The beta parameter for F-beta score calculation
            
        Returns:
        --------
        Tuple[float, float, float, float, float]
            Accuracy, recall, precision, F1, and F-beta scores
        """
        y_pred = model.predict(X_test)

        accuracy = float(accuracy_score(y_test, y_pred))
        recall = float(recall_score(y_test, y_pred, average='weighted'))
        precision = float(precision_score(y_test, y_pred, average='weighted'))
        f1 = float(f1_score(y_test, y_pred, average='weighted'))
        f_beta = float(fbeta_score(y_test, y_pred, beta=beta_param, average='weighted'))

        if visualize:
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
            
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        
        return accuracy, recall, precision, f1, f_beta
    

    def base_models(self, X: pd.DataFrame, y: pd.Series, scoring: str = "roc_auc") -> None:
        """
        Runs and evaluates multiple base classification models on a dataset.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features dataframe
        y : pd.Series
            Target variable
        scoring : str, default="roc_auc"
            Scoring metric to use for evaluation
            
        Returns:
        --------
        None
            Prints cross-validation scores for each model
        
        Notes:
        ------
        Automatically adjusts scoring for multiclass problems, changing 'roc_auc' to 'roc_auc_ovr'
        """
        print("Base Models...")
        classifiers = [
            ('LR', LogisticRegression(max_iter=1000)),
            ('KNN', KNeighborsClassifier()),
            ('SVC', SVC(probability=True)),
            ('CART', DecisionTreeClassifier()),
            ('RF', RandomForestClassifier()),
            ('Adaboost', AdaBoostClassifier()),
            ('GBM', GradientBoostingClassifier()),
            # ('XGBoost', xgboost.XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
            # ('LightGBM', LGBMClassifier()),
            # ('CatBoost', CatBoostClassifier(verbose=False))
        ]

        # Detect multiclass and adjust scoring if needed
        n_classes = len(np.unique(y))
        if scoring == "roc_auc" and n_classes > 2:
            scoring = "roc_auc_ovr"

        for name, classifier in classifiers:
            try:
                CV_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
                print(f"{scoring}: {round(CV_results['test_score'].mean(), 4)} ({name}) ")
            except Exception as e:
                print(f"{scoring}: Error ({name}) - {e}")






     