import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, fbeta_score, classification_report, confusion_matrix
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Tuple
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
    def __init__(self):
        pass

    def eval_regression_model(self, X_test: pd.DataFrame, y_test: pd.Series, model) -> Tuple[float, float, float]:
        y_pred = model.predict(X_test)
        
        score = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        print("R2: ", score, "MSE: ", mse, "MAE: ", mae)

        return score, mae, mse

    def eval_class_model(self, X_test, y_test, model, visualize: bool = False, beta_param: float = 0.5) -> Tuple[float, float, float, float, float]:
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
    

    def base_models(self, X, y, scoring="roc_auc"):
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






     