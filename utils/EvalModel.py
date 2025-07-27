import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, fbeta_score, classification_report, confusion_matrix
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Tuple

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

    def eval_class_model(self, X_test, y_test, model, beta_param: float = 0.5) -> Tuple[float, float, float, float, float]:
        y_pred = model.predict(X_test)

        accuracy = float(accuracy_score(y_test, y_pred))
        recall = float(recall_score(y_test, y_pred, average='weighted'))
        precision = float(precision_score(y_test, y_pred, average='weighted'))
        f1 = float(f1_score(y_test, y_pred, average='weighted'))
        f_beta = float(fbeta_score(y_test, y_pred, beta=beta_param, average='weighted'))

        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        
        return accuracy, recall, precision, f1, f_beta




     