# Data Science and Machine Learning Project

This repository contains a collection of data science and machine learning notebooks, utility classes, and datasets organized for educational and practical purposes.

## Directory Structure

- **`data-ds/`**: Contains datasets for general data science tasks including CSV and Excel files
  - Various datasets such as WineQT, googleplaystore data, employee data, weather data, etc.

- **`introduction/`**: Introductory notebooks for data science concepts
  - `feature_eng.ipynb`: Feature engineering techniques
  - `main.ipynb`: Main introductory notebook
  - `pandas_playground.ipynb`: Exploration of pandas functionalities
  - `visualizing_data.ipynb`: Data visualization techniques
  - `implementation/`: Implementation of introductory concepts

- **`ml-data/`**: Datasets for machine learning tasks
  - Various datasets for classification, regression, and other machine learning tasks
  - Includes datasets like iris, diamonds, diabetes, heart data, housing data, etc.

- **`ml-hw/`**: Machine learning homework assignments
  - `car_price.ipynb`: Car price prediction homework
  - `data/`: Data directory for homework assignments

- **`ml-notebooks/`**: Main machine learning notebooks for different algorithms
  - `bayes.ipynb`: Naive Bayes algorithms
  - `decision_tree.ipynb`: Decision tree algorithms
  - `ensemble_learning.ipynb`: Ensemble learning methods
  - `knn.ipynb`: K-Nearest Neighbors algorithm
  - `logistic_reg.ipynb`: Logistic regression
  - `regression.ipynb`: Regression techniques
  - `ridge_lasso_elasticnet.ipynb`: Regularized regression methods
  - `svm.ipynb`: Support Vector Machines
  - `implementation/`: Implementation examples for these algorithms

- **`ml-solved-notebooks/`**: Completed examples of machine learning notebooks
  - Contains numbered notebooks covering various algorithms and visualization techniques

- **`notebooks-ds/`**: Additional data science notebooks

- **`utils/`**: Utility classes for data science and machine learning tasks

## Utils Module

The `utils` directory contains several Python classes that provide utility functions for data science and machine learning tasks. Here's a detailed explanation of each class:

### AnalyzeDataFrame

A class for comprehensive dataframe analysis with the following methods:

- `analyze_df(df)`: Provides a comprehensive analysis of the dataframe including columns, info, describe, duplicates, unique values, and value counts.
- `check_df(df, head=5)`: Shows basic dataframe information including shape, data types, head, tail, null values, and quantiles.
- `missing_values_table(df, na_name=False)`: Creates a table showing columns with missing values, count, and percentage of missing values.
- `missing_vs_target(df, target, na_columns)`: Analyzes the relationship between missing values and target variable.
- `target_summary_with_num(df, target, numerical_col)`: Summarizes target variable with numerical columns.
- `target_summary_with_cat(df, target, categorical_col)`: Summarizes target variable with categorical columns.
- `grab_col_names(df, cat_th=10, car_th=20)`: Identifies and categorizes columns as categorical, numerical, or categorical but cardinal.
- `correlation_for_drop(df, threshold=0.85)`: Identifies highly correlated columns that could be dropped.

### EvalModel

A class for evaluating machine learning models with the following methods:

- `eval_regression_model(X_test, y_test, model)`: Evaluates regression models using R², MAE, and MSE.
- `eval_class_model(X_test, y_test, model, visualize=False, beta_param=0.5)`: Evaluates classification models using accuracy, recall, precision, F1, and F-beta scores.
- `base_models(X, y, scoring="roc_auc")`: Runs and evaluates multiple base classification models on a dataset.

### HyperParameterTuning

A class for hyperparameter optimization with the following methods:

- `create_params_dict(keys, values)`: Creates a parameter dictionary for tuning.
- `grid_search(model, X_train, y_train, params_dict, scoring="accuracy", n_jobs=-1)`: Performs grid search for hyperparameter tuning.
- `randomized_search(model, params_dict, X_train, y_train, cv=5, n_iter=10, scoring="accuracy")`: Performs randomized search for hyperparameter tuning.
- `hyperparameter_optimization(X, y, cv=3, scoring="roc_auc")`: Optimizes hyperparameters for multiple models and returns the best models.
- `voting_classifier(best_models, X, y)`: Creates a voting classifier from the best models.

### PreprocessDataFrame

A class for data preprocessing with the following methods:

- `outlier_thresholds(df, col_name, q1=0.25, q3=0.75)`: Calculates outlier thresholds for a column.
- `chech_outlier(df, col_name)`: Checks if outliers exist in a column.
- `grab_outliers(df, col_name, index=False)`: Retrieves outliers from a column.
- `remove_outlier(df, col_name)`: Removes outliers from a column.
- `replace_with_thresholds(df, variable)`: Replaces outliers with threshold values.
- `one_hot_encoder(df, categorical_cols, drop_first=False)`: One-hot encodes categorical columns.

### SetupDataFrame

A class for loading datasets with the following methods:

- `setup_ml(file_dir)`: Loads machine learning datasets from the ml-data directory.
- `setup_ds(file_dir)`: Loads data science datasets from the data-ds directory.

### VisualizeDataFrame

A class for data visualization with the following methods:

- `cat_summary(df, col_name, plot=False)`: Summarizes categorical columns and optionally plots them.
- `num_summary(df, numerical_col, plot=False)`: Summarizes numerical columns and optionally plots histograms.
- `plot_3d(df, data_x, data_y, data_z, color)`: Creates 3D scatter plots.
- `barplot_maker(df, cat_x, cat_y, title)`: Creates bar plots.
- `boxplot_maker(df, cat_x, cat_y)`: Creates box plots.
- `subplot_maker(df, num_cols)`: Creates subplots for multiple numerical columns.
- `scatterplot_maker(df, data_x, data_y, data_hue)`: Creates scatter plots with hue.

## Usage Example

```python
# Import the utility classes
from utils.SetupDataFrame import SetupDataFrame
from utils.AnalyzeDataFrame import AnalyzeDataFrame
from utils.PreprocessDataFrame import PreprocessDataFrame
from utils.VisualizeDataFrame import VisualizeDataFrame
from utils.EvalModel import EvalModel
from utils.HyperParameterTuning import HyperParameterTuning

# Initialize the classes
setup_df = SetupDataFrame()
analyze_df = AnalyzeDataFrame()
preprocess_df = PreprocessDataFrame()
visualize_df = VisualizeDataFrame()
eval_model = EvalModel()
hp_tuning = HyperParameterTuning()

# Load a dataset
df = setup_df.setup_ml("11-iris.csv")

# Analyze the dataset
analyze_df.check_df(df)

# Get column names by type
cat_cols, num_cols, cat_but_car = analyze_df.grab_col_names(df)

# Visualize numerical columns
for col in num_cols:
    visualize_df.num_summary(df, col, plot=True)

# Preprocess the data - One-hot encode categorical columns
df = preprocess_df.one_hot_encoder(df, cat_cols)

# Train a model and evaluate it
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
accuracy, recall, precision, f1, f_beta = eval_model.eval_class_model(X_test, y_test, model, visualize=True)

# Tune hyperparameters
best_models = hp_tuning.hyperparameter_optimization(X, y)
```

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- lightgbm
- plotly
