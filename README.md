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

A comprehensive class for dataframe analysis providing methods to examine and understand the structure, content, and statistics of pandas DataFrames:

- `analyze_df(df)`: Provides a comprehensive analysis including columns, info, describe, duplicates, unique values, and value counts for each column.
- `check_df(df, head=5)`: Shows basic dataframe information including shape, data types, head, tail, null values, and quantiles.
- `missing_values_table(df, na_name=False)`: Creates a table showing columns with missing values, count, and percentage. Returns column names with missing values if na_name=True.
- `missing_vs_target(df, target, na_columns)`: Analyzes the relationship between missing values and target variable by creating flag columns.
- `target_summary_with_num(df, target, numerical_col)`: Summarizes target variable with numerical columns showing mean values.
- `target_summary_with_cat(df, target, categorical_col)`: Summarizes target variable with categorical columns showing mean values.
- `grab_col_names(df, cat_th=10, car_th=20)`: Identifies and categorizes columns as categorical, numerical, or categorical but cardinal based on data types and unique value thresholds.
- `correlation_for_drop(df, threshold=0.85)`: Identifies highly correlated columns that could be dropped to reduce multicollinearity.

### EvalModel

A comprehensive class for evaluating machine learning models with automatic problem type detection and support for both regression and classification:

**Model Evaluation Methods:**
- `eval_regression_model(X_test, y_test, model)`: Evaluates regression models using R², MAE, and MSE metrics.
- `eval_class_model(X_test, y_test, model, visualize=False, beta_param=0.5)`: Evaluates classification models using accuracy, recall, precision, F1, and F-beta scores with optional confusion matrix visualization.

**Automated Model Comparison:**
- `base_models_auto(X, y, scoring=None)`: Automatically detects problem type (regression/classification) and runs appropriate models with suitable scoring metrics.
- `base_models(X, y, scoring="roc_auc")`: Runs and evaluates multiple base classification models (LR, KNN, SVC, CART, RF, AdaBoost, GBM). Automatically handles multiclass problems.
- `base_regression_models(X, y, scoring="neg_mean_squared_error")`: Runs and evaluates multiple regression models (Linear, Ridge, Lasso, KNN, CART, RF, AdaBoost, GBM, XGBoost).
- `base_tree_models(X, y, scoring=None)`: Runs tree-based models with automatic problem type detection (CART, RF, AdaBoost, GBM, XGBoost).

**Utility Methods:**
- `_is_regression_problem(y)`: Private method that determines if the target variable represents a regression or classification problem.

### HyperParameterTuning

A class for hyperparameter optimization with predefined parameter grids for popular machine learning models:

**Search Methods:**
- `create_params_dict(keys, values)`: Creates a parameter dictionary for tuning from separate lists of keys and values.
- `grid_search(model, X_train, y_train, params_dict, scoring="accuracy", n_jobs=-1)`: Performs exhaustive grid search for hyperparameter tuning.
- `randomized_search(model, params_dict, X_train, y_train, cv=5, n_iter=10, scoring="accuracy")`: Performs randomized search cross-validation for efficient hyperparameter tuning.

**Advanced Optimization:**
- `hyperparameter_optimization(X, y, cv=3, scoring="roc_auc")`: Optimizes hyperparameters for multiple models (KNN, CART, RF, XGBoost, LightGBM, AdaBoost) and returns the best models. Automatically handles categorical target encoding and multiclass problems.
- `voting_classifier(best_models, X, y)`: Creates a soft voting classifier from the best models (KNN, RF, LightGBM) and evaluates it using accuracy, F1, and ROC_AUC metrics.

### PreprocessDataFrame

A class for data preprocessing operations focusing on outlier handling and categorical encoding:

**Outlier Detection and Handling:**
- `outlier_thresholds(df, col_name, q1=0.25, q3=0.75)`: Calculates outlier thresholds using the IQR method with customizable quartile values.
- `check_outlier(df, col_name)`: Checks if outliers exist in a column using the IQR method.
- `grab_outliers(df, col_name, index=False)`: Retrieves and displays outliers from a column with optional index return.
- `remove_outlier(df, col_name)`: Removes outliers from a column and returns the cleaned dataframe.
- `replace_with_thresholds(df, variable)`: Replaces outliers with threshold values (caps outliers) - modifies dataframe in-place.

**Encoding:**
- `one_hot_encoder(df, categorical_cols, drop_first=False)`: One-hot encodes categorical columns with option to drop first category for each feature.

### SetupDataFrame

A class for loading datasets from predefined directories with hardcoded paths:

- `setup_ml(file_dir)`: Loads machine learning datasets from the ml-data directory (`/Users/user/Desktop/Projects/data-science/ml-data`).
- `setup_ds(file_dir)`: Loads data science datasets from the data-ds directory (`/Users/user/Desktop/Projects/data-science/data-ds`).

**Note:** This class uses hardcoded absolute paths and assumes CSV format. Paths may need to be updated based on your system configuration.

### VisualizeDataFrame

A class for data visualization with various plotting methods for exploratory data analysis:

**Summary and Distribution Plots:**
- `cat_summary(df, col_name, plot=False)`: Summarizes categorical columns with value counts and ratios, optionally creates count plots.
- `num_summary(df, numerical_col, plot=False)`: Summarizes numerical columns with descriptive statistics including custom quantiles, optionally creates histograms.
- `plot_all_histograms(df, title_prefix="")`: Creates a grid of histograms for all numerical columns in the dataframe with KDE overlays.

**Advanced Visualizations:**
- `plot_3d(df, data_x, data_y, data_z, color)`: Creates interactive 3D scatter plots using plotly with color coding.
- `barplot_maker(df, cat_x, cat_y, title)`: Creates customized bar plots with proper formatting and error handling.
- `boxplot_maker(df, cat_x, cat_y)`: Creates box plots for categorical vs numerical variable analysis.
- `subplot_maker(df, num_cols)`: Creates a grid of density plots (KDE) for multiple numerical columns.
- `scatterplot_maker(df, data_x, data_y, data_hue)`: Creates scatter plots with hue for three-variable relationships.

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

# Comprehensive dataset analysis
analyze_df.check_df(df)
analyze_df.analyze_df(df)

# Get column names by type
cat_cols, num_cols, cat_but_car = analyze_df.grab_col_names(df)

# Check for missing values
na_columns = analyze_df.missing_values_table(df, na_name=True)

# Visualize the data
visualize_df.plot_all_histograms(df, "Distribution of")
for col in num_cols:
    visualize_df.num_summary(df, col, plot=True)

# Check and handle outliers
for col in num_cols:
    if preprocess_df.check_outlier(df, col):
        print(f"Outliers found in {col}")
        preprocess_df.grab_outliers(df, col)
        # Option 1: Remove outliers
        # df = preprocess_df.remove_outlier(df, col)
        # Option 2: Cap outliers with thresholds
        preprocess_df.replace_with_thresholds(df, col)

# Preprocess the data - One-hot encode categorical columns
if cat_cols:
    df = preprocess_df.one_hot_encoder(df, cat_cols)

# Prepare features and target
X = df.drop("target", axis=1)
y = df["target"]

# Automatically evaluate multiple models based on problem type
print("=== Automatic Model Selection ===")
eval_model.base_models_auto(X, y)

# For classification problems specifically
print("\n=== Classification Models ===")
eval_model.base_models(X, y, scoring="accuracy")

# For regression problems specifically
# eval_model.base_regression_models(X, y, scoring="neg_mean_squared_error")

# Tree-based models only
print("\n=== Tree-based Models ===")
eval_model.base_tree_models(X, y)

# Train and evaluate a specific model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model with detailed metrics
accuracy, recall, precision, f1, f_beta = eval_model.eval_class_model(
    X_test, y_test, model, visualize=True
)

# Advanced hyperparameter optimization
print("\n=== Hyperparameter Optimization ===")
best_models = hp_tuning.hyperparameter_optimization(X, y, cv=3, scoring="accuracy")

# Create voting classifier from best models
voting_clf = hp_tuning.voting_classifier(best_models, X, y)

# Advanced visualizations
if len(num_cols) >= 3:
    visualize_df.plot_3d(df, num_cols[0], num_cols[1], num_cols[2], "target")

# Correlation analysis for feature selection
highly_corr_cols = analyze_df.correlation_for_drop(X, threshold=0.9)
print(f"Highly correlated columns to consider dropping: {highly_corr_cols}")
```

## Key Features

### 🔍 **Comprehensive Data Analysis**
- Automated dataframe inspection with detailed statistics
- Missing value analysis with target correlation
- Intelligent column type detection (categorical, numerical, cardinal)
- Correlation analysis for feature selection

### 🤖 **Smart Model Evaluation**
- **Automatic problem type detection** (regression vs classification)
- **Multi-model comparison** with cross-validation
- Support for both **binary and multiclass** classification
- Specialized **tree-based model evaluation**
- **Regression model benchmarking** with multiple algorithms

### ⚙️ **Advanced Hyperparameter Optimization**
- **Grid search and randomized search** implementations
- **Pre-configured parameter grids** for popular ML algorithms
- **Automated hyperparameter tuning** for multiple models
- **Voting classifier** creation from best-performing models
- Support for **custom scoring metrics**

### 🛠️ **Robust Data Preprocessing**
- **IQR-based outlier detection** with customizable thresholds
- **Flexible outlier handling** (removal or capping)
- **One-hot encoding** with drop_first option
- **Comprehensive outlier analysis** with visualization options

### 📊 **Rich Data Visualization**
- **Interactive 3D plotting** with Plotly
- **Automated histogram generation** for all numerical columns
- **Customizable categorical summaries** with plotting options
- **Advanced subplot creation** for multiple variables
- **Error handling** for missing columns in visualizations

### 🚀 **Production-Ready Features**
- **Type hints** throughout all classes for better code quality
- **Comprehensive error handling** and validation
- **Modular design** for easy integration
- **Automatic multiclass problem handling**
- **Extensible architecture** for custom implementations

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
