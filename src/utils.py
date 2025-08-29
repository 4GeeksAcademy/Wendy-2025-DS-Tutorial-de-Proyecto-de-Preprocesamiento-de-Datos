from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import missingno as msno
import seaborn as sns
import statsmodels.api as sm
import xgboost as xgb
import joblib
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.stats.outliers_influence import variance_inflation_factor
from lazypredict.Supervised import LazyRegressor
from lightgbm import LGBMRegressor
from skopt import BayesSearchCV
from scipy.stats import zscore
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import utils as eda
import importlib

# Convert categorical columns to numeric using label encoding
def label_encode_total_data(total_data):
    label_encoders = {}
    categorical_cols = total_data.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        le = LabelEncoder()
        # Start encoding from 1 instead of 0
        total_data[col] = le.fit_transform(total_data[col]) + 1
        label_encoders[col] = le
    return total_data, label_encoders

# To revert back to original names after analysis:
def revert_label_encoding(total_data, label_encoders):
    for col, le in label_encoders.items():
        total_data[col] = le.inverse_transform(total_data[col])
    return total_data

#To remove outliers using IQR method
def remove_outliers_igr(total_data, column):
    total_data = total_data.copy()
    for col in column:
        Q1 = total_data[col].quantile(0.25)
        Q3 = total_data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        total_data = total_data[(total_data[col] >= lower_bound) & (total_data[col] <= upper_bound)]
    return total_data

# To do histogram, boxplot and scatter plot of numerical data
def plot_numerical_data(total_data):
    numerical_columns = total_data.select_dtypes(include=["int64", "float64"]).columns
    for column in numerical_columns:
        fig, ax = plt.subplots(1, 3, figsize=(18, 4))
        mean_val = np.mean(total_data[column])
        median_val = np.median(total_data[column])
        std_dev = np.std(total_data[column])
        sns.histplot(total_data[column], kde=True, ax=ax[0])
        ax[0].axvline(mean_val, color="red", linestyle="dashed", label="Mean")
        ax[0].axvline(median_val, color="green", linestyle="dashed", label="Median")
        ax[0].axvline(mean_val + std_dev, color="blue", linestyle="dashed", label="Mean ± 1 Std Dev")
        ax[0].axvline(mean_val - std_dev, color="blue", linestyle="")
        ax[0].set_title(f"Distribution of {column}")
        ax[0].legend()
        sns.boxplot(y=total_data[column], ax=ax[1])
        ax[1].axhline(mean_val, color="red", linestyle="dashed", label="Mean")
        ax[1].axhline(median_val, color="green", linestyle="dashed", label="Median")
        ax[1].axhline(mean_val + std_dev, color="blue", linestyle="dashed", label="Mean ± 1 Std Dev")
        ax[1].axhline(mean_val - std_dev, color="blue", linestyle="dashed")
        ax[1].set_title(f"Boxplot of {column}")
        ax[1].legend()
        sns.scatterplot(x=total_data.index, y=total_data[column], ax=ax[2])
        ax[2].set_title(f"Scatter plot of {column}")
        plt.tight_layout()
        fig.suptitle(f"Analysis of {column}", fontsize=16, y=1.05)
        plt.show()

#To do analysis between the variable target and the rest of the variables:
def plot_regplot_heatmap(cleaned_data, target_variable):
    numerical_columns = cleaned_data.select_dtypes(include=["int64", "float64"]).columns
    numerical_columns = [col for col in numerical_columns if col != target_variable]
    num_vars = len(numerical_columns)
    num_cols = 2
    fig, ax = plt.subplots(num_vars, num_cols, figsize=(15, 5 * num_vars))
    # Ensure ax is always 2D
    if num_vars == 1:
        ax = np.array([ax])
    for i, x_variable in enumerate(numerical_columns):
        sns.regplot(x=cleaned_data[x_variable], y=cleaned_data[target_variable], ax=ax[i, 0])
        ax[i, 0].set_title(f"Regplot of {target_variable} vs {x_variable}")
        corr_matrix = cleaned_data[[target_variable, x_variable]].corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax[i, 1], vmin=-1, vmax=1)
        ax[i, 1].set_title(f"Correlation heatmap of {target_variable} and {x_variable}")
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()

# To do correlation heatmap for numerical features
def plot_correlation_heatmap(cleaned_data, target_variable):
    plt.figure(figsize=(10, 6))
    sns.heatmap(cleaned_data.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

# To do Outlier analysis for the target variable using Z-score
def outlier_analysis_zscore(cleaned_data, target_variable):
    z_scores = zscore(cleaned_data.select_dtypes(include=["int64", "float64"]))
    target_idx = cleaned_data.columns.get_loc(target_variable)
    price_z_scores = z_scores[:, target_idx]
    outlier_mask = np.abs(price_z_scores) > 3  # Z-score threshold for outliers
    num_outliers = np.sum(outlier_mask)
    print(f"Number of outliers in '{target_variable}': {num_outliers}")
    outlier_rows = cleaned_data[outlier_mask]
    return outlier_rows

# To do VIF Variance Inflation Factor analysis:
def calculate_vif(cleaned_data):
    vif_data = pd.DataFrame()
    vif_data["feature"] = cleaned_data.columns
    vif_data["VIF"] = [variance_inflation_factor(cleaned_data.values, i) for i in range(cleaned_data.shape[1])]
    return vif_data.sort_values(by="VIF", ascending=False)

