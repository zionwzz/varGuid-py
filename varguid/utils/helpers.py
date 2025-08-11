"""
Utility functions for VarGuid package.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def compute_rmse(y_true, y_pred):
    """
    Compute Root Mean Square Error.
    
    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted values.
        
    Returns
    -------
    float : RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_predictions(y_true, predictions_dict):
    """
    Evaluate multiple prediction methods using RMSE.
    
    Parameters
    ----------
    y_true : array-like
        True target values.
    predictions_dict : dict
        Dictionary of prediction arrays with method names as keys.
        
    Returns
    -------
    dict : Dictionary of RMSE values for each method
    """
    rmse_results = {}
    for method, y_pred in predictions_dict.items():
        if y_pred is not None:
            rmse_results[method] = compute_rmse(y_true, y_pred)
    return rmse_results


def train_test_split_by_indices(data, test_size=0.4, random_state=None):
    """
    Split data into train/test sets using random indices (mimicking R's approach).
    
    Parameters
    ----------
    data : pandas.DataFrame
        Data to split.
    test_size : float, default=0.4
        Proportion of data to use for testing.
    random_state : int, default=None
        Random seed for reproducibility.
        
    Returns
    -------
    tuple : (train_data, test_data, train_indices, test_indices)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(data)
    n_test = int(n_samples * test_size)
    
    test_indices = np.random.choice(n_samples, size=n_test, replace=False)
    train_indices = np.setdiff1d(np.arange(n_samples), test_indices)
    
    train_data = data.iloc[train_indices].reset_index(drop=True)
    test_data = data.iloc[test_indices].reset_index(drop=True)
    
    return train_data, test_data, train_indices, test_indices


def create_feature_target_split(data, target_column='y'):
    """
    Split DataFrame into features and target.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input data.
    target_column : str, default='y'
        Name of the target column.
        
    Returns
    -------
    tuple : (X, y) where X is features and y is target
    """
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data.")
    
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    return X, y


def format_results_table(rmse_dict, title="Model Performance"):
    """
    Format RMSE results as a nice table string.
    
    Parameters
    ----------
    rmse_dict : dict
        Dictionary of RMSE values.
    title : str, default="Model Performance"  
        Title for the table.
        
    Returns
    -------
    str : Formatted table string
    """
    if not rmse_dict:
        return f"{title}\nNo results to display."
    
    # Calculate column widths
    method_width = max(len("Method"), max(len(method) for method in rmse_dict.keys()))
    rmse_width = max(len("RMSE"), 10)  # At least 10 for number formatting
    
    # Create table
    header = f"{'Method':<{method_width}} | {'RMSE':>{rmse_width}}"
    separator = "-" * len(header)
    
    lines = [title, "=" * len(title), header, separator]
    
    for method, rmse in rmse_dict.items():
        line = f"{method:<{method_width}} | {rmse:>{rmse_width}.6f}"
        lines.append(line)
    
    return "\n".join(lines)


def check_varguid_fitted(estimator):
    """
    Check if a VarGuid estimator is fitted.
    
    Parameters
    ----------
    estimator : VarGuidRegressor
        Estimator to check.
        
    Raises
    ------
    ValueError : If estimator is not fitted
    """
    if not hasattr(estimator, 'varguid_estimator_'):
        raise ValueError("VarGuid model must be fitted before use.")


def summary_statistics(data):
    """
    Compute summary statistics for a dataset.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input data.
        
    Returns
    -------
    pandas.DataFrame : Summary statistics
    """
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    summary = data[numeric_columns].describe()
    
    # Add additional statistics
    summary.loc['variance'] = data[numeric_columns].var()
    summary.loc['skewness'] = data[numeric_columns].skew()
    summary.loc['kurtosis'] = data[numeric_columns].kurtosis()
    
    return summary
