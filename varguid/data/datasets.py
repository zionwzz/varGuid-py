"""
Dataset utilities for VarGuid package.
"""

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state


def generate_cobra2d(n_samples=500, n_features=15, noise_std=0.1, correlation=0.0, random_state=1):
    """
    Generate the cobra2d dataset with nonlinear and interaction relationships.
    
    This function replicates the cobra2 simulation from the R package, generating
    a dataset with known nonlinear relationships and interactions.
    
    Parameters
    ----------
    n_samples : int, default=500
        Number of samples to generate.
    n_features : int, default=15
        Number of features to generate (minimum 10).
    noise_std : float, default=0.1
        Standard deviation of the noise term.
    correlation : float, default=0.0
        Correlation parameter for generating correlated features.
    random_state : int, default=1
        Random seed for reproducibility.
        
    Returns
    -------
    dict : Dictionary containing:
        - 'data': DataFrame with features (x1, x2, ..., x_n_features) and target (y)
        - 'formula': String representation of the true generating function
        - 'formula_statsmodels': String for use with statsmodels
    """
    rng = check_random_state(random_state)
    
    # Ensure minimum number of features
    n_features = max(10, n_features)
    
    # Generate correlated uniform random variables in [-1, 1]
    if correlation > 0:
        # Generate correlated normal variables first, then transform
        mean = np.zeros(n_features)
        cov = np.eye(n_features) * (1 - correlation) + correlation
        normal_vars = rng.multivariate_normal(mean, cov, n_samples)
        
        # Transform to uniform via CDF
        from scipy.stats import norm
        X = norm.cdf(normal_vars)  # Now uniform [0,1]
        X = 2 * X - 1  # Transform to [-1, 1]
    else:
        # Generate independent uniform variables
        X = rng.uniform(-1, 1, size=(n_samples, n_features))
    
    # True generating function: y = x1*x2 + x3^2 - x4*x7 + x8*x10 - x6^2 + noise
    # Note: Using 0-based indexing, so this becomes:
    # y = x0*x1 + x2^2 - x3*x6 + x7*x9 - x5^2 + noise
    y_true = (X[:, 0] * X[:, 1] +           # x1 * x2
              X[:, 2] ** 2 -                # x3^2  
              X[:, 3] * X[:, 6] +           # -x4 * x7
              X[:, 7] * X[:, 9] -           # x8 * x10
              X[:, 5] ** 2)                 # -x6^2
    
    # Add noise
    noise = rng.normal(0, noise_std, n_samples)
    y = y_true + noise
    
    # Create DataFrame
    feature_names = [f'x{i+1}' for i in range(n_features)]
    data = pd.DataFrame(X, columns=feature_names)
    data['y'] = y
    
    # Formula strings
    formula = "x1 * x2 + x3 ^ 2 - x4 * x7 + x8 * x10 - x6 ^ 2"
    formula_statsmodels = "I(x1 * x2) + I(x3 ** 2) + I(-x4 * x7) + I(x8 * x10) + I(-x6 ** 2)"
    
    return {
        'data': data,
        'formula': formula,
        'formula_statsmodels': formula_statsmodels
    }


def load_cobra2d():
    """
    Load the default cobra2d dataset.
    
    Returns
    -------
    dict : Dictionary containing the cobra2d dataset and metadata
    """
    return generate_cobra2d()


# Create a default dataset instance for backward compatibility
_default_cobra2d = generate_cobra2d()
cobra2d = _default_cobra2d['data']
