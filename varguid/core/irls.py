"""
Iteratively Reweighted Least Squares (IRLS) implementation for VarGuid regression.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array
import warnings


class VarGuidRegressor(BaseEstimator, RegressorMixin):
    """
    Variance Guided Regression using Iteratively Reweighted Least Squares.
    
    This estimator implements the VarGuid method for robust coefficient estimation
    under linear variance patterns using either IRLS or iteratively reweighted Lasso.
    
    Parameters
    ----------
    max_iter : int, default=10
        Maximum number of iterations for the IRLS algorithm.
    step : float, default=1.0
        Scale parameter for the data weight.
    tol : float, default=1e-10
        Tolerance parameter for convergence.
    use_lasso : bool, default=False
        Whether to use iteratively reweighted Lasso method. 
        If False, IRLS will be used.
    alpha : float, default=1.0
        Regularization strength for Lasso (only used if use_lasso=True).
    cv_folds : int, default=10
        Number of cross-validation folds for Lasso (only used if use_lasso=True).
    
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients from the VarGuid method.
    intercept_ : float
        Intercept term.
    ols_estimator_ : sklearn estimator
        OLS estimator used for initialization (when use_lasso=False).
    lasso_estimator_ : sklearn estimator  
        Lasso estimator used for initialization (when use_lasso=True).
    varguid_estimator_ : sklearn estimator
        Final VarGuid estimator from the last iteration.
    residuals_ : ndarray of shape (n_samples,)
        Residuals from the VarGuid model.
    weights_ : ndarray of shape (n_samples,)
        Final weights used in the last iteration.
    n_iter_ : int
        Number of iterations performed.
    """
    
    def __init__(self, max_iter=10, step=1.0, tol=1e-10, use_lasso=False, 
                 alpha=1.0, cv_folds=10):
        self.max_iter = max_iter
        self.step = step
        self.tol = tol
        self.use_lasso = use_lasso
        self.alpha = alpha
        self.cv_folds = cv_folds
        
    def _estimate_beta(self, X, y, weights, step):
        """Estimate coefficients with given weights."""
        sample_weights = np.exp(-step * weights)
        
        if self.use_lasso:
            # Use LassoCV for cross-validation
            n_folds = min(self.cv_folds, len(y) // 8) if len(y) // 8 >= 3 else 3
            
            estimator = LassoCV(
                alphas=None, 
                cv=n_folds,
                fit_intercept=True,
                max_iter=1000
            )
            estimator.fit(X, y, sample_weight=sample_weights)
            
        else:
            estimator = LinearRegression(fit_intercept=True)
            estimator.fit(X, y, sample_weight=sample_weights)
            
        return estimator
    
    def _estimate_weights(self, X, residuals):
        """Estimate variance weights from residuals."""
        X_squared = X ** 2
        residuals_squared = residuals ** 2
        
        if self.use_lasso:
            n_folds = min(self.cv_folds, len(residuals) // 8) if len(residuals) // 8 >= 3 else 3
            
            weight_estimator = LassoCV(
                alphas=None,
                cv=n_folds, 
                fit_intercept=True,
                max_iter=1000
            )
            weight_estimator.fit(X_squared, residuals_squared)
            
        else:
            weight_estimator = LinearRegression(fit_intercept=True)
            weight_estimator.fit(X_squared, residuals_squared)
            
        fitted_variance = weight_estimator.predict(X_squared)
        fitted_variance = np.maximum(fitted_variance, 1e-8)  # Avoid division by zero
        max_variance = np.max(fitted_variance)
        
        return fitted_variance / max_variance, weight_estimator
    
    def fit(self, X, y):
        """
        Fit the VarGuid regression model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape
        
        # Initialize with uniform weights
        weights = np.ones(n_samples)
        
        # Initial estimation (OLS or Lasso)
        initial_estimator = self._estimate_beta(X, y, weights, self.step)
        self.ols_estimator_ = initial_estimator if not self.use_lasso else None
        self.lasso_estimator_ = initial_estimator if self.use_lasso else None
        
        current_estimator = initial_estimator
        current_coef = np.concatenate([[current_estimator.intercept_], current_estimator.coef_])
        
        # IRLS iterations
        step = self.step
        
        for iteration in range(self.max_iter):
            old_coef = current_coef.copy()
            
            # Get residuals
            residuals = y - current_estimator.predict(X)
            
            # Estimate new weights
            weights, weight_estimator = self._estimate_weights(X, residuals)
            
            # Estimate new coefficients
            current_estimator = self._estimate_beta(X, y, weights, step)
            current_coef = np.concatenate([[current_estimator.intercept_], current_estimator.coef_])
            
            # Check convergence
            diff = np.sum((current_coef - old_coef) ** 2)
            
            if diff <= self.tol:
                break
            elif diff > self.tol and iteration < self.max_iter - 1:
                continue
            else:
                step *= 0.1  # Reduce step size if not converging
                
        self.varguid_estimator_ = current_estimator
        self.coef_ = current_estimator.coef_
        self.intercept_ = current_estimator.intercept_
        self.weights_ = weights
        self.residuals_ = y - current_estimator.predict(X)
        self.n_iter_ = iteration + 1
        self.weight_estimator_ = weight_estimator
        
        return self
    
    def predict(self, X):
        """
        Predict using the VarGuid model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        check_array(X)
        return self.varguid_estimator_.predict(X)
    
    def get_inference_results(self):
        """
        Get inference results including various robust standard errors.
        
        Note: This is a simplified version. For full robust inference,
        consider using statsmodels or implementing robust covariance matrices.
        
        Returns
        -------
        dict : Dictionary containing coefficient estimates and basic statistics
        """
        if not hasattr(self, 'varguid_estimator_'):
            raise ValueError("Model must be fitted before getting inference results.")
            
        results = {
            'coefficients': np.concatenate([[self.intercept_], self.coef_]),
            'n_iterations': self.n_iter_,
            'final_weights_mean': np.mean(self.weights_),
            'final_weights_std': np.std(self.weights_),
            'residuals_mse': np.mean(self.residuals_ ** 2)
        }
        
        return results
