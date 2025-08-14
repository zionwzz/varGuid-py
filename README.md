# VarGuid: Variance Guided Regression Models for Heteroscedastic Data

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v2+](https://img.shields.io/badge/License-GPL%20v2+-blue.svg)](https://www.gnu.org/licenses/gpl-2.0)

A Python implementation of advanced regression techniques to address heteroscedasticity in linear models. This package is a Python port of the original R package `varGuid` by Sibei Liu and Min Lu.

## Features

The package implements the following algorithm:

1. **Iteratively Reweighted Least Squares (IRLS)**: Robust coefficient estimation under linear variance patterns

## Original Authors & Research

**Original R Package Authors:**
- **Sibei Liu** - PhD Student in Biostatistics, University of Miami (sxl4188@miami.edu)
- **Min Lu** - Research Associate Professor, University of Miami (m.lu6@umiami.edu)

**Python Implementation:**
- **Zihao Wang** - PhD Student in Biostatistics, University of Miami (zxw832@miami.edu)

## Reference

Liu S. and Lu M. (2025) Variance guided regression models for heteroscedastic data (under review)

## Acknowledgments

This Python package is a port of the original R package [`varGuid`](https://github.com/luminwin/varGuid) developed by Sibei Liu and Min Lu at the University of Miami. The Python implementation maintains the same methodology and algorithms while adapting to the Python scientific computing ecosystem.

**Original R Package**: https://github.com/luminwin/varGuid

## Installation

### Prerequisites

The package requires Python 3.8+ and depends on several scientific computing libraries:

```bash
pip install numpy scipy scikit-learn pandas matplotlib cvxpy statsmodels
```

### Install VarGuid

#### Option 1: Install from PyPI (when available)
```bash
pip install varguid
```

#### Option 2: Install from source
```bash
git clone https://github.com/zionwzz/varGuid-py.git
cd varGuid-py
pip install -e .
```

## Quick Start

```python
import numpy as np
from varguid import VarGuidRegressor, load_cobra2d
from varguid.utils.helpers import train_test_split_by_indices, create_feature_target_split

# Load example dataset
dataset = load_cobra2d()
data = dataset['data']

# Split into train/test sets
train_data, test_data, _, _ = train_test_split_by_indices(data, test_size=0.4, random_state=1)
X_train, y_train = create_feature_target_split(train_data)
X_test, y_test = create_feature_target_split(test_data)

# Step 1: Fit VarGuid regression with IRLS
varguid_model = VarGuidRegressor(max_iter=10, use_lasso=False)
varguid_model.fit(X_train.values, y_train.values)

# View results
print("VarGuid Coefficients:")
results = varguid_model.get_inference_results()
print(results['coefficients'])

# Make predictions
y_pred = varguid_model.predict(X_test.values)
rmse = np.sqrt(np.mean((y_test.values - y_pred) ** 2))
print(f"RMSE: {rmse:.4f}")
```

## Advanced Usage

### Using Lasso Regularization

```python
# Fit VarGuid with Lasso regularization
varguid_lasso = VarGuidRegressor(
    max_iter=10,
    use_lasso=True,
    alpha=1.0,
    cv_folds=5
)
varguid_lasso.fit(X_train.values, y_train.values)
```

## API Reference

### Core Classes

#### `VarGuidRegressor`

Main regression class implementing the VarGuid method.

**Parameters:**
- `max_iter` (int): Maximum number of IRLS iterations (default: 10)
- `step` (float): Scale parameter for data weights (default: 1.0)
- `tol` (float): Convergence tolerance (default: 1e-10)
- `use_lasso` (bool): Use Lasso regularization instead of OLS (default: False)
- `alpha` (float): Lasso regularization strength (default: 1.0)
- `cv_folds` (int): Cross-validation folds for Lasso (default: 10)

**Methods:**
- `fit(X, y)`: Fit the model
- `predict(X)`: Make predictions
- `get_inference_results()`: Get coefficient estimates and diagnostics


### Utility Functions

#### Data Functions
- `load_cobra2d()`: Load the cobra2d example dataset
- `generate_cobra2d(n_samples, n_features, ...)`: Generate cobra2d dataset with custom parameters

#### Helper Functions
- `train_test_split_by_indices(data, test_size, random_state)`: Split data using random indices
- `create_feature_target_split(data, target_column)`: Separate features and target
- `evaluate_predictions(y_true, predictions_dict)`: Compute RMSE for multiple methods
- `compute_rmse(y_true, y_pred)`: Compute Root Mean Square Error

### Getting Help

- **Issues**: Report bugs on [GitHub Issues](https://github.com/zionwzz/varGuid-py/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/zionwzz/varGuid-py/discussions)

## License

This project is licensed under the GNU General Public License v2.0 or later (GPL-2.0-or-later). See the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite the original paper:

```bibtex
@article{liu2025varguid,
  title={Variance guided regression models for heteroscedastic data},
  author={Liu, Sibei and Lu, Min},
  journal={Under review},
  year={2025}
}
```

## Contact

- **Python Implementation**: Zihao Wang (zxw832@miami.edu)
- **Original R Package & Research**: Sibei Liu (sxl4188@miami.edu), Min Lu (m.lu6@umiami.edu)
