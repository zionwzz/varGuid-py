"""
VarGuid: Variance Guided Regression Models for Heteroscedastic Data

This Python package is a port of the original R package 'varGuid' developed by 
Sibei Liu and Min Lu at the University of Miami. It implements advanced regression 
techniques to address heteroscedasticity in linear models.

The package features two key algorithms:

1. An iteratively reweighted least squares (IRLS) method for robust coefficient 
   estimation under linear variance patterns
2. A biconvex algorithm that creates artificial grouping effects to capture 
   nonlinear relationships

Original R Package Authors: Sibei Liu and Min Lu
Python Implementation: Zihao Wang
Reference: Liu S. and Lu M. (2025) Variance guided regression models for 
           heteroscedastic data (under review)
Original R Package: https://github.com/luminwin/varGuid
"""

from .core.irls import VarGuidRegressor
from .core.clustering import ConvexClustering, ArtificialGrouping
from .data.datasets import load_cobra2d

__version__ = "0.1.0"
__author__ = "Zihao Wang (Python port)"
__original_authors__ = "Sibei Liu, Min Lu"
__email__ = "YOUR_EMAIL@example.com"

__all__ = [
    "VarGuidRegressor",
    "ConvexClustering", 
    "ArtificialGrouping",
    "load_cobra2d"
]