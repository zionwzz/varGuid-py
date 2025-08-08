"""
Setup script for VarGuid Python package.
"""

from setuptools import setup, find_packages
import os

# Read README file
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="varguid",
    version="0.1.0",
    description="Variance Guided Regression Models for Heteroscedastic Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zionwzz/varGuid-py",
    author="Zihao Wang",
    author_email="zxw832@miami.edu",  # Replace with your actual email
    maintainer="Zihao Wang",
    maintainer_email="zxw832@miami.edu",  # Replace with your actual email
    license="GPL-2.0-or-later",
    
    # Classifiers help users find your project
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    keywords="regression heteroscedasticity clustering machine-learning statistics",
    
    # Package structure
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0", 
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "cvxpy>=1.2.0",
        "statsmodels>=0.13.0",
    ],
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
            "sphinx", 
            "sphinx-rtd-theme",
        ],
        "examples": [
            "jupyter",
            "seaborn",
        ],
    },
    
    # Include additional files
    include_package_data=True,
    package_data={
        "varguid": ["data/*.csv"],
    },
    
    # Entry points (if any)
    entry_points={
        "console_scripts": [
            # "varguid-cli=varguid.cli:main",  # Uncomment if you add CLI
        ],
    },
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/zionwzz/varGuid-py/issues",
        "Source": "https://github.com/zionwzz/varGuid-py",
        "Documentation": "https://github.com/zionwzz/varGuid-py#readme",
        "Original R Package": "https://github.com/luminwin/varGuid",
    },
)
