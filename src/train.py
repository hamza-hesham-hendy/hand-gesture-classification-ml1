"""
train.py
========
Model training utilities for hand-gesture classification.

This module provides a standardized interface for training multiple 
scikit-learn classifiers using GridSearchCV for hyperparameter optimization.
The models currently supported are:
- Random Forest (Ensemble method)
- Support Vector Machine (RBF Kernel)
- Logistic Regression (Linear baseline)
"""

import os
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

logging.basicConfig(level=logging.INFO)

def train_random_forest(X_train, y_train, debug=False):
    """
    Train a Random Forest classifier using GridSearchCV.

    Random Forest is an ensemble of decision trees. It is robust to overfitting
    and handles non-linear relationships well.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix (n_samples, n_features).
    y_train : array-like
        Training labels (n_samples,).
    debug : bool, optional
        If True, use a minimal parameter grid for fast testing (default False).

    Returns
    -------
    GridSearchCV
        The fitted grid-search object. Use .best_estimator_ for the optimal model.
    """
    logging.info("Training Random Forest Classifier...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [20, 30],
        'criterion': ['gini', 'entropy'],
    }

    debug_param_grid = {
        'n_estimators': [50],
        'max_depth': [10],
        'criterion': ['gini']
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid if not debug else debug_param_grid,
        refit=True,
        verbose=1,
        cv=3,
        n_jobs=-1,
        scoring='accuracy'
    )
    model = grid.fit(X_train, y_train)
    logging.info(f"RF best params: {grid.best_params_}")
    return model


def train_svm(X_train, y_train, debug=False):
    """
    Train an SVM (RBF kernel) classifier using GridSearchCV.

    Support Vector Machines are effective in high dimensional spaces.
    The RBF (Radial Basis Function) kernel allows the model to capture
    complex non-linear boundaries.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix.
    y_train : array-like
        Training labels.
    debug : bool, optional
        If True, use a minimal parameter grid for fast testing (default False).

    Returns
    -------
    GridSearchCV
        The fitted grid-search object. Use .best_estimator_ for the optimal model.
    """
    logging.info("Training SVM...")

    param_grid = {
        'C': [1, 10, 100],
        'gamma': ['scale', 0.1, 0.01],
        'kernel': ['rbf'],
    }

    debug_param_grid = {
        'C': [1],
        'gamma': ['scale'],
        'kernel': ['rbf']
    }

    grid = GridSearchCV(
        SVC(random_state=42, probability=True),
        param_grid if not debug else debug_param_grid,
        refit=True,
        verbose=1,
        cv=3,
        n_jobs=-1,
        scoring='accuracy'
    )
    model = grid.fit(X_train, y_train)
    logging.info(f"SVM best params: {grid.best_params_}")
    return model


def train_logistic_regression(X_train, y_train, debug=False):
    """
    Train a Logistic Regression classifier using GridSearchCV.

    Logistic Regression is a simple yet powerful linear model for classification.
    It serves as an excellent baseline and provides probabilistic predictions.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix.
    y_train : array-like
        Training labels.
    debug : bool, optional
        If True, use a minimal parameter grid for fast testing (default False).

    Returns
    -------
    GridSearchCV
        The fitted grid-search object. Use .best_estimator_ for the optimal model.
    """
    logging.info("Training Logistic Regression...")

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs'],
    }

    debug_param_grid = {
        'C': [1],
        'penalty': ['l2'],
        'solver': ['lbfgs'],
    }

    grid = GridSearchCV(
        LogisticRegression(random_state=42, max_iter=2000),
        param_grid if not debug else debug_param_grid,
        refit=True,
        verbose=1,
        cv=3,
        n_jobs=-1,
        scoring='accuracy'
    )
    model = grid.fit(X_train, y_train)
    logging.info(f"LogReg best params: {grid.best_params_}")
    return model


def save_models(models, folder='models/'):
    """
    Serialize trained models to disk as .pkl files.

    Parameters
    ----------
    models : dict
        Mapping of model name (str) to fitted model object.
    folder : str, optional
        Directory to save .pkl files into (default 'models/').
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    for name, model in models.items():
        path = os.path.join(folder, f"{name}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        logging.info(f"Saved {name} -> {path}")
        