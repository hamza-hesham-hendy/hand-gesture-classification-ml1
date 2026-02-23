"""
evaluation.py
=============
Model evaluation utilities for hand-gesture classification.

This module provides tools for testing model performance on held-out data.
It calculates standard classification metrics and generates visual aids
(confusion matrices) to understand misclassifications.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, precision_score, recall_score


def evaluate_model(grid_model, X_test, y_test):
    """
    Evaluate a single GridSearchCV model on the test set.

    Parameters
    ----------
    grid_model : GridSearchCV
        A fitted grid-search object (uses .best_estimator_ internally).
    X_test : array-like
        Test feature matrix.
    y_test : array-like
        True labels for the test set.

    Returns
    -------
    accuracy, f1, auc, precision, recall, y_pred : tuple
        Classification metrics and the predicted labels.
    """
    best_model = grid_model.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    return accuracy, f1, auc, precision, recall, y_pred

def plot_confusion_matrix(y_test, y_pred, labels, title="Confusion Matrix", save_path=None):
    """
    Plot a confusion matrix as a seaborn heatmap.

    Parameters
    ----------
    y_test : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    labels : list
        Ordered list of class labels for axis tick-labels.
    title : str, optional
        Plot title (default "Confusion Matrix").
    save_path : str, optional
        Path to save the plot image (default None).
    """
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()

def evaluate_models(grid_models, X_test, y_test, labels):
    """
    Evaluate multiple models, print metrics, and display confusion matrices.

    Parameters
    ----------
    grid_models : dict
        Mapping of model name (str) to fitted GridSearchCV object.
    X_test : array-like
        Test feature matrix.
    y_test : array-like
        True labels for the test set.
    labels : list
        Ordered list of class labels.

    Returns
    -------
    results : dict
        Mapping of model name to a dict of metric values.
    """
    results = {}
    for name, grid_model in grid_models.items():
        print(f"Evaluating {name}...")
        accuracy, f1, auc, precision, recall, y_pred = evaluate_model(grid_model, X_test, y_test)
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc,
            'precision': precision,
            'recall': recall
        }
        results[name] = metrics
        plot_confusion_matrix(y_test, y_pred, labels, title=f"{name} Confusion Matrix")
        print(f"{name} metrics: {metrics}\n")
    return results