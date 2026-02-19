import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

def evaluate_model(grid_model, X_test, y_test):
    """
    Evaluate a single model and return metrics and predictions
    """
    best_model = grid_model.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    return accuracy, f1, auc, y_pred

def plot_confusion_matrix(y_test, y_pred, labels, title="Confusion Matrix"):
    """
    Plot a confusion matrix using seaborn heatmap
    """
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.show()

def evaluate_models(grid_models, X_test, y_test, labels):
    """
    Evaluate multiple models and print metrics + confusion matrix
    """
    results = {}
    for name, grid_model in grid_models.items():
        print(f"Evaluating {name}...")
        accuracy, f1, auc, y_pred = evaluate_model(grid_model, X_test, y_test)
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc
        }
        results[name] = metrics
        plot_confusion_matrix(y_test, y_pred, labels, title=f"{name} Confusion Matrix")
        print(f"{name} metrics: {metrics}\n")
    return results