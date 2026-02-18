import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a single model and return metrics as a dictionary
    """
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }
    return metrics, y_pred

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

def evaluate_models(models, X_test, y_test, labels):
    """
    Evaluate multiple models and print metrics + confusion matrix
    """
    results = {}
    for name, model in models.items():
        print(f"Evaluating {name}...")
        metrics, y_pred = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        plot_confusion_matrix(y_test, y_pred, labels, title=f"{name} Confusion Matrix")
        print(f"{name} metrics: {metrics}\n")
    return results
