import os
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

logging.basicConfig(level=logging.INFO)

def train_random_forest(X_train, y_train, debug=False):
    logging.info("Training Random Forest...")

    rf = RandomForestClassifier(random_state=42)

    # Use different grids for debug vs full run
    if debug:
        param_grid = {'n_estimators': [200], 'max_depth': [20], 'criterion': ['gini']}
        cv = 3
    else:
        param_grid = {
            'n_estimators': [200, 400, 700],
            'max_depth': [10, 20, 30],
            'criterion': ["gini", "entropy"],
            'max_leaf_nodes': [50, 100]
        }
        cv = 5

    grid = GridSearchCV(rf, param_grid, cv=cv, n_jobs=-1, scoring='accuracy', verbose=0, return_train_score=True)
    model = grid.fit(X_train, y_train)
    logging.info(f"RF best params: {grid.best_params_}")
    return model


def train_svm(X_train, y_train, debug=False):
    logging.info("Training SVM...")

    svm = SVC(kernel='rbf', probability=True, random_state=42)

    if debug:
        param_grid = {'C': [10], 'gamma': ['scale']}
        cv = 3
    else:
        param_grid = {
            'C': [1, 10, 50],
            'gamma': ['scale', 0.01, 0.1]
        }
        cv = 5

    grid = GridSearchCV(svm, param_grid, cv=cv, n_jobs=-1, scoring='accuracy', verbose=0, return_train_score=True)
    model = grid.fit(X_train, y_train)
    logging.info(f"SVM best params: {grid.best_params_}")
    return model


def train_logistic_regression(X_train, y_train, debug=False):
    logging.info("Training Logistic Regression...")

    lr = LogisticRegression(random_state=42, solver='lbfgs', max_iter=500)

    if debug:
        param_grid = {'C': [1.0]}
        cv = 3
    else:
        param_grid = {'C': [100, 10, 1.0, 0.1, 0.01]}
        cv = 5

    grid = GridSearchCV(lr, param_grid, cv=cv, n_jobs=-1, scoring='accuracy', verbose=0, return_train_score=True)
    model = grid.fit(X_train, y_train)
    logging.info(f"LogReg best params: {grid.best_params_}")
    return model

def save_models(models, folder='models/'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for name, model in models.items():
        path = os.path.join(folder, f"{name}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        logging.info(f"Saved {name} -> {path}")