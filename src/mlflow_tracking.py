import mlflow
import mlflow.data
import mlflow.models
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from evaluation import evaluate_model, plot_confusion_matrix

import logging
import os
import datetime
import pandas as pd


# =========================================================
# Setup / Restore Experiment
# =========================================================
def setup_mlflow_experiment(
    experiment_name: str,
    tracking_uri: str = "http://localhost:5000"
) -> str:
    """
    Setup or restore an MLflow experiment.
    Returns the experiment_id.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    exp = client.get_experiment_by_name(experiment_name)

    # Restore if soft-deleted
    if exp and exp.lifecycle_stage == "deleted":
        client.restore_experiment(exp.experiment_id)

    exp = mlflow.set_experiment(experiment_name)

    return exp.experiment_id


# =========================================================
# Log Model + Metrics + Artifacts
# =========================================================
def log_model_with_mlflow(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    exp_id: str,
    labels=None,
    stage: str = "baseline"
):
    """
    Logs a model, metrics, dataset, confusion matrix, and registers the model in MLflow
    with a clean, organized folder structure.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{model_name}-{stage}-{timestamp}"

    with mlflow.start_run(
        experiment_id=exp_id,
        run_name=run_name
    ) as run:

        logging.info(f"\n========== Logging '{model_name}' ({stage}) ==========")

        # ----------------------------
        # Tags
        # ----------------------------
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("stage", stage)
        mlflow.set_tag("project", "hand-gesture-classification")

        # ----------------------------
        # Log GridSearchCV results (flattened)
        # ----------------------------
        if hasattr(model, "cv_results_"):
            for i, params in enumerate(model.cv_results_["params"]):
                mlflow.log_params({f"trial_{i}_{k}": v for k, v in params.items()})
                mlflow.log_metric(f"trial_{i}_mean_test_score", model.cv_results_["mean_test_score"][i])
                mlflow.log_metric(f"trial_{i}_std_test_score", model.cv_results_["std_test_score"][i])

        # ----------------------------
        # Evaluate Best Model
        # ----------------------------
        accuracy, f1, auc, precision, recall, pred = evaluate_model(model, X_test, y_test)

        # Log best params if available
        if hasattr(model, "best_params_"):
            mlflow.log_params(model.best_params_)

        # Log main metrics
        mlflow.log_metrics({
            "Mean_CV_score": getattr(model, "best_score_", None),
            "Accuracy": accuracy,
            "F1_score": f1,
            "AUC": auc,
            "Precision": precision,
            "Recall": recall
        })

        # ----------------------------
        # Log Dataset (Testing)
        # ----------------------------
        os.makedirs("tmp_dataset", exist_ok=True)
        X_path = os.path.join("tmp_dataset", "X_test.csv")
        y_path = os.path.join("tmp_dataset", "y_test.csv")
        X_test.to_csv(X_path, index=False)
        y_test.to_csv(y_path, index=False)
        mlflow.log_artifact(X_path, artifact_path="dataset")
        mlflow.log_artifact(y_path, artifact_path="dataset")
        # Clean up
        os.remove(X_path)
        os.remove(y_path)
        os.rmdir("tmp_dataset")

        # ----------------------------
        # Log Confusion Matrix
        # ----------------------------
        if labels is not None:
            cm_path = f"{model_name}_confusion_matrix.png"
            plot_confusion_matrix(
                y_test,
                pred,
                labels,
                title=f"{model_name} Confusion Matrix",
                save_path=cm_path
            )
            mlflow.log_artifact(cm_path, artifact_path="metrics")
            os.remove(cm_path)

        # ----------------------------
        # Log & Register Model
        # ----------------------------
        signature = mlflow.models.infer_signature(X_test, y_test)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=model_name,
            signature=signature,
            input_example=X_test.iloc[[0]],
            registered_model_name=model_name
        )

        # ----------------------------
        # Console Summary
        # ----------------------------
        logging.info(f"Run ID: {run.info.run_id}")
        logging.info(f"Model '{model_name}' successfully logged under stage '{stage}'")
        logging.info(f"Metrics: Accuracy={accuracy:.3f}, F1={f1:.3f}, AUC={auc:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
        logging.info(f"Artifacts saved under structured folders: dataset/, metrics/, models/\n")