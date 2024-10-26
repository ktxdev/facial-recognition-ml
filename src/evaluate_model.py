from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

from experiment_logger import log_experiment


def evaluate_model(model_name, pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)

    param_grid = pipeline.param_grid

    eval_metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1-score": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4)
    }

    log_experiment(model_name, pipeline.best_params_, pipeline.refit_time_, eval_metrics, param_grid)

    return eval_metrics
