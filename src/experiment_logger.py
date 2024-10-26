import os
import json
from typing import Dict

LOG_FILE: str = "experiments/experiment_results.json"


def load_logs():
    try:
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_logs(logs):
    exp_dir = os.path.dirname(LOG_FILE)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)


def log_experiment(model_name: str,
                   best_params: Dict[str, any],
                   best_model_fit_time: float,
                   eval_metrics: Dict[str, float],
                   param_grid: Dict[str, any]):
    logs = load_logs()

    experiment_key = f"experiment_{len(logs) + 1}"

    logs[experiment_key] = {
        "model_name": model_name,
        "best_params": best_params,
        "best_fit_time_in_sec": round(best_model_fit_time, 4),
        "param_grid": param_grid,
        **eval_metrics
    }

    save_logs(logs)
    print(f"Experiment {experiment_key} logged successfully")
