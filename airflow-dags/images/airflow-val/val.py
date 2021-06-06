import os
import json
from typing import Dict

import numpy as np
import pandas as pd
import click
import joblib
from sklearn.metrics import f1_score, accuracy_score


def evaluate_model(predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:
    """Evaluate model from configs"""
    return {
        "f1_score": f1_score(target, predicts, average="weighted"),
        "acc_score": accuracy_score(target, predicts),
    }

@click.command("val")
@click.argument("input_dir")
@click.argument("model_dir")
def val(input_dir: str, model_dir: str):
    df = pd.read_csv(os.path.join(input_dir, "train.csv"))
    feats = df.drop(['target'], axis=1)
    target = df['target']

    clf = joblib.load(os.path.join(model_dir, "model.clf"))
    preds = clf.predict(feats)

    metrics = evaluate_model(preds, target)

    # dump metrics to json
    with open(os.path.join(model_dir, "metrics.json"), "w") as metric_file:
        json.dump(metrics, metric_file)

if __name__ == "__main__":
    val()
