import os
import sys
import logging

import pandas as pd
import click
import joblib

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

@click.command("predict")
@click.argument("input_dir")
@click.argument("model_dir")
@click.argument("output_dir")
def predict(input_dir: str, model_dir: str, output_dir: str):
    df = pd.read_csv(os.path.join(input_dir, "data.csv"))
    logger.debug(df)

    clf = joblib.load(os.path.join(model_dir, "model.pkl"))
    preds = pd.Series(clf.predict(df))

    preds.to_csv(os.path.join(output_dir, "predictions.csv"))


if __name__ == "__main__":
    predict()
