import os

import pandas as pd
import click
import joblib
from sklearn.linear_model import LogisticRegression


def serialize_model(model: LogisticRegression, output: str) -> str:
    """Serialize model from configs"""
    with open(output, "wb") as file:
        joblib.dump(model, file)
    return output


@click.command("train")
@click.argument("input_dir")
@click.argument("output_dir")
def train(input_dir: str, output_dir: str):
    df = pd.read_csv(os.path.join(input_dir, "train.csv"))
    feats = df.drop(['target'], axis=1)
    target = df['target']

    clf = LogisticRegression()
    clf.fit(feats, target)

    path_to_model = serialize_model(clf, os.path.join(output_dir, "model.pkl"))


if __name__ == "__main__":
    train()
