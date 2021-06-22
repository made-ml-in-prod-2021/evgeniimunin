import os

import pandas as pd
import click
import joblib

from sklearn.preprocessing import StandardScaler


def serialize_model(model: StandardScaler, output: str) -> str:
    """Serialize model from configs"""
    with open(output, "wb") as file:
        joblib.dump(model, file)
    return output

@click.command("preprocess")
@click.argument("input_dir")
@click.argument("output_dir")
@click.argument("model_dir")
def preprocess(input_dir: str, output_dir: str):
    feats = pd.read_csv(os.path.join(input_dir, "data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))

    scaler = StandardScaler()
    feats = pd.DataFrame(scaler.fit_transform(feats), columns=feats.columns)

    data = pd.concat([feats, target], axis=1)
    data.to_csv(os.path.join(output_dir, "processed.csv"), index=False)

    path_to_model = serialize_model(scaler, os.path.join(output_dir, "scaler.pkl"))

if __name__ == "__main__":
    preprocess()
