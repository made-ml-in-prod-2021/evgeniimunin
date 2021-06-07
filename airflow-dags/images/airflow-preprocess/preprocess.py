import os

import pandas as pd
import click

from sklearn.preprocessing import StandardScaler

@click.command("preprocess")
@click.argument("input_dir")
@click.argument("output_dir")
def preprocess(input_dir: str, output_dir: str):
    feats = pd.read_csv(os.path.join(input_dir, "data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))

    scaler = StandardScaler()
    feats = pd.DataFrame(scaler.fit_transform(feats), columns=feats.columns)

    data = pd.concat([feats, target], axis=1)
    data.to_csv(os.path.join(output_dir, "processed.csv"), index=False)

if __name__ == "__main__":
    preprocess()
