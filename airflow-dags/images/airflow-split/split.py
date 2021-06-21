import os

import pandas as pd
import click
from sklearn.model_selection import train_test_split

@click.command("split")
@click.argument("input_dir")
@click.argument("output_dir")
@click.argument("test_size")
def split(input_dir: str, output_dir: str, test_size: str):
    df = pd.read_csv(os.path.join(input_dir, "processed.csv"))

    train, test = train_test_split(df, test_size=float(test_size))

    train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False)


if __name__ == "__main__":
    split()
