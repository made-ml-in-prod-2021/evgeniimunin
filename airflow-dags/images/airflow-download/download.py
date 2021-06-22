import os
import sys
import logging

import numpy as np
import pandas as pd
import click

SIZE = 10

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

@click.command("download")
@click.argument("output_dir")
def generate_dataset(output_dir: str):
    fake_data = pd.DataFrame()
    fake_data["age"] = np.random.normal(size=SIZE)
    fake_data["sex"] = np.random.choice([0, 1], size=SIZE)
    fake_data["cp"] = np.random.randint(0, 3, size=SIZE)
    fake_data["trestbps"] = np.random.normal(size=SIZE)
    fake_data["chol"] = np.random.normal(size=SIZE)
    fake_data["fbs"] = np.random.choice([0, 1], size=SIZE)
    fake_data["restecg"] = np.random.randint(0, 3, size=SIZE)
    fake_data["thalach"] = np.random.normal(size=SIZE)
    fake_data["exang"] = np.random.choice([0, 1], size=SIZE)
    fake_data["oldpeak"] = np.random.normal(size=SIZE)
    fake_data["slope"] = np.random.randint(0, 3, size=SIZE)
    fake_data["ca"] = np.random.randint(0, 3, size=SIZE)
    fake_data["thal"] = np.random.randint(0, 4, size=SIZE)

    logger.debug(fake_data)

    fake_data.to_csv(os.path.join(output_dir, "data.csv"), index=False)

    fake_target = pd.DataFrame()
    fake_target["target"] = np.random.choice([0, 1], size=SIZE)
    fake_target.to_csv(os.path.join(output_dir, "target.csv"), index=False)


if __name__ == "__main__":
    generate_dataset()
