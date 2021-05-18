import logging
import sys

import click
import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


@click.command()
@click.option("--host", default="localhost")
@click.option("--port", default=8000)
@click.option("--dataset_path", default="data/heart.csv")
def predict(host, port, dataset_path):
    data = pd.read_csv(dataset_path)
    data["id"] = data.index + 1

    request_feaures = list(data.columns)
    for i in range(data.shape[0]):
        logger.info(data.columns)
        request_data = [
            x.item() if isinstance(x, np.generic) else x for x in data.iloc[i].tolist()
        ]

        logger.info(f"Request: {request_data}")
        response = requests.get(
            f"http://{host}:{port}/predict/",
            json={"data": [request_data], "features": request_feaures},
        )

        logger.info(f"Response code: {response.status_code}, body: {response.json()}")


if __name__ == "__main__":
    predict()
