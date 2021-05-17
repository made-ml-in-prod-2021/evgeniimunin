import os
import pytest
from typing import List
import logging
import sys
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


NB_COLS = 26


@pytest.fixture()
def make_fake_dataset():
    nb_rows = np.random.randint(10, 500)
    fake_data = pd.DataFrame(np.random.normal(size=(nb_rows, NB_COLS)))
    fake_data.to_csv("train_data_sample.csv", index=False)


@pytest.fixture()
def dataset_path():
    curdir = os.path.dirname(__file__)
    logger.info(curdir)
    return os.path.join(curdir, "train_data_sample.csv")


@pytest.fixture()
def target_col():
    return "target"


@pytest.fixture()
def categorical_features() -> List[str]:
    return ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]


@pytest.fixture()
def numerical_features() -> List[str]:
    return ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]

