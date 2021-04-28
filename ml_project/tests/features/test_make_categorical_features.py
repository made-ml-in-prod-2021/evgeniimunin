from typing import List
import numpy as np
import pandas as pd
import pytest
import logging
import sys

from src.features.build_features import process_categorical_features

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

@pytest.fixture()
def categorical_feature() -> str:
    return 'categorical_feature'


@pytest.fixture()
def categorical_values() -> List[str]:
    return ['cat', 'dog', 'cow']


@pytest.fixture()
def categorical_values_with_nan(categorical_values: List[str]) -> List[str]:
    return categorical_values + [np.nan]


@pytest.fixture
def fake_categorical_data(
        categorical_feature: str, categorical_values_with_nan: List[str]
) -> pd.DataFrame:
    return pd.DataFrame({categorical_feature: categorical_values_with_nan})

def test_process_categorical_features(
        fake_categorical_data: pd.DataFrame,
        categorical_feature: str,
        categorical_values: List[str],
):
    logger.info(fake_categorical_data)
    transformed: pd.DataFrame = process_categorical_features(fake_categorical_data)
    logger.info(transformed)
    # assert transformed.shape[1] == 3 # ???? how to process nans in cat features - need changes to process_categorical_features/ process_numerical_features
    assert transformed.sum().sum() == 4
