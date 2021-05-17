from typing import List
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from make_dataset import read_data
from entities.feature_params import FeatureParams
from src.features.build_features import process_features, extract_target, build_transformer

@pytest.fixture
def feature_params(
        categorical_features: List[str],
        numerical_features: List[str],
        target_col: str,
) -> FeatureParams:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_col=target_col,
    )
    return params


def test_make_features(
        feature_params: FeatureParams, dataset_path: str,
):
    data = read_data(dataset_path)
    transformer = build_transformer(feature_params)
    transformer.fit(data)
    features = process_features(transformer, data)
    assert not pd.isnull(features).any().any()
    assert len(features) == len(data)


def test_extract_features(
        feature_params: FeatureParams,
        dataset_path: str,
):
    data = read_data(dataset_path)
    target = extract_target(data, feature_params)
    assert_allclose(
        data[feature_params.target_col].to_numpy(), target.to_numpy()
    )

def test_target(
        feature_params: FeatureParams,
        dataset_path: str,
):
    data = read_data(dataset_path)
    assert len(data[feature_params.target_col]) > 0
    assert len(data[feature_params.numerical_features]) == len(data[feature_params.target_col])