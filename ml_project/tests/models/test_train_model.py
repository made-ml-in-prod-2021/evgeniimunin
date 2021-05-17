import os
import joblib
from typing import List, Tuple

import pandas as pd
import pytest
from py._path.local import LocalPath
from sklearn.ensemble import RandomForestClassifier

from make_dataset import read_data
from entities.train_params import TrainingParams
from entities.feature_params import FeatureParams
from src.features.build_features import process_features, extract_target, build_transformer
from models.model_fit_predict import train_model, serialize_model


@pytest.fixture
def features_and_target(
        dataset_path: str,
        categorical_features: List[str],
        numerical_features: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_col='target'
    )
    data = read_data(dataset_path)
    transformer = build_transformer(params)
    transformer.fit(data)
    features = process_features(transformer, data)
    target = extract_target(data, params)
    return features, target


def test_train_model(
        features_and_target: Tuple[pd.DataFrame, pd.Series]
):
    features, target = features_and_target
    model = train_model(features, target, train_params=TrainingParams())
    assert isinstance(model, RandomForestClassifier)
    assert model.predict(features).shape[0] == target.shape[0]


def test_serialization_model(tmpdir: LocalPath):
    expected_output = tmpdir.join('model.pkl')
    model = RandomForestClassifier(n_estimators=10)
    real_output = serialize_model(model, expected_output)
    assert real_output == expected_output
    assert os.path.exists
    with open(real_output, 'rb') as f:
        model = joblib.load(f)
    assert isinstance(model, RandomForestClassifier)