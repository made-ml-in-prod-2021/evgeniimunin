from typing import List

from data.make_dataset import read_data
from entities.feature_params import FeatureParams
from entities.train_params import TrainingParams
from src.features.build_features import process_features, extract_target, build_transformer
from models.model_fit_predict import train_model
from predict_pipeline import predict

def test_fit_predict(
        dataset_path: str,
        categorical_features: List[str],
        numerical_features: List[str],
):
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
    model = train_model(features, target, train_params=TrainingParams())
    preds = model.predict(features)
    assert len(preds) == len(features)
    assert len(set(preds)) == 2


def test_predict(dataset_path: str):
    preds = predict(dataset_path)
    assert len(set(preds)) == 2