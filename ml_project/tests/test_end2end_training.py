import os
from typing import List
from py._path.local import LocalPath

from train_pipeline import train_pipeline
from entities.train_params import TrainingParams
from entities.train_pipeline_params import TrainingPipelineParams
from entities.split_params import SplittingParams
from entities.feature_params import FeatureParams


def test_train_e2e(
    tmpdir: LocalPath,
    dataset_path: str,
    categorical_features: List[str],
    numerical_features: List[str],
    target_col: str,
):
    expected_output_model_path = tmpdir.join("model.pkl")
    expected_output_transformer_path = tmpdir.join("transformer.pkl")
    expected_metric_path = tmpdir.join("metrics.json")
    params = TrainingPipelineParams(
        input_data_path=dataset_path,
        output_model_path=expected_output_model_path,
        output_transformer_path=expected_output_transformer_path,
        metric_path=expected_metric_path,
        splitting_params=SplittingParams(val_size=0.2, random_state=42),
        feature_params=FeatureParams(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            target_col=target_col,
        ),
        train_params=TrainingParams(model_type="RandomForestClassifier"),
    )
    real_model_path, real_transformer_path, metrics = train_pipeline(params)
    assert metrics["f1_score"] > 0
    assert os.path.exists(real_model_path)
    # assert os.path.exists(params.metric_path)
