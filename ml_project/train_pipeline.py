import json
import logging
import sys

from data.make_dataset import read_data, split_train_val_data

from src.features.build_features import (
    process_features,
    build_transformer,
    extract_target,
)
from models.model_fit_predict import (
    train_model,
    predict_model,
    evaluate_model,
    serialize_model,
)
from entities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_pipeline(training_pipeline_params: TrainingPipelineParams):
    data = read_data(training_pipeline_params.input_data_path)
    logger.info(f"Start train pipeline with params {training_pipeline_params}")
    logger.info(f"Data.shape is  {data.shape}")

    train_df, val_df = split_train_val_data(
        data, training_pipeline_params.splitting_params
    )
    logger.info(f"train_df.shape is  {train_df.shape}")
    logger.info(f"val_df.shape is  {val_df.shape}")

    # prepare train features
    transformer = build_transformer(training_pipeline_params.feature_params)
    transformer.fit(train_df)
    train_features = process_features(transformer, train_df)
    train_target = extract_target(
        train_df, training_pipeline_params.feature_params
    )
    logger.info(f"train_features.shape is  {train_features.shape}")

    # prepare val features
    val_features = process_features(transformer, val_df)
    val_target = extract_target(
        val_df, training_pipeline_params.feature_params
    )
    logger.info(f"val_features.shape is  {val_features.shape}")

    model = train_model(
        train_features, train_target, training_pipeline_params.train_params
    )
    preds = predict_model(model, val_features)
    metrics = evaluate_model(preds, val_target)

    # dump metrics to json
    with open(training_pipeline_params.metric_path, 'w') as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f'Metric is {metrics}')

    # serialize model
    path_to_model = serialize_model(model, training_pipeline_params.output_model_path)
    path_to_transformer = serialize_model(transformer, training_pipeline_params.output_transformer_path)

    return path_to_model, path_to_transformer, metrics


# @click.command(name='train_pipeline')
# @click.argument('config_path')
def train_pipeline_command(config_path: str):
    params = read_training_pipeline_params(config_path)
    train_pipeline(params)

if __name__ == '__main__':
    config_path = 'configs/train_config_lr.yaml'
    train_pipeline_command(config_path)
