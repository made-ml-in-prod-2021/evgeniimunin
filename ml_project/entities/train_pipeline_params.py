from dataclasses import dataclass, field
from entities.split_params import SplittingParams
from entities.feature_params import FeatureParams
from entities.train_params import TrainingParams
from marshmallow_dataclass import class_schema
import yaml
import logging
import sys

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@dataclass()
class TrainingPipelineParams:
    output_model_path: str
    output_transformer_path: str
    metric_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: TrainingParams
    input_data_path: str = field(default='../data/raw/heart.csv')


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, 'r') as input_stream:
        config_dict = yaml.safe_load(input_stream)
        schema = TrainingPipelineParamsSchema().load(config_dict)
        logger.info(f'Check schema: {schema}')
        return schema

if __name__ == '__main__':
    path = '../configs/train_config.yaml'
    read_training_pipeline_params(path)