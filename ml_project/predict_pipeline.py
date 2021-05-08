import logging
import pandas as pd
import sys
import joblib

from src.features.build_features import process_features, build_transformer
from models.model_fit_predict import predict_model
from entities.train_pipeline_params import TrainingPipelineParams
from entities.feature_params import FeatureParams

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

CONFIG_PATH = 'configs/train_config.yaml'

def predict(
    data: pd.DataFrame,
    # training_pipeline_params: TrainingPipelineParams,
    model_path: str = 'models/model.pkl',
    transformer_path: str = 'models/transformer.pkl'
) -> pd.Series:

    transformer = joblib.load(transformer_path)
    features = process_features(transformer, data)

    model = joblib.load(model_path)
    preds = pd.Series(model.predict(features))
    logger.info(f'preds shape: {preds.shape}')

    return preds


