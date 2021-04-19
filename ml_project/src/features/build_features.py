import sys
import numpy as np
import pandas as pd
import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def build_numerical_pipeline() -> Pipeline:
    numerical_pipeline = Pipeline(
        [
            ('scaler', StandardScaler())
        ]
    )
    return numerical_pipeline

def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ('ohe', OneHotEncoder())
        ]
    )
    return categorical_pipeline

def build_transformer(df: pd.DataFrame, cat_cols: list, target: str) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                'categorical_pipeline',
                build_categorical_pipeline(),
                cat_cols,
            ),
            (
                'numerical_pipeline',
                build_numerical_pipeline(),
                df.columns[(~df.columns.isin(cat_cols)) & (df.columns != target)],
            )
        ]
    )
    return transformer

def process_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    categorical_pipeline = build_categorical_pipeline()
    return pd.DataFrame(categorical_pipeline.fit_transform(df).toarray())

def process_numerical_features(df: pd.DataFrame) -> pd.DataFrame:
    numercial_pipeline = build_numerical_pipeline()
    return pd.DataFrame(numercial_pipeline.fit_transform(df))

def process_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(transformer.fit_transform(df))


if __name__ == '__main__':
    df = pd.read_csv('../../data/raw/heart.csv')
    logger.info(f'init df: {df.info()}')
    logger.info(f'init df.shape: {df.shape}')

    df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar',
                  'rest_ecg', 'max_heart_rate_achieved',
                  'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']

    cat_cols = ['sex', 'chest_pain_type', 'fasting_blood_sugar', 'rest_ecg', 'exercise_induced_angina', 'st_slope',
                'thalassemia']
    target = 'target'

    transformer = build_transformer(df, cat_cols, target)
    transdf = process_features(transformer, df)
    catdf = process_categorical_features(df[cat_cols])
    numdf = process_numerical_features(df[df.columns[(~df.columns.isin(cat_cols)) & (df.columns != target)]])

    logger.info(f'processed catdf.shape: {catdf.shape}')
    logger.info(f'processed numdf.shape: {numdf.shape}')
    logger.info(f'processed transdf.shape: {transdf.shape}')