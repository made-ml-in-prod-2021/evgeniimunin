import sys
import numpy as np
import pandas as pd
import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from entities.feature_params import FeatureParams

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def build_numerical_pipeline() -> Pipeline:
    numerical_pipeline = Pipeline([("scaler", StandardScaler())])
    return numerical_pipeline


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline([("ohe", OneHotEncoder())])
    return categorical_pipeline


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                params.categorical_features
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                params.numerical_features
            ),
        ]
    )
    return transformer


def process_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    categorical_pipeline = build_categorical_pipeline()
    df.fillna('nan', inplace=True)
    return pd.DataFrame(categorical_pipeline.fit_transform(df).toarray())


def process_numerical_features(df: pd.DataFrame) -> pd.DataFrame:
    numercial_pipeline = build_numerical_pipeline()
    df.fillna(0, inplace=True)
    return pd.DataFrame(numercial_pipeline.fit_transform(df))


def process_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(transformer.transform(df))


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    return df[params.target_col]


if __name__ == "__main__":
    df = pd.read_csv("../../data/raw/heart.csv")
    logger.info(f"init df: {df.info()}")
    logger.info(f"init df.shape: {df.shape}")

    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
    target = "target"

    params = FeatureParams(
        categorical_features=cat_cols,
        numerical_features=num_cols,
        target_col='target'
    )
    transformer = build_transformer(params)
    transformer.fit(df)

    transdf = process_features(transformer, df)
    catdf = process_categorical_features(df[cat_cols])
    numdf = process_numerical_features(
        df[num_cols]
    )

    logger.info(f"processed catdf.shape: {catdf.shape}")
    logger.info(f"processed numdf.shape: {numdf.shape}")
    logger.info(f"processed transdf.shape: {transdf.shape}")
    logger.info(transdf.columns)
