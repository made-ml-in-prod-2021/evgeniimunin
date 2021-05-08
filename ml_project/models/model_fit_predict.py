import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from typing import Dict, Union
import joblib

from entities.train_params import TrainingParams

SklearnClassifier = Union[RandomForestClassifier, LogisticRegression]

def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> SklearnClassifier:
    if train_params.model_type == 'RandomForestClassifier':
        model = RandomForestClassifier(
            max_depth=train_params.max_depth, random_state=train_params.random_state
        )
    elif train_params.model_type == 'LogisticRegression':
        model = LogisticRegression(
            solver=train_params.solver,
            C=train_params.C,
            random_state=train_params.random_state,
        )
    model.fit(features, target)
    return model


def predict_model(
       model: RandomForestClassifier, features: pd.DataFrame
) -> np.ndarray:
    return model.predict(features)


def evaluate_model(
        predicts: np.ndarray, target: pd.Series
) -> Dict[str, float]:
    return {
        'f1_score': f1_score(target, predicts, average='weighted'),
        'acc_score': accuracy_score(target, predicts),
    }


def serialize_model(
        model: RandomForestClassifier, output: str
) -> str:
    with open(output, 'wb') as f:
        joblib.dump(model, f)
    return output