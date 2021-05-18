from typing import List, Union

import numpy as np
from pydantic import BaseModel, conlist, validator

# In prod this features should be in config file
FEATURES_MODELS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
    "id",
]

# FEATURES_MODELS = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'thal', 'id']


class DiagnosisRequest(BaseModel):
    data: List[conlist(Union[float, str], min_items=1, max_items=20)]
    features: List[str]

    @validator("features")
    def validate_model_features(cls, features):
        if not features == FEATURES_MODELS:
            raise ValueError(
                f"Invalid features or order: Valid features are {FEATURES_MODELS}"
            )
        return features

    @validator("data")
    def validate_number_data_columns_and_features(cls, data):
        if np.array(data).shape[1] != len(FEATURES_MODELS):
            raise ValueError(
                f"Invalid columns number for data: Valid number is: {len(FEATURES_MODELS)}"
            )
        return data


class DiagnosisResponse(BaseModel):
    id: str
    diagnosis: int
