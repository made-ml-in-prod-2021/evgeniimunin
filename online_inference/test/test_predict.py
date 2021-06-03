import pandas as pd
from fastapi.testclient import TestClient
from src.app import app


def test_read_main():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == "It is entry point of predictor"


def test_prediction(dataset_path):
    with TestClient(app) as client:
        data_df = pd.read_csv(dataset_path)
        data_df = data_df.drop("target", axis=1)
        data_df["id"] = data_df.index + 1

        data = data_df.values.tolist()[:15]
        features = data_df.columns.tolist()

        response = client.get("/predict/", json={"data": data, "features": features})
        assert response.status_code == 200
        assert response.json()[0]["diagnosis"] == 0
        assert response.json()[2]["diagnosis"] == 1


def test_prediction_invalid_data():
    with TestClient(app) as client:
        data_df = pd.DataFrame(data={"col1": [None, 2], "col2": [None, 4]})

        data = data_df.values.tolist()
        features = data_df.columns.tolist()

        response = client.get("/predict/", json={"data": data, "features": features})
        assert response.status_code == 400


def test_prediction_invalid_order_features():
    with TestClient(app) as client:
        data_df = pd.DataFrame(data={"col1": [2, 2], "col2": [1, 4]})

        data = data_df.values.tolist()
        features = [
            "id",
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
        ]

        response = client.get("/predict/", json={"data": data, "features": features})
        assert response.status_code == 400
        assert "Invalid features or order. Valid features are" in response.text


def test_prediction_invalid_features():
    with TestClient(app) as client:
        data_df = pd.DataFrame(data={"col1": [2, 2], "col2": [1, 4]})

        data = data_df.values.tolist()
        features = data_df.values.tolist()

        response = client.get("/predict/", json={"data": data, "features": features})
        assert response.status_code == 400
        assert "Invalid features or order. Valid features are" in response.text


def test_prediction_invalid_columns_data(dataset_path):
    with TestClient(app) as client:
        data_df = pd.read_csv(dataset_path)
        data_df = data_df.drop("target", axis=1)
        data_df["id"] = data_df.index + 1

        data = data_df.values.tolist()
        features = data_df.columns.tolist()

        response = client.get("/predict/", json={"data": data, "features": features})
        assert response.status_code == 400
        assert "Invalid features or order. Valid features are" in response.text
