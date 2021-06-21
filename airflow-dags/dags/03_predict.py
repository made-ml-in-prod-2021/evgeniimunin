import os

import airflow
from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.providers.docker.operators.docker import DockerOperator


MODEL_PATH = "data/models/{{ ds }}/model.pkl"
SCALER_PATH = "data/models/{{ ds }}/scaler.pkl"
RAW_DATA_PATH = "data/raw/{{ ds }}/data.csv"


with DAG(
    dag_id="hw3_predict",
    start_date=airflow.utils.dates.days_ago(5),
    schedule_interval="@daily",
) as dag:

    wait_for_data = FileSensor(
        task_id="wait-for-data", poke_interval=5, retries=5, filepath=RAW_DATA_PATH
    )

    wait_for_model = FileSensor(
        task_id="wait-for-model", poke_interval=5, retries=5, filepath=MODEL_PATH
    )

    predict = DockerOperator(
        image="airflow-predict",
        command="--input_dir data/raw/{{ ds }} --model_dir data/models/{{ ds }} --scaler_dir data/models/{{ ds }} --output_dir data/predictions/{{ ds }}",
        task_id="predict",
        do_xcom_push=False,
        volumes=[f'{os.environ["DATA_VOLUME_PATH"]}:/data'],
    )

    # parallel sensors
    [wait_for_model, wait_for_data] >> predict
