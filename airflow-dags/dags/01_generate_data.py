import os

import airflow
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator


SIZE = 10
OUTPUT_DIR = 'data/raw/'

with DAG(
    dag_id="hw3_generate_fake_data",
    start_date=airflow.utils.dates.days_ago(5),
    schedule_interval="@daily",
) as dag:

    get_data = DockerOperator(
        image="airflow-download",
        # command="--output-dir /data/raw/{{ ds }}",
        task_id="get_data",
        do_xcom_push=False,
        volumes=[f'{os.environ["DATA_VOLUME_PATH"]}:/data']
    )

    notify = BashOperator(
        task_id="notify",
        bash_command=f'echo "{SIZE} new rows of data generated ... "',
    )

    get_data >> notify