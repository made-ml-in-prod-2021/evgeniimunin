import os
import json
import pathlib
import pandas as pd
import numpy as np

import airflow
import requests
import requests.exceptions as requests_exceptions
from airflow import DAG
# from airflow.operators.bash import BashOperator
# from airflow.operators.python import PythonOperator
# from airflow.providers.docker.operators.docker import DockerOperator
# from airflow_docker.operator import Operator

from airflow.operators.bash_operator import BashOperator
from airflow.operators.docker_operator import DockerOperator


SIZE = 10
OUTPUT_DIR = 'data/raw/'

with DAG(
    dag_id="generate_fake_data",
    start_date=airflow.utils.dates.days_ago(5),
    schedule_interval="@daily",
) as dag:

    get_data = DockerOperator(
        image="airflow-download",
        # command="--output-dir /data/raw/{{ ds }}",
        task_id="get_data",
        do_xcom_push=False,
        volumes=["/home/evgenii/Documents/05_Study/08_MADE_MailGroup_2020/sem2_MLProd/repo/evgeniimunin/airflow-dags/data:/data"]
    )

    notify = BashOperator(
        task_id="notify",
        bash_command=f'echo "{SIZE} new rows of data generated ... "',
    )

    get_data >> notify