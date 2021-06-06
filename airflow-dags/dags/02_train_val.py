import os
import json
import pathlib
import pandas as pd
import numpy as np

import airflow
from airflow import DAG

from airflow.operators.bash_operator import BashOperator
from airflow.operators.docker_operator import DockerOperator

with DAG(
    dag_id="generate_train_val",
    start_date=airflow.utils.dates.days_ago(21),
    schedule_interval="@weekly",
) as dag:
    preprocess = DockerOperator(
        image="airflow-preprocess",
        # command="--output-dir /data/raw/{{ ds }}",
        task_id="preprocess",
        do_xcom_push=False,
        volumes=[
            "/home/evgenii/Documents/05_Study/08_MADE_MailGroup_2020/sem2_MLProd/repo/evgeniimunin/airflow-dags/data:/data"]
    )

    split = DockerOperator(
        image="airflow-split",
        # command="--output-dir /data/raw/{{ ds }}",
        task_id="split",
        do_xcom_push=False,
        volumes=[
            "/home/evgenii/Documents/05_Study/08_MADE_MailGroup_2020/sem2_MLProd/repo/evgeniimunin/airflow-dags/data:/data"]
    )

    train = DockerOperator(
        image="airflow-train",
        # command="--output-dir /data/raw/{{ ds }}",
        task_id="train",
        do_xcom_push=False,
        volumes=[
            "/home/evgenii/Documents/05_Study/08_MADE_MailGroup_2020/sem2_MLProd/repo/evgeniimunin/airflow-dags/data:/data"]
    )

    val = DockerOperator(
        image="airflow-val",
        # command="--output-dir /data/raw/{{ ds }}",
        task_id="val",
        do_xcom_push=False,
        volumes=[
            "/home/evgenii/Documents/05_Study/08_MADE_MailGroup_2020/sem2_MLProd/repo/evgeniimunin/airflow-dags/data:/data"]
    )

    notify = BashOperator(
        task_id="notify",
        bash_command=f'echo "Model train and validated ... "',
    )

    preprocess >> split >> train >> val >> notify