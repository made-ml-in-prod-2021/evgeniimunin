import os
import sys
import pytest
from airflow.models import DagBag

sys.path.append("dags")


@pytest.fixture()
def dag_bag():
    os.environ["DATA_VOLUME_PATH"] = "/tmp"
    return DagBag(dag_folder="dags/", include_examples=False)


def test_train_val_imports(dag_bag):
    assert dag_bag.dags is not None
    assert "hw3_generate_fake_data" in dag_bag.dags
    assert "hw3_train_val" in dag_bag.dags
    assert "hw3_predict" in dag_bag.dags


def test_generate_data_dag(dag_bag):
    dag = dag_bag.dags["hw3_generate_fake_data"]

    dag_flow = {"get_data": ["notify"], "notify": []}

    for name, task in dag.task_dict.items():
        assert task.downstream_task_ids == set(dag_flow[name])


def test_train_val_dag(dag_bag):
    dag = dag_bag.dags["hw3_train_val"]

    dag_flow = {
        "wait-for-features": ["preprocess"],
        "wait-for-target": ["preprocess"],
        "preprocess": ["split"],
        "split": ["train"],
        "train": ["val"],
        "val": ["notify"],
        "notify": [],
    }

    for name, task in dag.task_dict.items():
        assert task.downstream_task_ids == set(dag_flow[name])


def test_predict_dag(dag_bag):
    dag = dag_bag.dags["hw3_predict"]

    dag_flow = {
        "wait-for-data": ["predict"],
        "wait-for-model": ["predict"],
        "predict": [],
    }

    for name, task in dag.task_dict.items():
        assert task.downstream_task_ids == set(dag_flow[name])
