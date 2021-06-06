# airflow-examples
код для пары Data Pipelines

чтобы развернуть airflow, предварительно собрав контейнеры
~~~
docker compose up --build
~~~
Ссылка на документацию по docker compose up

https://docs.docker.com/compose/reference/up/

Create fake dataset
```
 python images/airflow-download/create_fake_dataset.py data/raw/
```
