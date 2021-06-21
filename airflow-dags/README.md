# Airflow ML project

## Run Airflow
Use Docker to run Airflow. Ensure that your Airflow version is at least ```2.0.1``` and 
``` apace-airflow-providers-docker``` is installed.

If you run for the first time, initialize Airflow database:
```
docker-compose up airflow-init
```

To build necessary docker images and run Airflow use the following command:
```
docker-compose up --build
```

To down the DAGs use the following command:
```
docker-compose down
```

If you want to modify the DAGs or the scripts inside containers to before rerun the Airflow ensure 
that docker-compose does not contain previously built images:
```
docker-compose ps
docker-compose rm
```

Airflow UI will be available by the link ```0.0.0.0:8084``` with login: admin, pwd: admin.

## Test Airflow DAGs
To test the DAGs use the command ```pytest``` from the root directory.

## Roadmap
№ | Описание | Баллы
--- | --- | ---
0 | ~~Поднимите airflow локально, используя docker compose (можно использовать из примера https://github.com/made-ml-in-prod-2021/airflow-examples/)~~ | -
1 | ~~Реализуйте dag, который генерирует данные для обучения модели (генерируйте данные, можете использовать как генератор синтетики из первой дз, так и что-то из датасетов sklearn), вам важно проэмулировать ситуации постоянно поступающих данных~~ | 5
2 | ~~Реализуйте dag, который обучает модель еженедельно, используя данные за текущий день. В вашем пайплайне должно быть как минимум 4 стадии, но дайте волю своей фантазии=)~~ | 10
3 | ~~Реализуйте dag, который использует модель ежедневно; принимает на вход данные из пункта 1; считывает путь до модельки из airflow variables(идея в том, что когда нам нравится другая модель и мы хотим ее на прод; делает предсказание и записывает их в /data/predictions/{{ds }}/predictions.csv~~ | 5
3a | ~~Реализуйте сенсоры на то, что данные готовы для дагов тренировки и обучения~~ | 3
4 | ~~Вы можете выбрать 2 пути для выполнения ДЗ. все даги реализованы только с помощью DockerOperator (10 баллов) (пример https://github.com/made-ml-in-prod-2021/airflow-examples/blob/main/dags/11_docker.py)~~ | 10
5 | ~~Протестируйте ваши даги (5 баллов) https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html~~ | 5
6 | В docker compose так же настройте поднятие mlflow и запишите туда параметры обучения, метрики и артефакт(модель) | 5
7 | Вместо пути в airflow variables  используйте апи Mlflow Model Registry | 5
8 | Настройте alert в случае падения дага | 3
9 | ~~Традиционно, самооценка~~ | 1

Самооценка: считаю что выполнил ДЗ на 39/52, поскольку на реализацию MLflow не хватило времени. Далее попробую прописать алерты 
для дагов.

