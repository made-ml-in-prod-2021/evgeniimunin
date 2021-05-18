# Production-ready ML project

## Data
The data is taken from: https://www.kaggle.com/ronitf/heart-disease-uci.

## Install
First clone the directory, create the venv and install the required packages in the root directory of `ml_project`
```
virtualend venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Docker
Build docker:
```
docker build -t evgeniimunin/online_inference:v1 .
```

Run docker. Launch server:
```
docker run -p 8000:8000 evgeniimunin/online_inference:v1
```

Test running application. Client request:
```
python -m src.make_request
```

Push/ Pull docker image from Docker Hub:
```
docker push evgeniimunin/online_inference:v1
docker pull evgeniimunin/online_inference:v1
```

## Roadmap
№ | Описание | Баллы
--- | --- | ---
0 | ~~Ветку назовите homework2, положите код в папку online_inference~~ | -
1 | ~~Оберните inference вашей модели в rest сервис(вы можете использовать как FastAPI, так и flask, другие желательно не использовать, дабы не плодить излишнего разнообразия для проверяющих), должен быть endpoint /predict~~ | 3
2 | ~~Напишите тест для /predict~~ | 3
3 | ~~Напишите скрипт, который будет делать запросы к вашему сервису~~ | 2
4 | ~~Сделайте валидацию входных данных (например, порядок колонок не совпадает с трейном, типы не те и пр, в рамках вашей фантазии)~~ | 3
5 | ~~Напишите dockerfile, соберите на его основе образ и запустите локально контейнер(docker build, docker run), внутри контейнера должен запускать сервис, написанный в предущем пункте, закоммитьте его, напишите в readme корректную команду сборки~~ | 4
6 | Оптимизируйте размер docker image(опишите в readme.md что вы предприняли для сокращения размера и каких результатов удалось добиться) | 3
7 | ~~Опубликуйте образ в https://hub.docker.com/, используя docker push~~ | 2
8 | ~~Напишите в readme корректные команды docker pull/run, которые должны привести к тому, что локально поднимется на inference ваша модель~~ | 3
9 | ~~Проведите самооценку~~ | 1

Самооценка: считаю, что выполнил задание на 21/23 баллов, поскольку необходимо соптимизировать Docker образ, который в даннй момент весит 1,2 ГБ.



