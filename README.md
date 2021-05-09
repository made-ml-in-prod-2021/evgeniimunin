![CI workflow](https://github.com/made-ml-in-prod-2021/evgeniimunin/actions/workflows/homework1.yml/badge.svg?branch=homework1)

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

## Train
To train the model you need to choose the config `.yaml` file in the `configs` directory defining the model, train parameters, splitting strategy and the feature set:
```
python train_pipeline.py --config configs/train_config_lr.yaml
```
After the train procedure the saved model and transformer are available in `models`directory.


## Test
To generate the fake data for unit-tests use the following command:
```
python tests/create_fake_dataset.py --size=200
```

To test the current config of the model:
```
pytest --cov
```

## Predict
Provide the path to the prediction dataset and the paths to the trained model and transformer. For example:
```
python predict_pipeline.py --data data/raw/heart.csv --model models/model.pkl --transformer models/transformer.pkl
```
The predictions made will be saved in `data` directory.

## EDA
EDA is available in the Jupyter Notebook `notebooks/eda.ipynb`

## Roadmap
№ | Описание | Баллы
--- | --- | ---
-2 | ~~Назовите ветку homework1~~ | 1
-1 | ~~Положите код в папку ml_project~~ | -
0 | ~~В описании к пулл реквесту описаны основные &quot;архитектурные&quot; и тактические решения, которые сделаны в вашей работе.~~ | 3
1 | ~~Выполнение EDA, закоммитьте ноутбук в папку с ноутбуками~~ | 3
2 | ~~Проект имеет модульную структуру(не все в одном файле =) )~~ | 3
3 | ~~Использованы логгеры~~ | 2
4 | ~~Написаны тесты на отдельные модули и на прогон всего пайплайна~~ | 5
5 | ~~Для тестов генерируются синтетические данные, приближенные к реальным~~ | 5
6 | ~~Обучение модели конфигурируется с помощью конфигов в json или yaml, закоммитьте как минимум 2 корректные конфигурации, с помощью которых можно обучить модель (разные модели, стратегии split, preprocessing)~~ | 2
7 | ~~Используются датаклассы для сущностей из конфига, а не голые dict~~ | 3
8 | ~~Используйте кастомный трансформер(написанный своими руками) и протестируйте его~~ | 3
9 | ~~Обучите модель, запишите в readme как это предлагается~~ | 3
10 | ~~Напишите функцию predict, которая примет на вход артефакт/ы от обучения, тестовую выборку(без меток) и запишет предикт, напишите в readme как это сделать~~ | 3
11 | Используется hydra  (https://hydra.cc/docs/intro/) | 3 (доп баллы)
12 | ~~Настроен CI(прогон тестов, линтера) на основе github actions~~  | 3 балла (доп баллы)
13 | ~~Проведите самооценку, опишите, в какое колво баллов по вашему мнению стоит оценить вашу работу и почему~~ | 1 (доп баллы)

Самооценка: считаю, что выполнил задание на 37/40 баллов, поскольку осталось прописать использование hydra.



