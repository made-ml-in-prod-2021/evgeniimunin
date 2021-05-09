![CI workflow](https://github.com/made-ml-in-prod-2021/evgeniimunin/actions/workflows/homework1.yml/badge.svg?branch=homework1)

# Production-ready ML project
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





