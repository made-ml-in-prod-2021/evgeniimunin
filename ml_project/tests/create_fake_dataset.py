import numpy as np
import pandas as pd


def generate_dataset(size=200):
    fake_data = pd.DataFrame()
    fake_data['age'] = np.random.normal(size=size)
    fake_data['sex'] = np.random.choice([0, 1], size=size)
    fake_data['cp'] = np.random.randint(0, 3, size=size)
    fake_data['trestbps'] = np.random.normal(size=size)
    fake_data['chol'] = np.random.normal(size=size)
    fake_data['fbs'] = np.random.choice([0, 1], size=size)
    fake_data['restecg'] = np.random.randint(0, 3, size=size)
    fake_data['thalach'] = np.random.normal(size=size)
    fake_data['exang'] = np.random.choice([0, 1], size=size)
    fake_data['oldpeak'] = np.random.normal(size=size)
    fake_data['slope'] = np.random.randint(0, 3, size=size)
    fake_data['ca'] = np.random.randint(0, 3, size=size)
    fake_data['thal'] = np.random.randint(0, 4, size=size)
    fake_data['target'] = np.random.choice([0, 1], size=size)
    fake_data.to_csv('train_data_sample.csv', index=False)

if __name__ == '__main__':
    generate_dataset()



