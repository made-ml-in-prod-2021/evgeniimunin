FROM python:3.7

COPY docker_requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY models/model.pkl models/model.pkl
COPY models/transformer.pkl models/transformer.pkl
COPY src/app.py app.py
COPY src/build_features.py build_features.py
COPY src/entities.py entities.py
COPY src/feature_params.py feature_params.py

WORKDIR .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
