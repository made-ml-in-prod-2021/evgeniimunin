FROM airflow-ml-base:latest

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY val.py /val.py

ENTRYPOINT ["python", "val.py"]