FROM python:3.7

COPY docker_requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY models/* models/
COPY src/* src/

WORKDIR .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
