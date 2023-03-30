FROM python:3.10

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install -r /code/requirements.txt

RUN python3 -m nltk.downloader punkt

RUN python3 -m nltk.downloader wordnet

COPY ./app /code/app

WORKDIR /code/app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]


