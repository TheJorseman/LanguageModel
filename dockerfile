FROM python:3.8-slim-buster
WORKDIR /languagemodel
COPY . /languagemodel
RUN pip3 --no-cache-dir install -r requirements.txt
CMD ["python3", "predict_phrase.py", "'Esto es una prueba'"]

