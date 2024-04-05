FROM python:3.9

WORKDIR /code

COPY ./docker-env/requirements.txt /code/docker-env/requirements.txt
 
RUN pip install --no-cache-dir --upgrade /code/requirements.txt

COPY ./app /code/app

CMD ["python", "./main.py"]
