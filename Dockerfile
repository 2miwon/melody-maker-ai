FROM python:3.10

WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app

CMD ["reflex", "run"]