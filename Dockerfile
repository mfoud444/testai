FROM python:3.7

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
# CMD ["uvicorn","-b","0.0.0.0:7860","--timeout","600" ,"app:app"]
