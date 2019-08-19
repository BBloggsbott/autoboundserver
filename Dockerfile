FROM continuumio/miniconda3

RUN pip install flask

RUN pip install pandas

RUN pip install pillow opencv-python

RUN apt-get update

RUN apt-get install libglib2.0-0 -y

RUN apt-get install -y libsm6 libxext6

RUN apt-get install -y libxrender1 gcc

RUN pip install fastai

WORKDIR /app

RUN mkdir autoboundserver

COPY autoboundserver/ /app/autoboundserver

COPY run.py /app

WORKDIR /app

RUN ls

CMD ["python3", "run.py"]
