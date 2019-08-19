FROM continuumio/miniconda3:4.6.14

RUN mkdir /code

WORKDIR /code

COPY ./conda_environment.yml /code/conda_environment.yml

RUN conda env create -f /code/conda_environment.yml && \
    conda clean --all --yes

RUN echo "conda activate fastai" >> ~/.bashrc

RUN pip install flask

RUN pip install pandas

RUN pip install pillow opencv-python

RUN apt-get update

RUN apt-get install libglib2.0-0 -y

RUN apt-get install -y libsm6 libxext6

RUN apt-get install -y libxrender1

RUN pip install fastai

COPY autoboundserver/ /app

COPY run.py /app

WORKDIR /app

CMD ["python3", "run.py"]