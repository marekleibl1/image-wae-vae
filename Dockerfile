
# Build using:
# docker build -t autoencoders-image .
# 
# Run docker interactively 
# docker run -it --gpus all -u $(id -u):$(id -g) -v .:/repo -w /repo autoencoders-image bash
# 
# Run training script:
# python autoencoders/scripts/step1_train_conv_ae.py 

FROM tensorflow/tensorflow:2.13.0-gpu-jupyter
RUN apt update -y 

COPY ./requirements.txt ./requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN mkdir /repo 
ENV PYTHONPATH "${PYTHONPATH}:/repo"

