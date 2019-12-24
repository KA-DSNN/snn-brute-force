FROM tensorflow/tensorflow:latest-gpu-py3

ENV SBF_HOME=""

ADD app/ /app

RUN pip install -r /app/config/req.list

ENTRYPOINT [ "python", "app/src/app.py" ]