# start by pulling the python image
FROM python:3.7-slim

WORKDIR /workspace

COPY requirements.txt requirements.txt
RUN apt-get update && \
    apt-get -y install gcc mono-mcs && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get -o Acquire::Check-Valid-Until=false update
RUN apt-get install -y libsndfile1
RUN apt-get install g++ -y
#RUN apt-get -y update && apt-get install -y libevent-dev
RUN pip install -r requirements.txt

COPY ./ /workspace

CMD ["python3", "server.py"]
