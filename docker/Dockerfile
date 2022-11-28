FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04

LABEL MAINTAINER="VTS <vts.gpmn.ai@viettel.com.vn>"

EXPOSE 8001

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements, cv2 dependencies 
RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx libglib2.0-0 vim
RUN apt-get -y install python3-pip
RUN apt-get install -y --no-install-recommends libgl1 libglib2.0-0 dos2unix

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /code
COPY . /code

# During debugging, this entry point will be overridden.
RUN dos2unix ./scripts/entrypoint.sh && apt-get --purge remove -y dos2unix && rm -rf /var/lib/apt/lists/*
RUN chmod u+x ./scripts/entrypoint.sh

ENTRYPOINT ["/bin/sh", "./scripts/entrypoint.sh"]