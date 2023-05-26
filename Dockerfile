FROM python:3.10-bullseye

WORKDIR /usr/src/app

# Install dependencies
#RUN apt-get update \
#    && apt-get upgrade -y \
#    && apt-get autoremove -y \
#    && apt-get install -y \
#        gcc \
#        build-essential \
#        zlib1g-dev \
#        cmake \
#        python3-dev \
#        gfortran \
#        libblas-dev \
#        liblapack-dev \
#        libatlas-base-dev \
#        libopenblas-dev liblapack-dev \
#    && apt-get clean


COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD [ "uvicorn", "main:app", "--reload", "--host", "0.0.0.0" ]