FROM python:3.7

# Create docker folder and make it the working directory
#WORKDIR docker/

# Copy all relevant files into the docker folder
COPY docker/data.py /src/
COPY docker/inference.py /src/
COPY docker/training.py /src/
COPY docker/utils.py /src/
COPY templates/ /templates/
COPY data/soup_model /.data/
COPY docker/app.py /
COPY darksky_key.txt /

# Install python libraries without storing a cache
COPY requirements.txt /
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir -r requirements.txt
