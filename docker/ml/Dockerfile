FROM python:3.7

# Install python libraries without storing a cache
# Numpy is installed separately to deal with a bug in scikit-garden
COPY requirements.txt /
RUN python -m pip install --no-cache-dir numpy==1.18.2
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy all relevant files into the docker folder
COPY . /ml

ENTRYPOINT ["python", "-m", "ml.train_predict"]
