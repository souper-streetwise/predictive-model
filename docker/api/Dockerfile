FROM python:3.7

# Update
RUN apt update

# Install python libraries without storing a cache
# Numpy is installed separately to deal with a bug in scikit-garden
COPY requirements.txt /tmp/
WORKDIR /tmp
RUN python -m pip install --no-cache-dir numpy==1.18.2
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy all relevant files into the docker folder
COPY . /api
WORKDIR /

# Spin up the API!
CMD ["gunicorn", "-w", "3", "-b", ":5000", "-t", "360", "--reload", "api.wsgi:app"]
