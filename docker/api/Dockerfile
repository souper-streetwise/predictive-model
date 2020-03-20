FROM python:3.7

# Update
RUN apt update

# Install python libraries without storing a cache
COPY requirements.txt /tmp/
WORKDIR /tmp
RUN python -m pip install --no-cache-dir numpy
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy all relevant files into the docker folder
COPY . /api
WORKDIR /

# Spin up the API!
CMD ["gunicorn", "-w", "3", "-b", ":5000", "-t", "360", "--reload", "api.wsgi:app"]
