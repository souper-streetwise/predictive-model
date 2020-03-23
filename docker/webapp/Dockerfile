FROM python:3.7

# Install python libraries without storing a cache
COPY requirements.txt /
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy all relevant files into the docker folder
COPY . /webapp

# Spin up the API!
ENTRYPOINT ["gunicorn", "-w", "3", "-b", ":5000", "-t", "360", "--reload", "webapp.wsgi:app"]
