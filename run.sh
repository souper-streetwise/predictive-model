#!/bin/sh

docker-compose build
docker-compose up -d

# Build ML image
docker build /root/ml --tag ml

# Initial run
docker run -v /root/models:/models --net root_soupnetwork ml

# Set up cron job to run ML image at 3am UTC every day
(crontab -l; echo "0 3 * * * /snap/bin/docker run -v /root/models:/models --net root_soupnetwork ml") | crontab -
