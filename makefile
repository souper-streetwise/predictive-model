all:
	make build-docker
	make start-services
	make cronjob

build-docker:
	docker-compose build

start-services:
	docker-compose up -d

# Set up cron job to run ML image at 3am UTC every day
cronjob:
	echo "0 3 * * * docker run -v models:/models --net predictive-model_soupnetwork ml" | crontab -
