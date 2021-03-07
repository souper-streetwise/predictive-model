start-services:
	# Build the images and start the containers
	docker-compose up -d

	# Set up a cronjob, which updates the database with new predictions every day
	# at 3am UTC. This will not overwrite other cronjobs, and if this cronjob
	# already exists then it will not do anything
	crontab -l | grep soupnetwork || (crontab -l; echo "0 3 * * * docker run -v models:/models --net predictive-model_soupnetwork ml") | crontab -

stop-services:
	# Stop and remove the containers
	docker-compose down

	# Remove the images
	docker image remove predictive-model_ml
	docker image remove predictive-model_webapp
	docker image remove predictive-model_dbapi
	docker image remove predictive-model_nginx

	# Remove the cronjob again
	crontab -l | sed '/soupnetwork/d' | crontab -
