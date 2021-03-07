# Soup for Thought

Predicting the demand for a 
[Bristol Soup Run](https://www.bristolsoupruntrust.org.uk/) at a given date.

This project was initiated at the social hackathon arranged by Bristol City 
Council on 8/11/19-10/11/19.

Developers:

  - Dan Saattrup Nielsen (saattrupdan@gmail.com)
  - Elisa Covato
  - Lewis Trinh
  - Kamran Soomro


## Quickstart

Firstly, make sure that you have [docker](https://docs.docker.com/get-docker/) 
and [docker-compose](https://docs.docker.com/compose/install/) installed.

After that, you simply run the command `make` and everything will get set up.
You will then be able to go to [localhost:5000](http://localhost:5000/) to see
the webapp. This will also update the database with new predictions at 3am UTC
every day, set up as a cronjob.

Run the command `make stop-services` to stop all the services.


## REST API

You can send GET requests to the database API at 
[localhost:8080](http://localhost:8080/) with `table` set to one of the
following:

  - `weather`, for the weather data fetched from the 
  [Darksky API](https://darksky.net/dev)
  - `counts` for the raw counts that we have received from the Bristol Soup 
  Run Trust
  - `predictions`, for the associated predictions from the model, including 
  prediction intervals

For instance, 
[localhost:8080/?table=predictions](http://localhost:8080/?table=predictions)
would show the table with the predictions.
