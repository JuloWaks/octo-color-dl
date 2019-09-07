export DB_NAME="octo-dl"
export CODE="."
docker build -t octo-dl-torch-docker .
docker stack deploy -c docker-compose.yml octo-dl-stack