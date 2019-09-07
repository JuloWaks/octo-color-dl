set -a
. ./.env
set +a
docker stack deploy -c docker-compose.yml octo-dl-stack