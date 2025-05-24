#!/bin/bash

DB_CONTAINER_NAME="database"
DB_DATA_DIR="mongodb"
DB_USER="admin"
DB_PASSWORD="webir2025"

echo "$(tput bold)Database initialization script running."
echo -e "$(tput dim)Deleting old data (if exists):$(tput sgr 0)\n"
docker rm -f $DB_CONTAINER_NAME
sudo rm -rf "$DB_DATA_DIR"
mkdir "$DB_DATA_DIR"

echo -e "\n\n$(tput bold dim)Starting Docker containers..$(tput sgr 0)"
docker compose up -d mongodb

sleep 10

# Check connectivity
echo -e "\n\n$(tput bold dim)Checking connectivity..$(tput sgr 0)\n"
echo $(docker exec -it $DB_CONTAINER_NAME mongosh --eval "db.runCommand({ ping: 1 })" --quiet | grep 'ok' | awk -v container="$DB_CONTAINER_NAME" -F ': ' '{print container "  ok : " $2}' | tr -d '},')

# Create the table
echo -e "\n\n$(tput bold dim)Setting up database..$(tput sgr 0)"
docker exec -it $DB_CONTAINER_NAME mongosh -u $DB_USER -p $DB_PASSWORD /app/scripts/init-tables.js

echo -e "\n\n$(tput setaf 2)Initialization DONE$(tput sgr 0)"
