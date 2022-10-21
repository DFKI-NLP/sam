#!/bin/sh

mkdir .brat && cd .brat

sudo mkdir config data
# create config with one single account (name: brat and password: brat)
echo "{\"brat\": \"brat\"}" > config/users.json

# copy collection data to data/ here! (or set file permissions later on)

sudo addgroup bratadmin
sudo chgrp bratadmin config
sudo chgrp bratadmin data
sudo chmod g+s config
sudo chmod g+s data
sudo chmod g+rwx config
sudo chmod g+rwx data

# create and start the container (later on, only use "docker start brat" to restart it)
docker run --name=brat -d -p 80:80 -v data:/bratdata -v config:/bratcfg -e BRAT_USERNAME=brat -e BRAT_PASSWORD=brat -e BRAT_EMAIL=brat@example.com cassj/brat
# stop container via "docker stop brat"