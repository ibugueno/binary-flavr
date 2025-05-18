#!/bin/bash

docker rm -fv ignacio_bin-flavr

docker run -it --gpus '"device=0,1,2,3,4,5,6,7"' --name ignacio_bin-flavr -v /home/ignacio.bugueno/cachefs/event_fer/input:/app/input -v /home/ignacio.bugueno/cachefs/event_fer/output:/app/results ignacio_bin-flavr
