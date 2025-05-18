#!/bin/bash

docker rm -fv ignacio_bin-flavr

docker run -it \
  --gpus '"device=0,1,2,3"' \
  --name ignacio_bin-flavr \
  --shm-size=16g \
  -v /home/ignacio.bugueno/cachefs/real-time_interpolation/input:/app/input \
  -v /home/ignacio.bugueno/cachefs/real-time_interpolation/output:/app/output \
  ignacio_bin-flavr