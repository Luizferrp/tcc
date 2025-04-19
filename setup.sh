#!/bin/bash
DOWNLOADDIR=data

mkdir -p $DOWNLOADDIR

if [ ! -d $DOWNLOADDIR ]; then 
  echo "error creating the $DOWNLOADDIR folder";
  exit 1;
fi

curl -L -o $DOWNLOADDIR/nslkdd.zip  https://www.kaggle.com/api/v1/datasets/download/hassan06/nslkdd
curl -L -o $DOWNLOADDIR/bot-iot.zip  https://www.kaggle.com/api/v1/datasets/download/vigneshvenkateswaran/bot-iot
curl -L -o $DOWNLOADDIR/cicidscollection.zip https://www.kaggle.com/api/v1/datasets/download/dhoogla/cicidscollection