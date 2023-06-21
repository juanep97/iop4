#!/bin/sh

rm ~/iop4data/iop4.db
rm -rf ../migrations
rm -rf ../iop4api/migrations
python manage.py flush
#python manage.py migrate iop4api zero
python manage.py makemigrations iop4api
python manage.py migrate
python manage.py loaddata ../config/priv.iop4.*.yaml