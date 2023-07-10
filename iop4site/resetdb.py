#!/usr/bin/env python
"""Script to reset the database, keeping the users and catalog data. USE WITH CARE."""

# iop4lib config
import iop4lib
iop4conf = iop4lib.Config(config_db=False)

# other imports
import os
from datetime import datetime
from termcolor import colored, cprint

manage_fpath = f"{iop4conf.basedir}/iop4site/manage.py"
backupdir_path = f"{iop4conf.basedir}/priv.backups/"
datetime_str = datetime.now().strftime("%Y-%m-%d_%H%M")

print(colored("DANGER! This script will reset the database, keeping the users and catalog data. USE WITH CARE.", "yellow"))

if not os.path.exists(backupdir_path):
    os.makedirs(backupdir_path)

c = input("Are you sure you want to continue? (y/n) ")
if c != "y":
    print("Aborting.")
    exit(1)

print(f"Backing up catalog and users to {backupdir_path}/priv.iop4.dump.*.{datetime_str}.yaml ...")

os.system(f"python {manage_fpath} dumpdata --natural-primary --natural-foreign --format=yaml iop4api.astrosource > {backupdir_path}/priv.iop4.dump.catalog.{datetime_str}.yaml")
os.system(f"python {manage_fpath} dumpdata --natural-primary --natural-foreign --format=yaml auth > {backupdir_path}/priv.iop4.dump.auth.{datetime_str}.yaml")

c = input("Reset the DB? (y/n) ")
if c != "y":
    print("Aborting.")
    exit(1)

print("Resetting database ...")

os.system(rf"rm {iop4conf.db_path}")
os.system(rf"rm -rf {iop4conf.basedir}/iop4site/migrations")
os.system(rf"rm -rf {iop4conf.basedir}/iop4site/iop4api/migrations")
os.system(rf"python {manage_fpath} flush")
#os.system(r"python manage.py migrate iop4api zero")
os.system(rf"python {manage_fpath} makemigrations iop4api")
os.system(rf"python {manage_fpath} migrate")

print(f"Loading catalog and users from priv.iop4.dump.*.{datetime_str}.yaml ...")

c = input("Do you want to load the data? (y/n) ")
if c != "y":
    print("Aborting.")
    exit(1)

os.system(rf"python {manage_fpath} loaddata  {backupdir_path}/priv.iop4.*.{datetime_str}.yaml")