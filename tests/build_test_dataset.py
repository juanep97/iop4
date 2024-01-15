#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This script will create a iop4testdata.tar.gz file in your home directory.

Use it to build a test dataset from your reduced and tested results. The .tar.gz 
will contain all the raw data needed to reduce the given photo-polarimetry results,
including raw light, bias, darks, flats and a test catalog file.

Edit the pk_L list to add the pk of the results you want to include in the test dataset.

Edit the raw_fileloc_L list to add the paths of additional files you want to include in the test dataset.

"""

import iop4lib.config
iop4conf = iop4lib.Config(config_db=True)

from iop4lib.db import Epoch, RawFit, ReducedFit, PhotoPolResult, MasterBias, MasterFlat, AperPhotResult, AstroSource
from iop4lib.telescopes import Telescope

import os
import subprocess
import shutil
import hashlib
from pathlib import Path


######################
### Configuration ####
######################


clean_workdir = False

Nmax_raw_per_master = 1

# only Nmax_raw_per_master will be used for these files

fileloc_L = [
    # some dipol files to test calibration
    "OSN-T090/2023-11-06/BLLac_IAR-0001R.fit", # DIPIL photometry astrocalibration / shotgun
    "OSN-T090/2023-10-11/OJ248_R_IAR-0111.fts", # DIPOL polarimetry astrocalibration / catalog matching in a blazar
    "OSN-T090/2023-10-25/HD204827_R_IAR-0384.fts", # DIPOL polarimetry astrocalibration / target E, O in a star
    # DIPOL polarimetry astrocalibration / quad matching in a blazar
    "OSN-T090/2023-11-06/BLLac_IAR-0001R.fit", # the photometry file
    "OSN-T090/2023-11-06/BLLAC_R_IAR-0760.fts", # the polarimetry file
]

# for this, Nmax_raw_per_master will be ignored (to keep the result as close to the real one as possible)

pk_L = [
            # 48572, # OSN-T090 Andor Polarimetry 2022-09-08 23:03:14 2200+420 reduced fits 6565, 6566, 6567, 6568  mag R 13.328 \pm 0.164 p 12.48 % \pm 0.35 % chi 15.38 \pm 0.79, iop3 mag 13.38, p 14.0, chi 14.7
            # 34354, # OSN-T090 Andor Photometry 2022-09-18 23:05:05 2200+420 reduced fits 39071 mag R 13.292 \pm 0.034, iop3 mag 13.35
            # 64092, # CAHA-T220 CAFOS Polarimetry 2022-09-18 23:02:53 2200+420 reduced fits 40827, 40828, 40829, 40830 mag R 13.369 \pm 0.036 p 11.18 % \pm 0.12 % chi 25.45 \pm 0.30, iop3 mag 13.38, p 10.9, chi 25.2
            # ?????, # OSN-T090 DIPOL Polarimetry 2023-10-11 3C345 [mjd 60228.82199 datetime 2023/10/11 19:43	p 29.238 ± 0.000 chi 38.379 ± 0.392]
        ]

output_file = Path("~/iop4testdata.tar.gz").expanduser()
workdir = Path("~/iop4testdata").expanduser()

# remove workdir if configured, and create it again if necessary

if clean_workdir and os.path.exists(workdir):
    shutil.rmtree(workdir)

os.makedirs(workdir, exist_ok=True)


###############################################
### Files for the PhotoPolResults pk_L list ###
###############################################


print("Copying files for the PhotoPolResults pk_L list")

# get the list of files needed and copy them to the workdir

rawfitL = list()

for pk in pk_L:
    res = PhotoPolResult.objects.get(id=pk)

    if res.instrument == "DIPOL":
        Nmax = 1 # too many files otherwise
    else:
        Nmax = None

    for redf in res.reducedfits.all():
        # append all rawfits needed for this reducedfit
        rawfitL.append(redf.rawfit)

        # the bias for its masterbias
        for bias in redf.masterbias.rawfits.all()[:None]:
            rawfitL.append(bias)

        # the darks for its masterdark
        if redf.masterdark is not None:
            for dark in redf.masterdark.rawfits.all()[:None]:
                rawfitL.append(dark)

        # the flats for its masterflat
        for flat in redf.masterflat.rawfits.all()[:None]:
            rawfitL.append(flat)

        # the bias for the masterbias of its masterflat
        for bias in redf.masterflat.masterbias.rawfits.all()[:None]:
            rawfitL.append(bias)

        # the bias for the masterbias of its masterdark
        if redf.masterdark is not None:
            for bias in redf.masterdark.masterbias.rawfits.all()[:None]:
                rawfitL.append(bias)

        # the bias and the darks for the masterdark of its masterflat
        if redf.masterflat.masterdark is not None:
            for dark in redf.masterflat.masterdark.rawfits.all()[:None]:
                rawfitL.append(dark)


files_to_download = set([rawfit for rawfit in rawfitL if not os.path.exists(rawfit.filepath)])

if len(files_to_download) > 0:
    # download the files if needed
    Telescope.get_by_name(rawfitL[0].epoch.telescope).download_rawfits(files_to_download)

for rawfit in rawfitL:

    print(f"  cp {rawfit.fileloc}")

    relative_path = Path(rawfit.filepath).relative_to(iop4conf.datadir)
    
    dest = workdir / relative_path

    if not os.path.exists(os.path.dirname(dest)):
        os.makedirs(os.path.dirname(dest))

    if not os.path.exists(dest):
        shutil.copy(rawfit.filepath, dest)



###############################
### create the catalog file ###
###############################

print("Creating the catalog file")

astrosources_ids_L = list()

for res in PhotoPolResult.objects.filter(id__in=pk_L):
    astrosources_ids_L.append(res.astrosource.id)

    for calibrator in AstroSource.objects.filter(calibrates=res.astrosource):
        astrosources_ids_L.append(calibrator.id)

from django.core.serializers import serialize
with open(workdir / "testcatalog.yaml", "a") as f:
    f.write(serialize("yaml", AstroSource.objects.filter(id__in=set(astrosources_ids_L)), use_natural_foreign_keys=True, use_natural_primary_keys=True))
    f.write('\n')



#############################
### copy additional files ###
#############################

print("Copying additional files")

rawfitL = list()

for fileloc in fileloc_L:

    redf = ReducedFit.by_fileloc(fileloc)

    # append all rawfits needed for this reducedfit
    rawfitL.append(redf.rawfit)

    # the bias for its masterbias
    for bias in redf.masterbias.rawfits.all()[:Nmax_raw_per_master]:
        rawfitL.append(bias)

    # the darks for its masterdark
    if redf.masterdark is not None:
        for dark in redf.masterdark.rawfits.all()[:Nmax_raw_per_master]:
            rawfitL.append(dark)

    # the flats for its masterflat
    for flat in redf.masterflat.rawfits.all()[:Nmax_raw_per_master]:
        rawfitL.append(flat)

    # the bias for the masterbias of its masterflat
    for bias in redf.masterflat.masterbias.rawfits.all()[:Nmax_raw_per_master]:
        rawfitL.append(bias)

    # the bias for the masterbias of its masterdark
    if redf.masterdark is not None:
        for bias in redf.masterdark.masterbias.rawfits.all()[:Nmax_raw_per_master]:
            rawfitL.append(bias)

    # the bias and the darks for the masterdark of its masterflat
    if redf.masterflat.masterdark is not None:
        for dark in redf.masterflat.masterdark.rawfits.all()[:Nmax_raw_per_master]:
            rawfitL.append(dark)


for rawfit in rawfitL:

    print(f"  cp {rawfit.fileloc}")

    relative_path = Path(rawfit.filepath).relative_to(iop4conf.datadir)
    
    dest = workdir / relative_path

    if not os.path.exists(os.path.dirname(dest)):
        os.makedirs(os.path.dirname(dest))

    if not os.path.exists(dest):
        shutil.copy(rawfit.filepath, dest)
        

###########################
### Create .tar.gz file ###
###########################

print("Creating .tar.gz file")

# in mac-os, additional ._* files are created to save xattrs, avoid it setting COPYFILE_DISABLE
os.environ["COPYFILE_DISABLE"] = "true"

result = subprocess.run(["tar", "--no-xattrs", "--exclude='.*'", "-czf", str(output_file), "-C", str(workdir.parent), str(workdir.name)])

if result.returncode != 0:
    print("Error creating the .tar.gz file")
    exit(1)

print("Created {}".format(output_file))

# read it and give the md5 sum

with open(output_file, "rb") as f:
    md5 = hashlib.md5(f.read()).hexdigest()
    print("{} {}".format(output_file.name, md5))

exit(0)