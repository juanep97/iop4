#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This script will create a iop4testdata.$MD5.tar.gz file in your home directory.

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

## Addtional files to include 

Nmax_raw_per_master = 1

# only Nmax_raw_per_master will be used for these files

raw_fileloc_L = [
    # some dipol files to test calibration
    "OSN-T090/2023-11-06/BLLac_IAR-0001R.fit", # DIPIL photometry astrocalibration / shotgun
    "OSN-T090/2023-10-25/HD204827_R_IAR-0384.fts", # DIPOL polarimetry astrocalibration / target E, O in a star
    # DIPOL polarimetry astrocalibration / quad matching in a blazar
    "OSN-T090/2023-11-06/BLLac_IAR-0001R.fit", # the photometry file
    "OSN-T090/2023-11-06/BLLAC_R_IAR-0760.fts", # the polarimetry file
    "OSN-T090/2023-10-11/OJ248_R_IAR-0111.fts", # the polarimetry file
    "OSN-T090/2023-11-13/OJ248_R_full_IAR-0001R.fit", # the photometry file
]

# Results to include 

# For these, Nmax_raw_per_master will be ignored (to keep the result as close to the real one as possible)

# Since the pks of the results might have changed since last built, build the list of pks from the filelocs

results_filelocs = [
    "OSN-T090/2022-09-08/BLLac-0001R0.fit",

    # <PhotoPolResult(id: 287755                                                                                                                                                            
    #     reducedfits: [6565, 6566, 6567, 6568]
    #     POLARIMETRY R / 2200+420                                                                 
    #     JD: 2459831.46059 (2022-09-08 23:03:14.825)                                            
    #     mag: 13.346 ± 0.086  
    #     p: 0.125 ± 0.004                                                                       
    #     chi: 15.504 ± 0.909)> 

    "OSN-T090/2022-09-18/BLLac-0001R.fit", 

    # <PhotoPolResult(id: 284969
    #     reducedfits: [39071]                                                                   
    #     PHOTOMETRY R / 2200+420
    #     JD: 2459841.46187 (2022-09-18 23:05:05.190)                                            
    #     mag: 13.310 ± 0.032)>

    "CAHA-T220/2022-09-18/caf-20220918-23:01:33-sci-agui.fits",

    # <PhotoPolResult(id: 285934
    #     reducedfits: [40827, 40828, 40829, 40830]                                                                                                                                         
    #     POLARIMETRY R / 2200+420 
    #     JD: 2459841.46034 (2022-09-18 23:02:53.250)
    #     mag: 13.338 ± 0.026
    #     p: 0.110 ± 0.002     
    #     chi: 25.180 ± 0.409)>   

    "OSN-T090/2023-10-09/3C345_R_IAR-0001.fts",

    # <PhotoPolResult(id: 239179
    #     reducedfits: [105740, 105741, 105742, 105743, 105744, 105745, 105746, 105747, 105748, 105749, 105750, 105751, 105752, 105753, 105754, 105755]
    #     DIPOL POLARIMETRY R / 1641+399
    #     JD: 2460227.33940 (2023-10-09 20:08:43.813)
    #     p: 0.301 ± 0.020
    #     chi: 45.556 ± 1.841)>

    "CAHA-T220/2023-10-09/caf-20231009-19:17:38-sci-agui.fits",

    # <PhotoPolResult(id: 238940
    #     reducedfits: [57754, 57755, 57756, 57757]
    #     POLARIMETRY R / 1641+399
    #     JD: 2460227.30780 (2023-10-09 19:23:13.500)
    #     mag: 16.537 ± 0.013
    #     p: 0.293 ± 0.003
    #     chi: 46.617 ± 0.457)>

    "OSN-T090/2023-10-25/HD204827_R_IAR-0369.fts",

    # <PhotoPolResult(id: 236438
    #     reducedfits: [100574, 100575, 100576, 100577, 100578, 100579, 100580, 100581, 100582, 100583, 100584, 100585, 100586, 100587, 100588, 100589]
    #     DIPOL POLARIMETRY R / HD 204827
    #     JD: 2460243.55174 (2023-10-26 01:14:30.563)
    #     p: 0.051 ± 0.005
    #     chi: 61.374 ± 2.836)>
]

pk_L = list(set([PhotoPolResult.objects.filter(astrosource__in=AstroSource.objects.filter(is_calibrator=False), reducedfits__in=[ReducedFit.by_fileloc(fileloc)]).first().id for fileloc in results_filelocs]))


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
        for bias in redf.masterbias.rawfits.all()[:Nmax]:
            rawfitL.append(bias)

        # the darks for its masterdark
        if redf.masterdark is not None:
            for dark in redf.masterdark.rawfits.all()[:Nmax]:
                rawfitL.append(dark)

        # the flats for its masterflat
        for flat in redf.masterflat.rawfits.all()[:Nmax]:
            rawfitL.append(flat)

        # the bias for the masterbias of its masterflat
        for bias in redf.masterflat.masterbias.rawfits.all()[:Nmax]:
            rawfitL.append(bias)

        # the bias for the masterbias of its masterdark
        if redf.masterdark is not None:
            for bias in redf.masterdark.masterbias.rawfits.all()[:Nmax]:
                rawfitL.append(bias)

        # the bias and the darks for the masterdark of its masterflat
        if redf.masterflat.masterdark is not None:
            for dark in redf.masterflat.masterdark.rawfits.all()[:Nmax]:
                rawfitL.append(dark)


files_to_download = set([rawfit for rawfit in rawfitL if not os.path.exists(rawfit.filepath)])

if len(files_to_download) > 0:
    # download the files if needed
    Telescope.get_by_name(rawfitL[0].epoch.telescope).download_rawfits(files_to_download)

rawfitL = list(set(rawfitL))

print(f"Copying {len(rawfitL)} files for the given PhotoPolResults")

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

for fileloc in raw_fileloc_L:
    rf = RawFit.by_fileloc(fileloc)
    if (src := rf.header_hintobject) is not None:
        astrosources_ids_L.append(src.id)
        for calibrator in AstroSource.objects.filter(calibrates=src):
            astrosources_ids_L.append(calibrator.id)

for astrosource in AstroSource.objects.filter(id__in=set(astrosources_ids_L)):
    print(f"  {astrosource}")

from django.core.serializers import serialize
with open(workdir / "testcatalog.yaml", "a") as f:
    f.write(serialize("yaml", AstroSource.objects.filter(id__in=set(astrosources_ids_L)), use_natural_foreign_keys=True, use_natural_primary_keys=True))
    f.write('\n')



#############################
### copy additional files ###
#############################

print(f"Copying additional {len(raw_fileloc_L)} files")

rawfitL = list()

for fileloc in raw_fileloc_L:

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

print(" Created {}".format(output_file))

# read it and give the md5 sum

with open(output_file, "rb") as f:
    md5 = hashlib.md5(f.read()).hexdigest()
    print("{} {}".format(output_file.name, md5))

# add the md5sum to the filename
os.rename(output_file, output_file.parent / f"iop4testdata.{md5}.tar.gz")

print(f"{output_file.parent / f'iop4testdata.{md5}.tar.gz'}")

exit(0)