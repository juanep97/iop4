#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This script will create a iop4testdata.$MD5.tar.gz file in your home directory.

Use it to build a test dataset from your reduced and tested results. The .tar.gz 
will contain all the raw data needed to reduce the given photo-polarimetry results,
including raw light, bias, darks, flats and a test catalog file.

Edit the pk_L list to add the pk of the results you want to include in the test dataset.

Edit the raw_fileloc_L list to add the paths of additional files you want to include in the test dataset.

"""

import iop4lib
iop4conf = iop4lib.Config(config_db=True)

from iop4lib.db import Epoch, RawFit, ReducedFit, PhotoPolResult, MasterBias, MasterFlat, AperPhotResult, AstroSource
from iop4lib.telescopes import Telescope

import os
import subprocess
import shutil
import hashlib
from pathlib import Path

from django.core.serializers import serialize

######################
### Configuration ####
######################

clean_workdir = False

## Addtional files to include 

# only Nmax_raw_per_master will be used for these files:
Nmax_raw_per_master = 1

raw_files_to_include = [
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

# Since the pks of the results might have changed since last built, specify them
# as a list of source and files.

results_to_include = [

    ("Hiltner960", "CAHA-T220/2025-09-13/caf-20250913-21:43:38-sci-agui.fits"),

    # <PhotoPolResult(id: 883611
    #     reducedfits: [298086, 298087, 298088, 298089]
    #     CAFOS2.2 POLARIMETRY R / Hiltner960
    #     JD: 2460932.40608 (2025-09-13T21:44:45)
    #     mag R: 9.797 ± 0.012
    #     p: (5.116 ± 0.271)%
    #     chi: (53.772 ± 2.127)º)>

    ("Hiltner960", "OSN-T090/2025-10-18/Hiltner960_R_IAR-0033_DIPOL.fts"),

    # <PhotoPolResult(id: 883661
    #     reducedfits: [303672, 303673, 303674, 303675, 303676, 303677, 303678, 303679, 303680, 303681, 303682, 303683, 303684, 303685, 303686, 303687]
    #     DIPOL POLARIMETRY R / Hiltner960
    #     JD: 2460967.41977 (2025-10-18T22:04:28)
    #     mag R: 9.830 ± 0.073
    #     p: (5.331 ± 0.087)%
    #     chi: (52.775 ± 3.040)º)>

    ("2200+420", "OSN-T090/2022-09-08/BLLac-0001R0.fit"),

    # <PhotoPolResult(id: 287755                                                                                                                                                            
    #     reducedfits: [6565, 6566, 6567, 6568]
    #     POLARIMETRY R / 2200+420                                                                 
    #     JD: 2459831.46059 (2022-09-08 23:03:14.825)                                            
    #     mag: 13.346 ± 0.086  
    #     p: 0.125 ± 0.004                                                                       
    #     chi: 15.504 ± 0.909)> 

    ("2200+420", "OSN-T090/2022-09-18/BLLac-0001R.fit"), 

    # <PhotoPolResult(id: 284969
    #     reducedfits: [39071]                                                                   
    #     PHOTOMETRY R / 2200+420
    #     JD: 2459841.46187 (2022-09-18 23:05:05.190)                                            
    #     mag: 13.310 ± 0.032)>

]

# For these, Nmax_raw_per_master will be ignored to keep the result as close to 
# the real one as possible (except for DIPOL, too many files).

################################################################################

results_pk_L = list()

for srcname, fileloc in results_to_include:
    redf = ReducedFit.by_fileloc(fileloc)
    r = PhotoPolResult.objects.filter(
        astrosource__name=srcname,
        reducedfits__in=[redf]
    ).get()
    results_pk_L.append(r.pk)

output_file = Path("~/iop4testdata.tar.gz").expanduser()
workdir = Path("~/iop4testdata").expanduser()

# remove workdir if configured, and create it again if necessary

if clean_workdir and os.path.exists(workdir):
    shutil.rmtree(workdir)

os.makedirs(workdir, exist_ok=True)


############################################
### Files for the specified results_pk_L ###
############################################

def get_files_needed(redf, nmax=None):

    rawfitL = list()

    rawfitL.append(redf.rawfit)

    # the bias for its masterbias
    for bias in redf.masterbias.rawfits.all()[:nmax]:
        rawfitL.append(bias)

    # the darks for its masterdark
    if redf.masterdark is not None:
        for dark in redf.masterdark.rawfits.all()[:nmax]:
            rawfitL.append(dark)

    # the flats for its masterflat
    for flat in redf.masterflat.rawfits.all()[:nmax]:
        rawfitL.append(flat)

    # the bias for the masterbias of its masterflat
    for bias in redf.masterflat.masterbias.rawfits.all()[:nmax]:
        rawfitL.append(bias)

    # the bias for the masterbias of its masterdark
    if redf.masterdark is not None:
        for bias in redf.masterdark.masterbias.rawfits.all()[:nmax]:
            rawfitL.append(bias)

    # the bias and the darks for the masterdark of its masterflat
    if redf.masterflat.masterdark is not None:
        for dark in redf.masterflat.masterdark.rawfits.all()[:nmax]:
            rawfitL.append(dark)

    return rawfitL

print(f"Copying files for the specified ({len(results_pk_L)}) results")

# get the list of files needed and copy them to the workdir

rawfitL = list()

for pk in results_pk_L:
    res = PhotoPolResult.objects.get(id=pk)

    if res.instrument == "DIPOL":
        Nmax = 1 # too many files otherwise
    else:
        Nmax = None

    for redf in res.reducedfits.all():
        rawfitL.extend(get_files_needed(redf, Nmax))

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

astrosources_pk_L = list()

# add any source that appears in included results, and their calibrators

for res in PhotoPolResult.objects.filter(id__in=results_pk_L):
    astrosources_pk_L.append(res.astrosource.id)

# add any source that appears in included raw files

for fileloc in raw_files_to_include:
    rf = RawFit.by_fileloc(fileloc)
    if (src := rf.header_hintobject) is not None:
        astrosources_pk_L.append(src.id)

# add any source that calibrates previously added sources
for calibrator in AstroSource.objects.filter(calibrates__in=astrosources_pk_L):
    astrosources_pk_L.append(calibrator.id)

# print sources to be included
for astrosource in AstroSource.objects.filter(id__in=set(astrosources_pk_L)):
    print(f"  {astrosource}")

# dump the catalog as .yaml
qs_sources = AstroSource.objects.filter(id__in=set(astrosources_pk_L))
with open(workdir / "testcatalog.yaml", "a") as f:
    f.write(serialize("yaml", qs_sources, use_natural_foreign_keys=True, use_natural_primary_keys=True))
    f.write('\n')

#############################
### copy additional files ###
#############################

print(f"Specified {len(raw_files_to_include)} files to include")

rawfitL = list()

for fileloc in raw_files_to_include:
    redf = ReducedFit.by_fileloc(fileloc)
    rawfitL.extend(get_files_needed(redf, nmax=1))

print(f"In total, {len(rawfitL)} raw files need to be copied")

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

result = subprocess.run([
    "tar", 
    "--no-xattrs",
    "--exclude='.*'",
    "-czf", str(output_file),
    "-C", str(workdir.parent), str(workdir.name)
])

if result.returncode != 0:
    print("Error creating the .tar.gz file")
    exit(1)

print(f"Created {output_file}")

# get the md5 checksum
with open(output_file, "rb") as f:
    md5 = hashlib.md5(f.read()).hexdigest()

# add the md5sum to the filename
print(f"md5 is {md5}")

output_path = output_file.rename(output_file.with_name(f"iop4testdata.{md5}.tar.gz"))

print(f"Test dataset created at {output_path}")

exit(0)
