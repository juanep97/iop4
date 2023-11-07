#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This script will create a iop4testdata.tar.gz file in your home directory.

Use it to build a test dataset from you reduced and tested results. The .tar.gz 
will contain all the raw data needed to reduce the given photo-polarimetry results,
including raw light, bias, darks, flats and a test catalog file.

Edit the pk_L list to add the pk of the results you want to include in the test dataset.
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

pk_L = [
            #48572, # OSN-T090 Andor Polarimetry 2022-09-08 23:03:14 2200+420 reduced fits 6565, 6566, 6567, 6568  mag R 13.328 \pm 0.164 p 12.48 % \pm 0.35 % chi 15.38 \pm 0.79, iop3 mag 13.38, p 14.0, chi 14.7
            #34354, # OSN-T090 Andor Photometry 2022-09-18 23:05:05 2200+420 reduced fits 39071 mag R 13.292 \pm 0.034, iop3 mag 13.35
            #64092, # CAHA-T220 CAFOS Polarimetry 2022-09-18 23:02:53 2200+420 reduced fits 40827, 40828, 40829, 40830 mag R 13.369 \pm 0.036 p 11.18 % \pm 0.12 % chi 25.45 \pm 0.30, iop3 mag 13.38, p 10.9, chi 25.2
            # OSN-T090 DIPOL Polarimetry 2023-10-11 1641+
        ]

output_file = Path("~/iop4testdata.tar.gz").expanduser()
workdir = Path("~/iop4testdata").expanduser()

# remove the workdir if exists, and create it again

if os.path.exists(workdir):
    shutil.rmtree(workdir)

os.makedirs(workdir)

# get the list of files needed and copy them to the workdir

for pk in pk_L:
    res = PhotoPolResult.objects.get(id=pk)

    rawfitL = list()

    for redf in res.reducedfits.all():
        # append all rawfits needed for this reducedfit
        rawfitL.append(redf.rawfit)

        # the bias for its masterbias
        for bias in redf.masterbias.rawfits.all():
            rawfitL.append(bias)

        # the darks for its masterdark
        for dark in redf.masterdark.rawfits.all():
            rawfitL.append(dark)

        # the flats for its masterflat
        for flat in redf.masterflat.rawfits.all():
            rawfitL.append(flat)

        # the bias for the masterbias of its masterflat
        for bias in redf.masterflat.masterbias.rawfits.all():
            rawfitL.append(bias)

        # the bias for the masterbias of its masterdark
        for bias in redf.masterdark.masterbias.rawfits.all():
            rawfitL.append(bias)

        # the bias and the darks for the masterdark of its masterflat
        for dark in redf.masterflat.masterdark.rawfits.all():
            rawfitL.append(dark)


    files_to_download = set([rawfit for rawfit in rawfitL if not os.path.exists(rawfit.filepath)])

    if len(files_to_download) > 0:
        # download the files if needed
        Telescope.get_by_name(rawfitL[0].epoch.telescope).download_files(files_to_download)

    for rawfit in rawfitL:

        relative_path = Path(rawfit.filepath).relative_to(iop4conf.datadir)
        
        dest = workdir / relative_path

        if not os.path.exists(os.path.dirname(dest)):
            os.makedirs(os.path.dirname(dest))

        if not os.path.exists(dest):
            shutil.copy(rawfit.filepath, dest)

# create the catalog file

astrosources_ids_L = list()

for res in PhotoPolResult.objects.filter(id__in=pk_L):
    astrosources_ids_L.append(res.astrosource.id)

    for calibrator in AstroSource.objects.filter(calibrates=res.astrosource):
        astrosources_ids_L.append(calibrator.id)

from django.core.serializers import serialize
with open(workdir / "testcatalog.yaml", "w") as f:
    f.write(serialize("yaml", AstroSource.objects.filter(id__in=set(astrosources_ids_L)), use_natural_foreign_keys=True, use_natural_primary_keys=True))

# Create .tar.gz file

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