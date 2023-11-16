import pytest
from pathlib import Path

from .conftest import TEST_CONFIG

# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_path=TEST_CONFIG)

# other imports
import os
from pytest import approx

# logging
import logging
logger = logging.getLogger(__name__)

# fixtures
from .fixtures import load_test_catalog




@pytest.mark.django_db(transaction=True)
def test_astrometric_calibration(load_test_catalog):

    from iop4lib.db import Epoch, RawFit, ReducedFit, AstroSource
    from iop4lib.enums import IMGTYPES, SRCTYPES
    from iop4lib.utils.quadmatching import distance

    epochname_L = ["OSN-T090/2023-10-25", "OSN-T090/2023-09-26", "OSN-T090/2023-10-11", "OSN-T090/2023-10-12", "OSN-T090/2023-11-06"]
    epoch_L = [Epoch.create(epochname=epochname) for epochname in epochname_L]

    for epoch in epoch_L:
        epoch.build_master_biases()

    for epoch in epoch_L:
        epoch.build_master_darks()

    for epoch in epoch_L:
        epoch.build_master_flats()


    # Test 1. Photometry field

    fileloc = "OSN-T090/2023-11-06/BLLac_IAR-0001R.fit"
    rawfit = RawFit.by_fileloc(fileloc=fileloc)
    redf = ReducedFit.create(rawfit=rawfit)
    redf.build_file()

    # Test 2. Polarimetry field with quad matching (uses previous photometry field)

    fileloc = "OSN-T090/2023-11-06/BLLAC_R_IAR-0760.fts"
    rawfit = RawFit.by_fileloc(fileloc=fileloc)
    redf = ReducedFit.create(rawfit=rawfit)
    redf.build_file()

    # check source position in the image

    src = AstroSource.objects.get(name="2200+420")

    assert redf.header_hintobject.name == src.name
    assert redf.sources_in_field.filter(name=src.name).exists()
    
    pos_O = src.coord.to_pixel(wcs=redf.wcs1)
    pos_E = src.coord.to_pixel(wcs=redf.wcs2)

    assert (distance(pos_O, [634, 297]) < 25) # O position
    assert (distance(pos_E, [437, 319]) < 50) # E position # might be worse b.c. of companion star

    # Test 3. Polarimetry field using catalog matching
    # This one is quite weak, so it might fail

    fileloc = "OSN-T090/2023-10-11/OJ248_R_IAR-0111.fts"
    rawfit = RawFit.by_fileloc(fileloc=fileloc)
    redf = ReducedFit.create(rawfit=rawfit)
    redf.build_file()

    # check source position in the image

    src = AstroSource.objects.get(name="0827+243")
    
    assert redf.header_hintobject.name == src.name
    assert redf.sources_in_field.filter(name=src.name).exists()
    
    pos_O = src.coord.to_pixel(wcs=redf.wcs1)
    pos_E = src.coord.to_pixel(wcs=redf.wcs2)

    assert (distance(pos_O, [618, 259]) < 50) # O position
    assert (distance(pos_E, [402, 268]) < 50) # E position

    # Test 4. Polarimetry field using target E, O
    
    fileloc = "OSN-T090/2023-10-25/HD204827_R_IAR-0384.fts"
    rawfit = RawFit.by_fileloc(fileloc=fileloc)
    redf = ReducedFit.create(rawfit=rawfit)
    redf.build_file()

    # check source position in the image

    src = AstroSource.objects.get(name="HD 204827")

    assert redf.header_hintobject.name == src.name
    assert redf.sources_in_field.filter(name=src.name).exists()
    
    pos_O = src.coord.to_pixel(wcs=redf.wcs1)
    pos_E = src.coord.to_pixel(wcs=redf.wcs2)

    assert (distance(pos_O, [684, 397]) < 50) # O position
    assert (distance(pos_E, [475, 411]) < 50) # E position