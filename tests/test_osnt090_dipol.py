import pytest
from pathlib import Path

from .conftest import TEST_CONFIG

# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_path=TEST_CONFIG)

# other imports
import os
from pytest import approx
import numpy as np

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




def test_quad_ish():
    r""" Test that the quad-ish matching works as expected. Procedure:
        1. Create N random 4-point sets
           a) Check if quad coords are invariant under permutations of pts
           b) Check that hash is invariant under permutations of pts
        2. Apply a random translation + rotation + reflection (X and/or Y) to them
           a) Check that the hashes for them are the equal.
        3. Check that the qorder_ish function works, i.e. that the order of the points is the same. We do this by:
           a) randomly ordering transformed points
           b) transforming them back
           c) checking the positions are the same to the original ones, one by one.
    """
    from iop4lib.utils.quadmatching import hash_ish, qorder_ish, quad_coords_ish

    N = 1000 

    # 1.a Create N random 4-point sets

    logger.debug("1. Creating random 4-point sets")

    list_of_points = list()
    for i in range(N):
        list_of_points.append(np.random.rand(4,2))

    # 1.a Check if quad coords are invariant under permutations of pts

    logger.debug("1.a. Checking if quad coords are invariant under permutations of pts")

    for points in list_of_points:
        assert np.allclose(quad_coords_ish(*points)[0], quad_coords_ish(*points[np.random.permutation(4),:])[0])

    # 1.b Check that hash is invariant under permutations of pts

    logger.debug("1.b. Checking that hash is invariant under permutations of pts")

    for points in list_of_points:
        assert hash_ish(points) == approx(hash_ish(points[np.random.permutation(4),:]))

    # 2. Apply a random translation + rotation + reflection to them

    logger.debug("2. Applying random transformations")

    list_of_points_transformed = list()
    M_L = list()
    t_L = list()
    for points in list_of_points:

        # translation
        t = np.random.rand(2)

        # orthogonal transformation (proper or improper rotation)
        from scipy.linalg import qr
        Q, R = qr(np.random.rand(2,2))
        M = Q.dot(Q.T)
        points = M@points.T + t[:,None]

        list_of_points_transformed.append(points.T)
        M_L.append(M)
        t_L.append(t)

    # 2.a Check that the hashes for them are the equal.

    logger.debug("2.a. Checking that the hashes are equal points")

    for p1, p2 in zip(list_of_points, list_of_points_transformed):
        logger.debug(f"p1 = {p1}, p2 = {p2}")
        assert hash_ish(p1) == approx(hash_ish(p2))

    # 3. Check that the qorder_ish function works

    logger.debug("3. Checking that the qorder_ish function works")

    # 3.a) randomly ordering transformed points

    logger.debug("3.a. Randomly ordering transformed points")

    list_of_points_transformed_reordered = list()
    for points in list_of_points_transformed:
        list_of_points_transformed_reordered.append(points[np.random.permutation(4),:])

    # 3.b) transforming them back (substract the translation, apply the inverse of the orthogonal transformation)

    logger.debug("3.b. Transforming them back")

    list_of_points_transformed_reordered_back = list()
    for points, M, t in zip(list_of_points_transformed_reordered, M_L, t_L):
        points = points - np.repeat(t[None,:], 4, axis=0)
        points = np.linalg.inv(M)@points.T
        list_of_points_transformed_reordered_back.append(points.T)

    # 3.c) checking the positions are the same to the original ones, one by one.

    logger.debug("3.c. Checking the positions are the same to the original ones, one by one.")

    for p1, p2 in zip(list_of_points, list_of_points_transformed_reordered_back):

        p1_ordered = np.array(qorder_ish([p for p in p1])).T
        p2_ordered = np.array(qorder_ish([p for p in p2])).T
        
        assert np.all(p1_ordered == approx(p2_ordered))

    

    

    