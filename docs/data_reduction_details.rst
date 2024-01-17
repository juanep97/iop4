
.. _data_reduction_details:

Data reduction details
======================

.. contents:: Table of Contents

Introduction
------------

The following steps are done by IOP4 to reduce data from one epoch.

#. Registring the epoch in the database, listing, registring and downloading the raw files.

#. Classifying the files from their FITS header (FLAT, BIAS, LIGHT -science-; image sizes, band, exposure time). Many of these are telescope-dependent and the implementation is relegated to the corresponding telescope class.

#. Create MasterBias images.

   Following standard procedures, the master bias is created by taking the median of all bias images in the epoch (for each image size available).

#. Create MasterFlat images.

   Following standard procedures, the master flat is created by taking the median of all flat images in the epoch, each image normalized to its median value (for each band, etc, available).

#. No dark flat images / dark current correction currently, but it should be pretty low anyway.

#. Calibrate science images, which creates a ReducedFit for each RawFit of type LIGHT. This includes:

   #. Subtract master bias
   #. Divide by master flat
   #. Astrometrically calibrate the images (find the WCS).

#. Compute the aperture photometry for each source in the catalog and reduced image.
#. Compute the results of relative photometry for each source in the catalog and reduced image.
#. Compute the results of relative polarimetry for each source in the catalog and group of reduced images.

CAHA T220 Information
---------------------

* Information about the telescope: https://www.caha.es/CAHA/Telescopes/2.2m.html

* Information about the camera: 

  * https://www.caha.es/es/telescope-2-2m-2/cafos
  * https://www.caha.es/CAHA/Instruments/CAFOS/detector.html

* Reduction of polarization is done according to :cite:t:`zapatero_caballero_bejar:2005`.

OSN T090 Information
--------------------

* Information about the telescope: https://www.osn.iaa.csic.es/page/telescopio-90-cm

RoperT90 instrument (photometry, polarimetry with filters). Retired 23rd October 2021.

* More information about the RoperT90 camera: https://www.osn.iaa.csic.es/page/camara-versarray-t90-retirada

AndorT90 instrument (photometry, polarimetry with filters):

* Information about the AndorT90 camera: https://www.osn.iaa.csic.es/page/camaras-ccdt150-y-ccdt90

DIPOL polarimeter: :cite:t:`dipol:2020`.


OSN T150 Information
--------------------

* Information about the telescope: https://www.osn.iaa.csic.es/page/telescopio-15-m
* Information about the camera:  https://www.osn.iaa.csic.es/page/camaras-ccdt150-y-ccdt90
* Old camera: https://www.osn.iaa.csic.es/page/camara-ccd-roper


References
----------
.. bibliography::
   :style: unsrt
   :all:
