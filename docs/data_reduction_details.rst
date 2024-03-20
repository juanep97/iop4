.. _data_reduction_details:

Data reduction details
======================

.. contents:: Table of Contents

Introduction
------------

The following steps are done by IOP4 to reduce data from one epoch (when invoked as `iop4 --epoch-list <epochname>`):

#. Registering the epoch in the database, listing, registering and downloading the raw files.

#. Classifying the files from their FITS header (FLAT, BIAS, DARK, LIGHT -science-; image sizes, band, exposure time). Many of these are telescope or instrument dependent and the implementation is relegated to the corresponding telescope or instrument class.

#. Create MasterBias, MasterDark and MasterFlat images.

   Following standard procedures, the master calibration frames are created by grouping all images of that type in the epoch (for each image size, exposure time, band, etc).

   Currently, MasterDark are created only for DIPOL. The dark current in the other instruments is negligible.

#. Calibrate science images, which creates a ReducedFit for each RawFit of type LIGHT. This includes:

   #. Apply the MasterBias, MasterDark (if any) and MasterFlat to the RawFit.
   #. Astrometrically calibrate the images (giving it a correct WCS).

#. Compute relative photometry and relative polarimetry results.
#. Correct the resulting flux and polarization to account for host galaxy contamination (:cite:t:`Nilsson:2007`).

CAHA T220 Information
---------------------

Information about the telescope: https://www.caha.es/CAHA/Telescopes/2.2m.html

* Information about the camera: 

  * https://www.caha.es/es/telescope-2-2m-2/cafos
  * https://www.caha.es/CAHA/Instruments/CAFOS/detector.html

* Reduction of polarization is done according to :cite:t:`zapatero_caballero_bejar:2005`.

OSN T090 Information
--------------------

Information about the telescope: https://www.osn.iaa.csic.es/page/telescopio-90-cm

* RoperT90 instrument (photometry, polarimetry with filters). Retired 23rd October 2021.

  More information about the RoperT90 camera: https://www.osn.iaa.csic.es/page/camara-versarray-t90-retirada

* AndorT90 instrument (photometry, polarimetry with filters):

  Information about the AndorT90 camera: https://www.osn.iaa.csic.es/page/camaras-ccdt150-y-ccdt90

* DIPOL polarimeter: :cite:t:`dipol:Piirola:2020`, :cite:t:`dipol:Jorge:2024`.


OSN T150 Information
--------------------

Information about the telescope: https://www.osn.iaa.csic.es/page/telescopio-15-m

* Old camera (RoperT150): https://www.osn.iaa.csic.es/page/camara-ccd-roper

* AndorT150 instrument (photometry, polarimetry with filters). 
  
  Information about the camera:  https://www.osn.iaa.csic.es/page/camaras-ccdt150-y-ccdt90


References
----------
.. bibliography::
   :style: unsrt
   :all:
