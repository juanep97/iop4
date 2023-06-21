
.. _data_reduction_details:

Data reduction details
======================

.. contents:: Table of Contents

Introduction
------------

The following steps are done by IOP4 to reduce the data, for each epoch.

#. Create Master Bias image
#. Create Master Flat image

(No dark flat images / dark current correction currently, should be pretty low anyway)

#. Calibrate science images. This includes:
   #. Subtract master bias
   #. Divide by master flat
   #. Astrometrically calibrate the images (find the WCS).

#. Compute the aperture photometry for each source in the catalog and reduced image.


CAHA T220 Information
---------------------

* Information about the telescope: https://www.caha.es/CAHA/Telescopes/2.2m.html

* Information about the camera: 

  * https://www.caha.es/es/telescope-2-2m-2/cafos
  * https://www.caha.es/CAHA/Instruments/CAFOS/detector.html

OSN T090 Information
--------------------

* Information about the telescope: https://www.osn.iaa.csic.es/page/telescopio-90-cm
* Information about the camera: https://www.osn.iaa.csic.es/page/camaras-ccdt150-y-ccdt90
* Old camera: https://www.osn.iaa.csic.es/page/camara-versarray-t90-retirada

OSN T150 Information
--------------------

* Information about the telescope: https://www.osn.iaa.csic.es/page/telescopio-15-m
* Information about the camera:  https://www.osn.iaa.csic.es/page/camaras-ccdt150-y-ccdt90
* Old camera: https://www.osn.iaa.csic.es/page/camara-ccd-roper