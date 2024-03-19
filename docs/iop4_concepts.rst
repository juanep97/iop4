IOP4 concepts
=============

**IOP4** is a pipeline to work with
**photometry** and **polarimetry** of **optical data** from
`CAHA <https://www.caha.es/es/>`__ and
`OSN <https://www.osn.iaa.csic.es/>`__ observatories. It is built to
ease debugging and inspection of data.

IOP4 implements *Object Relational Mapping* (**ORM**) to seamlessly
integrate all information about the reduction and results in a database
which can be used to query and plot results, flag data and inspect the
reduction process in a integrated fashion with the whole pipeline.

.. contents:: Table of Contents

Models
-----------

In ORM, classes are mapped to tables, and instances are 
mapped to rows. These classes are called usually called *Models*. IOP4 defines the following
models:

* **Epoch**: an observation session, defined by its `telescope` (str) and `night` (date).
  It provides methods to download the data from the remote telescope archives and classify them.

* **RawFit**: a raw FITS image, defined by its `epoch` (Epoch) and `filename` (str). 

* **MasterBias**, **MasterDark**, **MasterFlat**: master calibration images, defined by their 
  associated `epoch` (Epoch) and `filename` (str). They are built from the raw images of each night of the appropriate type.

* **ReducedFit**: a reduced FITS image, defined by its associated `rawfit` (RawFit). It is built from raw FITS images of LIGHT type, 
  after applying the appropriate master calibration images and performing the astrometric calibration (giving it a correct WCS).

* **AstroSource**: an astronomical source, in the IOP4 catalog, defined by its `name` (str). It contains information such as its 
  type (STAR, BLAZAR, CALIBRATOR), its coordinates (RA, DEC), its literature magnitudes (in the case of calibrators), etc.

* **AperPhotResult**: a result of aperture photometry, defined by its associated 
  `reducedfit` (ReducedFit), the `astrosource` (AstroSource) for which it was computed, 
  and the aperture `aperpix`` (float) used.

* **PhotoPolResult**: a result of photo-polarimetry, the end product of IOP4, as pipeline for
  optical photo-polarimetry.

Any of these objects can be queried from the database for use. For example, to check the last epoch 
reduced:

..  code-block:: python

    import iop4lib
    iop4lib.Config(config_db=True) # important

    from iop4lib.db import Epoch
    last_epoch = Epoch.objects.last()
    print(last_epoch)

or to get all photo-polarimetry results for a given source:

..  code-block:: python

    from iop4lib.db import PhotoPolResult
    results = list(PhotoPolResult.objects.filter(astrosource__name='2200+420'))
    print(results)


Web interface
-------------------

IOP4 uses Django's ORM as its ORM backend. This has the advantage of allowing
us to use Django's debug web server, the django admin interface, its templating 
engine and its database migration system with minimal effort [#otherORMs]_. After 
proper installation, you can start the debug server with:

.. code-block:: bash

    $ python manage.py runserver

This should open a tab in your browser with the IOP4 web interface.

.. warning::
   This server is only recommended for debugging purposes, and you should use another server 
   in production (`see Django documentation <https://docs.djangoproject.com/en/dev/ref/django-admin/#runserver>`_).
   The `iop4site` submodule is written to enable the use of Django's debug server and should be reconfigured when 
   used in production, or entirely replaced by a new Django project and used only as a guide.
   See :doc:`serving iop4 in production <serving_iop4_in_production>` for more information.

After login in with the credentials that you supplied during the `set up` 
</iop4/docs/#usage> you will have access to the following tabs:

* Explore > Plot: to plot and inspect the photometry and polarimetry results, flag data and download plots.
* Explore > Data: to inspect, filter and download data (e.g. in CSV format).
* Admin: to inspect the database.

Telescope and Instrument Specific Code
--------------------------------------

The procedure to analyze and reduce photometric and polarimetric images is similar to one 
observatory to another, but there are many instrument-specific details, for example, non-standard
FITS header keywords, different polarimeters, different pixel scales, etc. IOP4 is designed to abstract these
details from the main code. Telescope-specific code is relegated to the :code:`iop4lib.telescopes` submodule, while 
instrument-specific code is relegated to the :code:`iop4lib.instruments` submodule.
Adding a new telescope or instrument to IOP4 is as simple as adding a new class to these submodules, inheriting the 
:code:`iop4lib.telescopes.Telescope` or code:`iop4lib.instrument.Instrument` base classes, and implementing the required methods 
(like methods to list the available data in the remote observatory archives, reading of non-standard FITS header keywords, or 
specific reduction steps).

Information and details about the different telescopes and instruments can be found at :ref:`data_reduction_details`.


.. rubric:: Footnotes

.. [#otherORMs] There exists many other ORM engines, such as SQLAlchemy, 
                with different advantages. They can be used to access the database 
                if the models are properly translated. Automatic tools exist to this end.