Serving IOP4 in production
==========================

.. _production_web_server:

Setting up a production web server
----------------------------------

.. warning::

    This section requires some familiarity with system 
    administration and web server configuration.

IOP4 comes with the iop4site project to allow a fast and easy way of serving 
results. However, it should only be used for **local** debugging and not offered
to the public. For production use, you should use a proper web server. Here we 
provide some details on how to set up IOP4 with nginx and gunicorn. This guide 
is not complete and should be taken only as a starting point, since setting up
a real server is a complex task and depends on how you want to integrate
IOP4 with your existing infrastructure.

This setup has been tested on Ubuntu Server 22.04 LTS. You need to install nginx
and gunicorn (within your virtual environment).

Example nginx site configuration (to be placed at ``/etc/nginx/sites-available/``)

.. literalinclude:: production_example/nginx_example_site
  :language: nginx

where you should replace ``domain``, ``domain2`` with the domain names that you
will be using, and ``/path/to/static/`` should be accessible by the user 
running nginx (e.g. under ``/var/www/html/mysite/static/``).

Example gunicorn socket configuration (to be placed at ``/etc/systemd/system/``)

.. literalinclude:: production_example/gunicorn.socket
  :language: ini

Example gunicorn service configuration (to be placed at ``/etc/systemd/system/``)

.. literalinclude:: production_example/gunicorn.service
  :language: ini

where ``/path/to/your/django/site`` is the path to your Django site project 
(your modified `iop4site` or a different one) and 
``/path/to/gunicorn/executable`` is the path to the gunicorn executable (usually
within your virtual environment, you can get it by typing ``which gunicorn`` 
when your virtual environment is activated).

The most important settings to change in your new ``settings.py`` file within 
your Django project are:

.. code-block:: python

    SECRET_KEY = "new generated key"

    # SECURITY WARNING: don't run with debug turned on in production!
    DEBUG = True
    INTERNAL_IPS = []
    ALLOWED_HOSTS = ["domain", "domain2"]

    # Configure static files to the path served by nginx
    STATIC_ROOT = '/path/to/static/'


.. _production_cron_job:

Creating a cron job for reducing new observations
-------------------------------------------------

You might be interested in creating a cron job that routinely reduces new 
observations every morning. Here we provide an example. This assumes you are 
working in a linux system. Other OS provide different methods to create jobs.

Use ``crontab -e`` to edit your crontab file and add a line like the following
.. code-block:: cron
      00 08 * * * /home/vhega/run_iop4_daily.sh > /home/vhega/run_iop4_daily.log 2>&1

Then, create a file ``run_iop4_daily.sh``, give it execution permissions (``chmod +x run_iop4_daily.sh``) and add the following content:
.. code-block:: 
      #!/usr/bin/bash

      # save the current datetime
      printf -v date '%(%Y-%m-%d_%H%M)T' -1

      echo "#########################################"
      echo "Run daily job for IOP4: $date"
      echo "#########################################"

      . /home/vhega/miniconda3/bin/activate iop4

      # make sure all files created by iop4 are editable by the current user only
      umask 0022

      # Run iop4 for new observations (i.e. last night)
      iop4 --discover-missing -o log_file=/home/vhega/iop4data/logs/daily_$date.log

      # Create and send a summary of the results for last night
      iop4-night-summary  --fromaddr '{{YOUR SENDER ADDRESS}}' \
                          --mailto '{{ADDRESS 1}},{{ADDRESS 2}},{{ADDRESS 3}}' \
                          --contact-name '{{CONTACT NAME}}' \
                          --contact-email '{{CONTACT EMAIL}}' \
                          --site-url '{{DEPLOYMENT SITE URL}}' \
                          --saveto "/home/vhega/iop4data/logs/daily_$date.html"

The above script will run iop4 every morning, disovering and proccessing new 
observations. 

The last command is optional, and will send an email with a summary with the 
results from last night to the specified email addresses. 
The cron job does not need to be run on a daily basis, and you can run it 
whenever you expect new observations to become available in the telescope 
remote archives (e.g. every few hours). Alternatively, you can pass an argument 
to ``iop4-night-summary`` specifying the night that you want to generate the 
summary for. The email will be in HTML format (viewable in any modern browser or
email client), and can optionally be saved to any path. You can indicate the url 
of your deployed site so the links in the email (e.g. for files with error) 
point directly to the corresponding page in the iop4 web interface or admin 
site.


.. _production_share_datadir:

Sharing the data directory with other system users
--------------------------------------------------

If you are running IOP4 in a server with multiple users, and have created an 
user to run the IOP4 pipeline as a service, you might be interested in making 
the local archive available to other users, so they can process the data 
independently. By default, IOP4 will remove write permissions on raw files, 
protecting them from accidental modification. For example, you can link your raw
directory to the service account raw directory,

.. code-block:: bash

      ln -s /home/iop4user/.iop4data/raw ~/home/myuser/.iop4data/raw

The other directories can also be linked, but keep in mind that other users
might not be able to modify and reprocess reduced files (they will still be able
to inspect them). You will still need to create your database following the 
installation instructions.

Multiple users can also share the same data directory, but this is not 
recommended.

You should not confuse system users (which can run the iop4 pipeline) with IOP4 
portal users, that can access and inspect data from the web interface. These 
should be created only after following See :ref:`Setting up a production web server <production_web_server>` (the debug web 
server is not recommended for multiple users).