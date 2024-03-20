Serving IOP4 in production
==========================

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