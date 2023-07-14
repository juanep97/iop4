
**IOP4** is a complete rewrite of IOP3, a pipeline to work with **photometry** and **polarimetry** of **optical data** from [CAHA](https://www.caha.es/es/) and [OSN](https://www.osn.iaa.csic.es/) observatories. It is built to ease debugging and inspection of data.

IOP4 implements _Object Relational Mapping_ (**ORM**) to seamlessly integrate all information about the reduction and results in a database which can be used to query and plot results, flag data and inspect the reduction process in a integrated fashion with the whole pipeline. It also ships with an already **built-in web interface** which can be used out of the box to browse the database and supervise all pipeline processes.


## Installation
Clone this repository and run from a terminal
```bash
    $ conda env create  -f environment.yml
    $ conda activate iop4
    $ pip install .
```
or `pip install -e .` if you want to install it in developer mode. This will install the module `iop4lib` and the `iop4` command, and the `iop4site` project.

After installation, take a look at the example configuration file (`config/config.example.yaml`), set the appropriate variables (path to the database, data directory, astrometry index files path, credentials, etc) and rename it to `config/config.yaml`.

If no previous database exists, make sure to create it. You can do it automatically from the `iop4site/` directory by using
```bash
    $ python manage.py makemigrations
    $ python manage.py migrate
```
Then, populate it with the necessary default data (a catalog, etc):
```bash
    $ python manage.py loaddata priv.dumps.yaml
```
An utility script, `iop4site/resetdb.py`, is provided which will completely reset the DB keeping catalog and users data. To manually back up all data from the DB, you can use
```bash
    $ python manage.py dumpdata --natural-primary --natural-foreign --format=yaml > priv.dumps.yaml
```
If no previous users have been loaded, create one with
```bash
    $ python manage.py createsuperuser
```
You can later use these credentials to login to the admin site.

### Running Tests
To run the tests, first follow the previous steps to configure IOP4. The tests use a different database and data directory (under `~/.iop4tests/`), but need the real database already created (the migration steps above). They will also need the location of the astrometry index files. At the moment, you will also need to download the `iop4testdata.tar.gz` file manually and place it under your home directory. Then, run
```bash
    $ pytest
```

## Usage
### As A Program
The pipeline script `iop4` can be invoked as
```bash
    $ iop4 -l tel1/yymmdd tel2/yymmdd
```
to download and reduce the epoch `yymmdd` from telescopes `tel1` and `tel2` respectively. For example: `iop4 -l T090/230430`.

To serve the results in Django debug server, change to the iop4site directory and run
```bash
    $ python manage.py runserver
```
although this server is only recommended for debugging purposes, and you should use another server in production ([see Django documentation](https://docs.djangoproject.com/en/dev/ref/django-admin/#runserver)).

### As A Library
**iop4lib** uses django ORM and it needs to be configured before using it. Therefore, you should do
```python
    import iop4lib
    iop4lib.Config(config_db=True)
```
once at the start of your script. IOP4 configuration can be accessed anywhere without configuring the ORM doing `import iop4lib; iop4conf = iop4lib.Config(config_db=False)`.

> This way of configuring `IOP4` should be also valid inside IPython Shell, but not for Jupyter notebooks, since their asynchronous output interferes with Django ORM. To use IOP4 inside a notebook, see below. More details can be found in the documentation for `iop4lib.Config`.

Now you are ready to import and use IOP4 models from your Python script, e.g:
```python
    import iop4lib
    iop4lib.Config(config_db=True)
    from iop4lib.db import RawFit, ReducedFit, Epoch, PhotoPolResult

    # this will print the number of T220 nights reduced:
    print(Epoch.objects.filter(telescope="CAHA-T220").count()) 

    # this will reduce the last T220 night:
    Epoch.objects.filter(telescope="CAHA-T220").last().reduce()
```

### In Interactive Notebooks (JupyterLab)
You can use `IOP4` in an interactive manner inside a Jupyter notebook. The following lines also activate matplotlib's graphical output (deactivated by default, as some plots may be generated inside the server).
```python
%autoawait off
%load_ext autoreload
%autoreload all

import iop4lib.config
iop4conf = iop4lib.Config(config_db=True, gonogui=False, jupytermode=True)   
```

### Tips
You can get an IPython interactive terminal after running iop4 using the `-i` option as
```bash
    $ iop4 -i -l T090/230313 T090/230317 T090/230319
```

## Documentation
To build and show the documentation, run
````bash
    $ make docs-sphinx
    $ make docs-show
````

## Contribute

You are welcome to contribute to IOP4. Fork and create a PR!
