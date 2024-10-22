<div>
<a href="https://github.com/juanep97/iop4/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/juanep97/iop4/actions/workflows/ci.yml/badge.svg"></img></a>
<a href="https://zenodo.org/doi/10.5281/zenodo.10222722"><img src="https://zenodo.org/badge/636786270.svg" alt="DOI"></img></a>
<a href="https://pypi.org/project/iop4/"><img src="https://img.shields.io/pypi/v/iop4" alt="PyPI"></img></a>
<a href="https://dx.doi.org/10.3847/1538-3881/ad5a80"><img src="https://img.shields.io/badge/Astronomical_Journal-Published-green" alt="Publication AJ"></a>
</div>


**IOP4** is a complete rewrite of IOP3, a pipeline to work with **photometry** and **polarimetry** of **optical data** from [CAHA](https://www.caha.es/es/) and [OSN](https://www.osn.iaa.csic.es/) observatories. It is built to ease debugging and inspection of data.

IOP4 implements _Object Relational Mapping_ (**ORM**) to seamlessly integrate all information about the reduction and results in a database which can be used to query and plot results, flag data and inspect the reduction process in an integrated fashion with the whole pipeline. It also ships with an already **built-in web interface** which can be used out of the box to browse the database and supervise all pipeline processes.

See details in [Juan Escudero Pedrosa et al 2024 AJ 168 84](https://dx.doi.org/10.3847/1538-3881/ad5a80).


## Installation
We recommend installing IOP4 in an isolated environment as described below. IOP4 is hosted in [PyPI software repository](https://pypi.org/project/iop4/).

### Option 1: Using a virtual environment

**Note:** IOP4 requires Python 3.11 or later. You can check your Python version with `python --version`. If you have a compatible version, you can skip this step.
  
If you don't have Python 3.11 or later, you can install [pyenv](https://github.com/pyenv/pyenv) and [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv), which will manage python versions for you. You can use the automatic installer [pyenv-installer](https://github.com/pyenv/pyenv-installer):

```bash
    $ curl https://pyenv.run | bash
```

Follow the instructions that this command outputs to add `pyenv` to `PATH` (or copy the commands from https://github.com/pyenv/pyenv for your shell). Restart your terminal, or source the file (e.g. `. ~/.bashrc` or `. ~/.zshrc`) Then, run 
```bash 
    $ pyenv install 3.11
    $ pyenv virtualenv 3.11 iop4-venv
    $ pyenv activate iop4-venv
```
Now you will have a virtual environment with the right Python version, and you can continue with the next step. To deactivate, just run `pyenv deactivate`.

With the environment activated, you can install IOP4 latest version by running:
```bash
    $ pip install iop4
```

Alternatively, you can clone this repository and install IOP4:
```bash
    $ git clone 'git@github.com:juanep97/iop4.git'
    $ cd iop4
    $ pip install .
```
or `pip install -e .` if you want to install it in developer mode.


### Option 2: Using conda/mamba
As the previous option, create and activate the environment as follows:
```bash
    $ conda create -n iop4 python=3.11
    $ conda activate iop4
```

Then run:
```bash
    $ pip install iop4
```

Alternatively, you can also clone this repository and run (inside the root directory of the cloned repository):
```bash
    $ pip install .
```
or `pip install -e .` if you want to install it in developer mode.

If you followed the steps in any of the two options above, you will have installed the module `iop4lib` and the `iop4` command, and the `iop4site` project. 

## Configuration

After installation, take a look at the example configuration file (`iop4lib/config.example.yaml`), set the appropriate variables (path to the database, data directory, astrometry index files path, credentials, etc) and save it to `~/.iop4.config.yaml`.

### Running Tests
To run the tests, first follow the previous steps to install IOP4 in developer mode and configure it. The test dataset will be automatically downloaded to your home directory
```bash
    $ pytest -vxs tests/
```
If it is the first time executing IOP4, the astrometry index files will be downloaded to `astrometry_cache_path` (see `config.example.yaml`). This will take some time and a few tens of GB, depending on the exact version.

**Warning**: in some macOS systems, the process [might hang up](https://github.com/juanep97/iop4/issues/14#issuecomment-1748465276). Execute `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` or add that line to your shell init script.

## Usage

If no previous database exists, make sure to create it. You can do it automatically from the `iop4site/` directory by using
```bash
    $ python manage.py makemigrations
    $ python manage.py migrate
```
Then, create a user with
```bash
    $ python manage.py createsuperuser
```
You can later use these credentials to login to the admin site, where you will need to add any sources of interest to the empty catalog.

To manually back up all data from the DB, you can use
```bash
    $ python manage.py dumpdata --natural-primary --natural-foreign --format=yaml > priv.dumps.yaml
```
This file can be used to reload the data to the DB with:
```bash
    $ python manage.py loaddata priv.dumps.yaml
```
An utility script, `iop4site/resetdb.py`, is provided which will completely reset the DB keeping catalog and user's data.

### As A Program
The pipeline script `iop4` can be invoked as
```bash
    $ iop4 --epoch-list tel1/yymmdd tel2/yymmdd
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
You can get an IPython interactive terminal after running iop4 using the `-i` option. You can override any config option using the `-o` option, or by setting environment variables, e.g.
```bash
    $ IOP4_NTHREADS=20 iop4 -i -o log_file=test.log --epoch-list T090/230313 T090/230317
```
Check `iop4 --help` for more info.

## Documentation
To build and show the documentation, run
```bash
    $ make docs-sphinx
    $ make docs-show
```

The documentation for the latest release is hosted in this repository's [GitHub Pages](https://juanep97.github.io/iop4/).

## Contribute

You are welcome to contribute to IOP4. Fork and create a PR!

## Citing IOP4

If you use IOP4, or any result derived from it, we kindly ask you to cite the following references:

<div>
<a href="https://dx.doi.org/10.3847/1538-3881/ad5a80"><img src="https://img.shields.io/badge/Juan_Escudero_Pedrosa_et_al_2024-AJ_168_84-green" alt="DOI AJ"></img></a>

<a href="https://zenodo.org/doi/10.5281/zenodo.10222722"><img src="https://zenodo.org/badge/636786270.svg" alt="DOI Zenodo"></img></a>
</div>

You can use the following BibTeX entries:

```bibtex
@article{iop4_AJ,
  doi = {10.3847/1538-3881/ad5a80},
  url = {https://dx.doi.org/10.3847/1538-3881/ad5a80},
  year = {2024},
  month = {jul},
  publisher = {The American Astronomical Society},
  volume = {168},
  number = {2},
  pages = {84},
  author = {{Escudero Pedrosa}, Juan and {Agudo}, Ivan and {Morcuende}, Daniel and {Otero-Santos}, Jorge and {Bonnoli}, Giacomo and {Piirola}, Vilppu and {Husillos}, C{\'e}sar and {Bernardos}, Mabel and {L{\'o}pez-Coto}, Rub{\'e}n and {Sota}, Alfredo and {Casanova}, V{\'\i}ctor and {Aceituno}, Francisco and {Santos-Sanz}, Pablo},
  title = {IOP4, the Interactive Optical Photo-Polarimetric Python Pipeline},
  journal = {The Astronomical Journal},
}

@software{iop4_zenodo,
  author       = {{Escudero Pedrosa}, Juan and
                  {Morcuende Parrilla}, Daniel and
                  Otero-Santos, Jorge},
  title        = {IOP4},
  year         = 2024,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.10222722},
  url          = {https://zenodo.org/doi/10.5281/zenodo.10222722}
}
```
