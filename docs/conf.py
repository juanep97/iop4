import sys, os, subprocess
GIT_COMMIT_HASH = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(os.path.realpath(__file__))).decode('ascii').strip()
GIT_BRANCH = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=os.path.dirname(os.path.realpath(__file__))).decode('ascii').strip()
GIT_DESCRIBE = subprocess.check_output(['git', 'describe', '--always'], cwd=os.path.dirname(os.path.realpath(__file__))).decode('ascii').strip()

# thumnail_gallery extension
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "sphinxext"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# Get exact version from git

project = 'IOP4'
copyright = '2023, Juan Escudero Pedrosa'
author = 'Juan Escudero Pedrosa'
release = GIT_DESCRIBE

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_nb',
    "nbsphinx",
    'thumbnail_gallery',
    'sphinxcontrib.bibtex',
    'sphinx.ext.autodoc',
    'sphinx.ext.linkcode',
    'numpydoc',
]

templates_path = ['_templates']
exclude_patterns = ['conf.py', '_build', 'sphinxext', '.*']
source_suffix = ['.rst', '.md']

nb_custom_formats = {
    '.py': ['jupytext.reads', {'fmt': 'py:percent'}],
}

nb_merge_streams = True # otherwise several prints stmts result in multiple chunks
nb_execution_timeout = 60 # seconds

# automatically extract thumbnails from notebooks (with sphinxext/thumbnail_gallery)

def get_thumbnails():
    from glob import glob
    from pathlib import Path

    nb_paths = glob(f"recipes/*.py")
    nb_names = [Path(nb_path).stem for nb_path in nb_paths]
    nbsphinx_thumbnails = {f"recipes/{nb_name}":f"_thumbnails/{nb_name}.png" for nb_name in nb_names}

    return nbsphinx_thumbnails

nbsphinx_thumbnails = get_thumbnails()

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# -- options for pydata theme

html_theme = "pydata_sphinx_theme"
pygments_style = 'sphinx'

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/juanep97/iop4",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        }
   ],
   "show_toc_level": 2,
   "show_nav_level": 2,
   "navigation_depth": 2,
}

html_static_path = ['_static']

html_css_files = [
    'pydata/custom.css',
]

# --- bibliography options
bibtex_bibfiles = ['citations.bib']
bibtex_reference_style = "author_year"

# -- doc options -------------------------------------------------

autodoc_default_options = {
    'member-order': 'groupwise',
    'no-undoc-members': True,
    'exclude-members': 'TimeoutException, DoesNotExist, MultipleObjectsReturned, get_deferred_fields, save_base, serializable_value, validate_unique, full_clean, clean_fields, prepare_database_save, unique_error_message, date_error_message, validate_constraints, refresh_from_db'
}

numpydoc_show_class_members = False 

# -- Configure IOP4 for imports from sphinx --
# IMPORTANT! Config must use the example file, or you will show your credentials in the docs!
# config_db=True as it needs to import the models.
import os, sys, pathlib
sys.path.insert(0, os.path.abspath(os.path.join('..', 'iop4lib')))
import iop4lib.config
iop4conf = iop4lib.Config(config_path=pathlib.Path(iop4lib.config.Config.basedir) / "config" / "config.example.yaml", config_db=True)
import iop4lib.db

# -- Add models' fields and their help_text to the documentation --

import inspect
from django.utils.html import strip_tags

def process_docstring(app, what, name, obj, options, lines):
    # This causes import errors if left outside the function
    from django.db import models

    # Only look at objects that inherit from Django's base model class
    if inspect.isclass(obj) and issubclass(obj, models.Model):
        # Grab the field list from the meta class
        fields = obj._meta.fields

        for field in fields:
            # Decode and strip any html out of the field's help text
            help_text = strip_tags(field.help_text)

            # Decode and capitalize the verbose name, for use if there isn't
            # any help text
            verbose_name = field.verbose_name

            if help_text:
                # Add the model field to the end of the docstring as a param
                # using the help text as the description
                lines.append(u':param %s: %s' % (field.attname, help_text))
            else:
                # Add the model field to the end of the docstring as a param
                # using the verbose name as the description
                lines.append(u':param %s: %s' % (field.attname, verbose_name))

            # Add the field's type to the docstring
            lines.append(u':type %s: %s' % (field.attname, type(field).__name__))

    # Return the extended docstring
    return lines

def setup(app):
    # Register the docstring processor with sphinx
    app.connect('autodoc-process-docstring', process_docstring)

def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object definition.

    This function is used by the sphinx.ext.linkcode extension.
    """
    if domain != 'py':
        return None

    # Get the module and object name
    module = sys.modules.get(info['module'])
    if module is None:
        return None
    obj = module.__dict__.get(info['fullname'])
    if obj is None:
        return None

    # Get the file location of the object's source code
    try:
        file_path = inspect.getsourcefile(obj)
    except Exception:
        return None

    # Convert the file path to a GitHub URL
    repo_url = 'https://github.com/juanep97/iop4'
    if file_path.startswith(os.path.commonprefix([iop4conf.basedir, file_path])):
        print(f"file_path: {file_path}")
        print(f"basedir: {iop4conf.basedir}")
        rel_path = os.path.relpath(file_path, iop4conf.basedir)
        print(f"rel_path: {rel_path}")
        url = f'{repo_url}/blob/{GIT_COMMIT_HASH}/{rel_path}'
        return url
    else:
        return None