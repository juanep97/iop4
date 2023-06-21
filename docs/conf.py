# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'IOP4'
copyright = '2023, Juan Escudero Pedrosa'
author = 'Juan Escudero Pedrosa'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx_rtd_theme',
    'sphinx_mdinclude',
    'numpydoc',
]
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# -- options for rth theme

# html_theme = "sphinx_rtd_theme"
# html_static_path = ['_static']
# html_css_files = [
#     'custom.css',
# ]

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
   ]
}

# -- doc options -------------------------------------------------
autodoc_member_order = 'groupwise'

autodoc_default_options = {
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
iop4lib.Config(config_path=pathlib.Path(iop4lib.config.Config.basedir) / "config" / "config.example.yaml", config_db=True)
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
