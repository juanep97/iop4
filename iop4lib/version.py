# Taken from
# https://github.com/cta-observatory/project-template-python-pure/blob/main/src/template/version.py

try:
    try:
        from ._dev_version import version
    except Exception:
        from ._version import version
except Exception:
    import warnings

    warnings.warn(
        "Could not determine version; this indicates a broken installation."
        " Install from PyPI, using conda or from a local git repository."
        " Installing github's autogenerated source release tarballs "
        " does not include version information and should be avoided."
    )
    del warnings
    version = "0.0.0"

__version__ = version