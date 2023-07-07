#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="iop4",
    version="0.0.1",
    author="Juan Escudero Pedrosa",
    author_email="jescudero@iaa.es",
    description="A rewrite of IOP3, a pipeline to work with photometry and polarimetry of optical data from CAHA and OSN.",
    long_description=long_description,
    long_description_content_type="text/markdown",
      url='https://github.com/juanep97/iop4',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    entry_points={'console_scripts': ['iop4=iop4lib.iop4:main',],},
)