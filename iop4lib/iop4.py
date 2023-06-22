#!/usr/bin/env python3
""" iop4.py
    Invoke the IOP4 pipeline.

    This script should be invoked with the epoch you want to reduce (e.g. TEL/YYMMDD).
    Then it should download the data if not already downloaded, and reduce it.

    $ rm ~/iop4data/iop4.db; rm -rf ./iop4api/migrations; python manage.py flush; python manage.py makemigrations iop4api; python manage.py migrate; python manage.py loaddata ../iop4lib/db/flags.json
    $ python manage.py dumpdata --natural-primary --natural-foreign --format=yaml
"""

import iop4lib.config
iop4conf = iop4lib.Config(config_db=True)

import os
import sys
import argparse
import logging
import coloredlogs

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mplt
import matplotlib.pyplot as plt

from iop4lib.db import *

if __name__ == '__main__':
    # Parse args:

    parser = argparse.ArgumentParser(
                    prog='iop4',
                    description='Photometry and polarimetry of optical data from CAHA and OSN.',
                    epilog='Contact Juan Escudero Pedrosa (jescudero@iaa.es).',
                    allow_abbrev=False)
    
    parser.add_argument('-i', "--interactive", dest="interactive", action="store_true", help="<Optional> Jump to an IPython shell after finishing execution", required=False)
    parser.add_argument('--log-level', type=str, dest='log_level', choices=[None, 'debug', 'info', 'error', 'warning', 'critical'], default=None, help='Logging level to use (default: %(default)s)')   
    parser.add_argument('-m', '--mail-to', dest='mail_to', default=list(), nargs='+', help='<Optional> List of email addresses to send the log to', required=False)
    parser.add_argument("--nthreads", dest="nthreads", type=int, default=None, help="<Optional> Number of threads to use when possible (default: %(default)s)", required=False)
    
    parser.add_argument('--retry-failed', dest='retry_failed', action='store_true', help='<Optional> Retry failed reduced fits', required=False)
    parser.add_argument('-l', '--epoch-list', dest='epochname_list', nargs='+', help='<Optional> List of epochs (e.g: T090/230102 T090/230204)', required=False)
    parser.add_argument("--force-rebuild", dest="force_rebuild", action="store_true", help="<Optional> Force re-building of files (pass force_rebuild=True)", required=False)

    args = parser.parse_args()

    # Set up logging:

    # Configure root logger

    ROOT_LOGGER = logging.getLogger()
    
    if args.log_level is not None:
        iop4conf.log_level = args.log_level.upper()
    
    ROOT_LOGGER.setLevel(iop4conf.log_level)

    logger_h1 = logging.FileHandler(iop4conf.log_fname, mode="w")
    #logger_h1.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    logger_h1.setFormatter(coloredlogs.ColoredFormatter(iop4conf.log_format, datefmt=iop4conf.log_date_format))

    logger_h2 = logging.StreamHandler(sys.stdout)
    logger_h2.setFormatter(coloredlogs.ColoredFormatter(iop4conf.log_format, datefmt=iop4conf.log_date_format))

    ROOT_LOGGER.handlers.clear()
    ROOT_LOGGER.addHandler(logger_h1)
    ROOT_LOGGER.addHandler(logger_h2)


    # get our logger, it will inherit root logger handles
    logger = logging.getLogger("iop4.py")

    # Set up number of threads:

    if args.nthreads is not None:
          iop4conf.max_concurrent_threads = args.nthreads

    # Reduce indicated epochs
    
    if args.epochname_list is not None:

        epoch_L = list()
    
        logger.info("Epochs will be created.")

        for epochname in args.epochname_list:
                epoch = Epoch.create(epochname=epochname, check_remote_list=True)
                epoch_L.append(epoch)

        logger.info("Creating Master Biases.")

        for epoch in epoch_L:
                epoch.build_master_biases(force_rebuild=args.force_rebuild)

        logger.info("Creating Master Flats.")

        for epoch in epoch_L:
                epoch.build_master_flats(force_rebuild=args.force_rebuild)

        logger.info("Science files will be reduced.")

        rawfits = RawFit.objects.filter(epoch__in=epoch_L, imgtype=IMGTYPES.LIGHT).all()
        Epoch.reduce_rawfits(rawfits, force_rebuild=args.force_rebuild)

        logger.info("Computing relative photometry results.")

        for epoch in epoch_L:
                epoch.compute_relative_photometry()

        logger.info("Computing relative polarimetry results.")
        
        for epoch in epoch_L:
                epoch.compute_polarimetry()


    if args.retry_failed:
        qs = ReducedFit.objects.filter(flags__has=ReducedFit.FLAGS.ERROR_ASTROMETRY).all()
        logger.info(f"Retrying {qs.count()} failed reduced fits.")
        Epoch.reduce_reducedfits(qs)
        qs2 = ReducedFit.objects.filter(flags__has=ReducedFit.FLAGS.ERROR_ASTROMETRY).all()
        logger.info(f"Fixed {qs.count()-qs2.count()} out of {qs.count()} failed reduced fits.")


    if args.interactive:
        logger.info("Jumping to IPython shell.")
        import IPython
        IPython.embed(header="Start IOP4ing!", module=sys.modules['__main__'], user_ns=sys.modules['__main__'].__dict__)

    sys.exit(0)