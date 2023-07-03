#!/usr/bin/env python3
""" iop4.py
    Invoke the IOP4 pipeline.
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
from iop4lib.telescopes import *



def process_epochs(epochname_list, force_rebuild, check_remote_list):
    epoch_L = list()
    
    logger.info("Epochs will be created.")

    for epochname in epochname_list:
            epoch = Epoch.create(epochname=epochname, check_remote_list=check_remote_list)
            epoch_L.append(epoch)

    logger.info("Creating Master Biases.")

    for epoch in epoch_L:
            epoch.build_master_biases(force_rebuild=force_rebuild)

    logger.info("Creating Master Flats.")

    for epoch in epoch_L:
            epoch.build_master_flats(force_rebuild=force_rebuild)

    logger.info("Science files will be reduced.")

    rawfits = RawFit.objects.filter(epoch__in=epoch_L, imgtype=IMGTYPES.LIGHT).all()
    Epoch.reduce_rawfits(rawfits, force_rebuild=force_rebuild)

    logger.info("Computing relative photometry results.")

    for epoch in epoch_L:
            epoch.compute_relative_photometry()

    logger.info("Computing relative polarimetry results.")
    
    for epoch in epoch_L:
            epoch.compute_relative_polarimetry()


if __name__ == '__main__':
    # Parse args:

    parser = argparse.ArgumentParser(
                    prog='iop4',
                    description='Photometry and polarimetry of optical data from CAHA and OSN.',
                    epilog='Contact Juan Escudero Pedrosa (jescudero@iaa.es).',
                    allow_abbrev=False)
    
    parser.add_argument('-i', "--interactive", dest="interactive", action="store_true", help="<Optional> Jump to an IPython shell after finishing execution", required=False)

    # logging options
    parser.add_argument('--log-level', type=str, dest='log_level', choices=[None, 'debug', 'info', 'error', 'warning', 'critical'], default=None, help='Logging level to use (default: %(default)s)')   
    parser.add_argument('-m', '--mail-to', dest='mail_to', default=list(), nargs='+', help='<Optional> List of email addresses to send the log to', required=False)

    # parallelization options
    parser.add_argument("--nthreads", dest="nthreads", type=int, default=None, help="<Optional> Number of threads to use when possible (default: %(default)s)", required=False)
    parser.add_argument("--use-ray-cluster", dest="ray_use_cluster", action="store_true", help="<Optional> Use ray for parallelization", required=False)
    
    # processing options
    parser.add_argument('--retry-failed', dest='retry_failed', action='store_true', help='<Optional> Retry failed reduced fits', required=False)
    parser.add_argument('-l', '--epoch-list', dest='epochname_list', nargs='+', help='<Optional> List of epochs (e.g: T090/230102 T090/230204)', required=False)
    parser.add_argument('--skip-remote-file-list', dest='skip_remote_file_list', action='store_true', help='<Optional> Skip remote file list check', required=False)
    parser.add_argument('--discover-new', dest='discover_new', action='store_true', help='<Optional> Discover new epochs and process them (remote file list for each epoch is always checked)', required=False)
    parser.add_argument('--add-local-epochs-to-list', dest='add_local_epochs_to_list', action='store_true', help='<Optional> Add local epochs to the list created by --discover-new', required=False)
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

    # Set up use_ray:
    if args.ray_use_cluster:
        iop4conf.ray_use_cluster = True

    # Reduce indicated epochs
    
    if args.epochname_list is not None:
        process_epochs(args.epochname_list, args.force_rebuild, check_remote_list=~args.skip_remote_file_list)

    # Discover new epochs and process them
    
    if args.discover_new:   
        new_epochnames_all = set()   

        for tel_cls in Telescope.get_known():
            logger.debug(f"Listing remote epochs for {tel_cls.name}...")
            remote_epochnames = tel_cls.list_remote_epochnames()
            logger.info(f"Found {len(remote_epochnames)} remote epochs for {tel_cls.name}.")

            local_epochnames = [f"{tel_cls.name}/{night}" for night in os.listdir(f"{iop4conf.datadir}/raw/{tel_cls.name}/")]
            logger.info(f"Found {len(local_epochnames)} epochs for {tel_cls.name} in local raw archive.")

            new_epochnames = set(remote_epochnames).difference(local_epochnames)
            logger.info(f"New epochs discovered in {tel_cls.name} (n={len(new_epochnames)}): {new_epochnames}")

            new_epochnames_all = new_epochnames_all.union(new_epochnames)

        epochs_to_process = set(new_epochnames_all)

        if args.add_local_epochs_to_list:
            for tel_cls in Telescope.get_known():
                epochs_to_process = epochs_to_process.union([f"{tel_cls.name}/{night}" for night in os.listdir(f"{iop4conf.datadir}/raw/{tel_cls.name}/")])

        process_epochs(epochs_to_process, args.force_rebuild, check_remote_list=True)
                        

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