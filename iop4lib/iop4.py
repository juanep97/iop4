#!/usr/bin/env python
""" iop4.py
    Invoke the IOP4 pipeline.
"""

# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=True)

#other imports
import os
import sys
import argparse
import coloredlogs
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mplt
import matplotlib.pyplot as plt

# iop4lib imports
from iop4lib.db import *
from iop4lib.enums import *
from iop4lib.telescopes import Telescope

# logging
import logging
logger = logging.getLogger(__name__)



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



def discover_new_epochs(add_local_epochs_to_list=False):

    new_epochnames_all = set()   

    for tel_cls in Telescope.get_known():
        logger.info(f"Listing remote epochs for {tel_cls.name}...")
        
        remote_epochnames = tel_cls.list_remote_epochnames()
        logger.info(f"Found {len(remote_epochnames)} remote epochs for {tel_cls.name}.")

        if os.path.isdir(f"{iop4conf.datadir}/raw/{tel_cls.name}/"):
            local_epochnames = [f"{tel_cls.name}/{night}" for night in os.listdir(f"{iop4conf.datadir}/raw/{tel_cls.name}/")]
        else:
            local_epochnames = list()

        logger.info(f"Found {len(local_epochnames)} epochs for {tel_cls.name} in local raw archive.")

        if not add_local_epochs_to_list:
            new_epochnames = set(remote_epochnames).difference(local_epochnames)
            
        logger.info(f"New epochs discovered in {tel_cls.name} (n={len(new_epochnames)}): {new_epochnames}")

        new_epochnames_all = new_epochnames_all.union(new_epochnames)
    
    return new_epochnames_all



def discover_local_epochs():

    local_epochs = set()

    for tel_cls in Telescope.get_known():
        local_epochs = local_epochs.union([f"{tel_cls.name}/{night}" for night in os.listdir(f"{iop4conf.datadir}/raw/{tel_cls.name}/")])

    return local_epochs



def retry_failed_files():
    qs = ReducedFit.objects.filter(flags__has=ReducedFit.FLAGS.ERROR_ASTROMETRY).all()
    logger.info(f"Retrying {qs.count()} failed reduced fits.")
    Epoch.reduce_reducedfits(qs)
    qs2 = ReducedFit.objects.filter(flags__has=ReducedFit.FLAGS.ERROR_ASTROMETRY).all()
    logger.info(f"Fixed {qs.count()-qs2.count()} out of {qs.count()} failed reduced fits.")



def main():

    # Parse args:

    parser = argparse.ArgumentParser(
                    prog='iop4',
                    description='Photometry and polarimetry of optical data from CAHA and OSN.',
                    epilog='Contact Juan Escudero Pedrosa (jescudero@iaa.es).',
                    allow_abbrev=False)
    
    parser.add_argument('-i', "--interactive", dest="interactive", action="store_true", help="<Optional> Jump to an IPython shell after finishing execution", required=False)

    # logging options
    parser.add_argument('--log-level', type=str, dest='log_level', choices=['debug', 'info', 'error', 'warning', 'critical'], default=None, help='Logging level to use (default: %(default)s)')   
    parser.add_argument('-m', '--mail-to', dest='mail_to', default=list(), nargs='+', help='<Optional> List of email addresses to send the log to', required=False)

    # parallelization options
    parser.add_argument("--nthreads", dest="nthreads", type=int, default=None, help="<Optional> Number of threads to use when possible (default: %(default)s)", required=False)
    parser.add_argument("--use-ray-cluster", dest="ray_use_cluster", action="store_true", help="<Optional> Use ray for parallelization", required=False)
    
    # processing options
    parser.add_argument('-l', '--epoch-list', dest='epochname_list', nargs='+', help='<Optional> List of epochs (e.g: T090/230102 T090/230204)', required=False)
    parser.add_argument('--discover-new', dest='discover_new', action='store_true', help='<Optional> Discover new epochs to process them', required=False)
    parser.add_argument('--discover-local', dest='discover_local', action='store_true', help='<Optional> Discover local epochs to process them', required=False)
    parser.add_argument('--list-only', dest='list_only', action='store_true', help='<Optional> If given, the built list of epochs will be printed but not processed', required=False)

    ## other options
    parser.add_argument('--retry-failed', dest='retry_failed', action='store_true', help='<Optional> Retry failed reduced fits', required=False)
    parser.add_argument('--skip-remote-file-list', dest='skip_remote_file_list', action='store_true', help='<Optional> Skip remote file list check', required=False)
    parser.add_argument("--force-rebuild", dest="force_rebuild", action="store_true", help="<Optional> Force re-building of files (pass force_rebuild=True)", required=False)

    args = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_usage()
        sys.exit(0)

    # Configuration:

    ## logging

    ROOT_LOGGER = logging.getLogger()
    
    if args.log_level is not None:
        iop4conf.log_level = args.log_level.upper()
    
    ROOT_LOGGER.setLevel(iop4conf.log_level)

    logger_h1 = logging.FileHandler(iop4conf.log_fname, mode="w")
    logger_h1.setFormatter(coloredlogs.ColoredFormatter(iop4conf.log_format, datefmt=iop4conf.log_date_format))

    logger_h2 = logging.StreamHandler(sys.stdout)
    logger_h2.setFormatter(coloredlogs.ColoredFormatter(iop4conf.log_format, datefmt=iop4conf.log_date_format))

    ROOT_LOGGER.handlers.clear()
    ROOT_LOGGER.addHandler(logger_h1)
    ROOT_LOGGER.addHandler(logger_h2)

    ## parallelization:

    if args.nthreads is not None:
        iop4conf.max_concurrent_threads = args.nthreads

    if args.ray_use_cluster:
        iop4conf.ray_use_cluster = True

    # Reduce indicated epochs
    
    epochs_to_process = set()

    if args.discover_new:   
        epochs_to_process = epochs_to_process.union(discover_new_epochs())

    if args.discover_local:
        epochs_to_process = epochs_to_process.union(discover_local_epochs())

    if args.epochname_list is not None:
         epochs_to_process = epochs_to_process.union(args.epochname_list)

    if len(epochs_to_process) > 0 and not args.list_only:
        process_epochs(epochs_to_process, args.force_rebuild, check_remote_list=~args.skip_remote_file_list)
    else:
        logger.info("Invoked with --list-only:")
        logger.info(f"{epochs_to_process=}")

    # Retry failed files if indicated

    if args.retry_failed:
        retry_failed_files()

    # Start interactive shell if indicated

    if args.interactive:
        logger.info("Jumping to IPython shell.")
        import IPython
        IPython.embed(header="Start IOP4ing!", module=sys.modules['__main__'])

    sys.exit(0)



if __name__ == '__main__':
    main()
