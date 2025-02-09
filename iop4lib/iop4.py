#!/usr/bin/env python
""" iop4.py
    Invoke the IOP4 pipeline.

You can specify how to select epochs using the --epoch-list, --discover-missing 
and --list-local options. You can check whether to keep selected epochs that are 
already in DB using the --no-check-db option, otherwise they will be removed 
from the list of epochs to process. The --date-start and --date-end options 
allow to filter the selected epochs by date. 

Equivalent options exist for files.

You can override config options with environment variables, e.g., 
`IOP4_LOG_LEVEL=DEBUG iop4 [other options]`. This will work for any process 
using iop4lib (i.e., iop4 main script, API, etc.). E.g. to set the log level to
DEBUG, you would invoke iop4 as `IOP4_LOG_LEVEL=DEBUG iop4 [other options]`. 
Options set with environment variables will take precedence over config file 
options.

You can also use -o option=value to override config options, e.g., to set the 
log level to DEBUG and use 6 parallel processes, you would invoke iop4 as 
`iop4 -o log_level=DEBUG -o n_processes=6 [other options]`. This will work only 
for the iop4 main script. Options set with -o will take precedence over any 
other configuration method.

Contact: Juan Escudero Pedrosa (jescudero@iaa.es).
"""

import os
import sys
import argparse
import coloredlogs
import datetime

# logging
import logging
logger = logging.getLogger(__name__)

from typing import Sequence, Iterable

def process_epochs(epochname_list: Iterable[str], args):
    from iop4lib.db import Epoch, RawFit, PhotoPolResult
    from iop4lib.enums import IMGTYPES, OBSMODES

    epoch_L  : list[Epoch] = list()
    
    logger.info("Epochs will be created.")

    for epochname in epochname_list:
        epoch = Epoch.create(epochname=epochname, check_remote_list=~args.skip_remote_file_list)
        epoch_L.append(epoch)

    logger.info("Creating Master Biases")

    for epoch in epoch_L:
        
        epoch.build_master_biases(force_rebuild=args.force_rebuild)

    logger.info("Creating Master Darks.")

    for epoch in epoch_L:
        epoch.build_master_darks(force_rebuild=args.force_rebuild)
        
    logger.info("Creating Master Flats.")

    for epoch in epoch_L:
        epoch.build_master_flats(force_rebuild=args.force_rebuild)

    logger.info("Science files will be reduced.")

    rawfits = RawFit.objects.filter(epoch__in=epoch_L, imgtype=IMGTYPES.LIGHT).all()
    Epoch.reduce_rawfits(rawfits.filter(obsmode=OBSMODES.PHOTOMETRY), force_rebuild=args.force_rebuild)
    Epoch.reduce_rawfits(rawfits.filter(obsmode=OBSMODES.POLARIMETRY), force_rebuild=args.force_rebuild)

    logger.info("Computing results.")

    for epoch in epoch_L:
        PhotoPolResult.objects.filter(epoch=epoch).delete()
        epoch.compute_relative_photometry()
        epoch.compute_relative_polarimetry()

    logger.info("Auto-flagging points.")
    
    for result in PhotoPolResult.objects.filter(epoch__in=epoch_L).all():
        result.auto_flag()

    logger.info("Applying corrections.")

    for epoch in epoch_L:
        results = PhotoPolResult.objects.filter(epoch=epoch).all()
        for result in results:
            try:
                result.compute_host_galaxy_correction()
            except PhotoPolResult.NoHostCorrectionAvailable:
                pass
            except Exception as e:
                logger.error(f"Error computing host galaxy correction for {result}: {e}")

    logger.info("Done.")

def process_astrosource(args):
    from iop4lib.db import ReducedFit, AstroSource, PhotoPolResult, Epoch
    
    astrosource = AstroSource.objects.get(name=args.astrosource)

    qs_all = ReducedFit.objects.filter(epoch__night__gte=args.date_start, epoch__night__lte=args.date_end)

    logger.info(f"Found {qs_all.count()} reduced fits between {args.date_start} and {args.date_end}.")

    if args.only_sources_in_field:
        qs_all = qs_all.filter(sources_in_field__name=args.astrosource)
        redfL = list(qs_all)
    else:
        # filter files that have identified this source (in sources_in_field) or have this source as header_hintobject
        redfL = list()
        for redf in qs_all:
            if redf.sources_in_field.filter(name=args.astrosource).exists() or redf.header_hintobject == astrosource:
                redfL.append(redf)

    qs_redf = ReducedFit.objects.filter(pk__in=[redf.pk for redf in redfL])

    logger.info(f"Found {len(redfL)} reduced fits for astrosource {args.astrosource}")

    if args.force_rebuild:
        logger.info("Forcing rebuild of reduced fits.")
        Epoch.reduce_reducedfits(redfL)
    
    if args.retry_failed:
        logger.info("Retrying failed reduced fits.")
        redfL_failed = [redf for redf in redfL if redf.has_flag(ReducedFit.FLAGS.ERROR_ASTROMETRY)]
        Epoch.reduce_reducedfits(redfL_failed)

    if args.check_sources_in_field_again:
            for reducedfit in redfL:
                sources_in_field = AstroSource.get_sources_in_field(fit=reducedfit)    
                logger.debug(f"{reducedfit}: found {len(sources_in_field)} sources in field.")        
                reducedfit.sources_in_field.set(sources_in_field, clear=True)

    if args.recompute:
        qs_res = PhotoPolResult.objects.filter(reducedfits__in=redfL)
        n_res_before = qs_res.count()

        logger.info(f"Recomputing {n_res_before} results.")

        deletion = qs_res.delete()

        logger.info(f"Deleted {deletion[0]} results.")

        epoch_L = set([redf.epoch for redf in redfL])

        for epoch in epoch_L:
            epoch.compute_relative_photometry(qs_redf)
            epoch.compute_relative_polarimetry(qs_redf)

        qs_res = PhotoPolResult.objects.filter(reducedfits__in=redfL)
        n_res_after = qs_res.count()

        logger.info(f"Recomputed {n_res_after} ({n_res_after-n_res_before} new).")

        logger.info("Auto-flagging points.")
        
        for result in PhotoPolResult.objects.filter(epoch__in=epoch_L).all():
            result.auto_flag()

        logger.info("Applying corrections.")

        for epoch in epoch_L:
            results = PhotoPolResult.objects.filter(epoch=epoch).all()
            for result in results:
                try:
                    result.compute_host_galaxy_correction()
                except PhotoPolResult.NoHostCorrectionAvailable:
                    pass
                except Exception as e:
                    logger.exception(f"Error computing host galaxy correction for {result}.")

def list_local_epochnames() -> list[str]:
    """List all local epochnames in local archives (by looking at the raw directory)."""

    import iop4lib.config
    iop4conf = iop4lib.Config()

    from iop4lib.telescopes import Telescope

    local_epochnames = list()

    for tel_cls in Telescope.get_known():
        if os.path.isdir(f"{iop4conf.datadir}/raw/{tel_cls.name}/"):
            local_epochnames.extend([f"{tel_cls.name}/{night}" for night in os.listdir(f"{iop4conf.datadir}/raw/{tel_cls.name}/")])

    return local_epochnames
    
def list_remote_epochnames() -> list[str]:
    """List all remote epochnames in remote archives."""

    from iop4lib.telescopes import Telescope

    epochnames = list()

    for tel_cls in Telescope.get_known():
        epochnames.extend(tel_cls.list_remote_epochnames())

    return epochnames


def discover_missing_epochs() -> list[str]:
    """Discover missing epochs in remote archive."""
    return list(set(list_remote_epochnames()).difference(list_local_epochnames()))    


def list_remote_filelocs(epochnames: Sequence[str] = None) -> list[str]:
    """Discover files in remote archive for the given epochs.
    
    Use this function to list all files in the remote archive for the given epochs.
    It avoids calling list_remote_raw_fnames() for each epoch.
    """
    
    from iop4lib.telescopes import Telescope

    if epochnames is None:
         epochnames = list_remote_epochnames()

    grouped_epochnames = group_epochnames_by_telescope(epochnames)

    filelocs = list()

    for tel, epochnames in grouped_epochnames.items():
        if len(epochnames) > 0:
            filelocs.extend(Telescope.by_name(tel).list_remote_filelocs(epochnames))

    return filelocs

def list_local_filelocs() -> list[str]:
    """Discover local filelocs in local archive."""

    from iop4lib.telescopes import Telescope

    local_filelocs = list()

    for tel_cls in Telescope.get_known():
        if os.path.isdir(f"{iop4conf.datadir}/raw/{tel_cls.name}/"):
            for d in os.scandir(f"{iop4conf.datadir}/raw/{tel_cls.name}/"):
                local_filelocs.extend([f"{tel_cls.name}/{d.name}/{f.name}" for f in os.scandir(f"{iop4conf.datadir}/raw/{tel_cls.name}/{d.name}")])

    return local_filelocs

def discover_missing_filelocs() -> list[str]:
    """ Discover missing files in remote archive.

    Compares the lists of remote files with the list of local files and returns the fileloc of the missing files.

    If epochnames is None, all remote epochs will be checked.
    """

    return list(set(list_remote_filelocs()).difference(list_local_filelocs()))


def retry_failed_files():
    from iop4lib.db import ReducedFit, Epoch
    qs = ReducedFit.objects.filter(flags__has=ReducedFit.FLAGS.ERROR_ASTROMETRY).all()
    logger.info(f"Retrying {qs.count()} failed reduced fits.")
    Epoch.reduce_reducedfits(qs)
    qs2 = ReducedFit.objects.filter(flags__has=ReducedFit.FLAGS.ERROR_ASTROMETRY).all()
    logger.info(f"Fixed {qs.count()-qs2.count()} out of {qs.count()} failed reduced fits.")



def filter_epochname_by_date(epochname_list: Sequence[str], date_start: str = None, date_end: str = None) -> list[str]:
    from iop4lib.db import Epoch

    if date_start is not None:
        epochname_list = [epochname for epochname in epochname_list if Epoch.epochname_to_tel_night(epochname)[1] >= datetime.date.fromisoformat(date_start)]
    
    if date_end is not None:
        epochname_list = [epochname for epochname in epochname_list if Epoch.epochname_to_tel_night(epochname)[1] <= datetime.date.fromisoformat(date_end)]
    
    return epochname_list

def filter_filelocs_by_date(fileloc_list: Iterable[str], date_start: str = None, date_end: str = None) -> list[str]:
    from iop4lib.db import RawFit

    if date_start is not None:
        fileloc_list = [fileloc for fileloc in fileloc_list if RawFit.fileloc_to_tel_night_filename(fileloc)[1] >= datetime.date.fromisoformat(date_start)]
    
    if date_end is not None:
        fileloc_list = [fileloc for fileloc in fileloc_list if RawFit.fileloc_to_tel_night_filename(fileloc)[1] <= datetime.date.fromisoformat(date_end)]
    
    return fileloc_list


def group_epochnames_by_telescope(epochnames: Iterable[str]) -> dict[str, list[str]]:
    from iop4lib.db import Epoch
    from iop4lib.telescopes import Telescope

    epochnames_by_telescope = {tel_cls.name:list() for tel_cls in Telescope.get_known()}

    for epochname in epochnames:
        tel, night = Epoch.epochname_to_tel_night(epochname)
        epochnames_by_telescope[tel].append(epochname)

    return epochnames_by_telescope

def group_filelocs_by_telescope(filelocs: Iterable[str]) -> dict[str, list[str]]:
    from iop4lib.db import RawFit
    from iop4lib.telescopes import Telescope

    filelocs_by_telescope = {tel_cls.name:list() for tel_cls in Telescope.get_known()}

    for fileloc in filelocs:
        tel, night, filename = RawFit.fileloc_to_tel_night_filename(fileloc)
        filelocs_by_telescope[tel].append(fileloc)

    return filelocs_by_telescope


def parse_config_overrides(overrides: Iterable[str]) -> dict:

    config = dict()

    for override in overrides:
        key, value = override.split('=')
        config[key] = value

    # correct the type of the config options
    for k, v in config.items():
        # interpret the log_level option
        if k == "log_level" and not v.isnumeric():
            v = config[k] = getattr(logging, v.upper())
        # check the type of the config option, if it is an int, convert it
        if k in iop4conf and isinstance(iop4conf[k], int):
            config[k] = int(v)
            
    return config

def main():
    # Parse args:

    parser = argparse.ArgumentParser(
                    prog="iop4",
                    description="The Interactive Optical Photo-Polarimetric Python Pipeline",
                    epilog = __doc__,
                    formatter_class=argparse.RawDescriptionHelpFormatter,
                    allow_abbrev=False)
    
    parser.add_argument('-i', "--interactive", dest="interactive", action="store_true", help="<Optional> Jump to an IPython shell after finishing execution")
    parser.add_argument('--check-config', action="store_true", help="<Optional> Check the current configuration and exit")
    parser.add_argument('--config-file', dest='config_file', default=None, help="<Optional> Path to the config file to use. By default, IOP4 will try to use (in order) the file in IOP4_CONFIG_FILE env var, ~/.iop4.config.yaml, or the the example config file.")
    parser.add_argument('-o', action='append', dest='config_overrides', help="Override a config option (e.g., -o option=value)")

    # epoch processing options
    parser.add_argument('--epoch-list', dest='epochname_list', nargs='+', help='<Optional> List of epochs (e.g: T090/230102 T090/230204)')
    parser.add_argument('--discover-missing', action='store_true', help='<Optional> Discover new epochs to process them')
    parser.add_argument('--list-local', action='store_true', help='<Optional> Discover local epochs to process them')
    parser.add_argument('--list-only', action='store_true', help='<Optional> If given, the built list of epochs will be printed but not processed')
    parser.add_argument('--no-check-db', dest='keep_epochs_in_db', action='store_true', help='<Optional> Process discovered epochs even if they existed in archive')

    ## file processing  options
    parser.add_argument('--file-list', dest='fileloc_list', nargs='+', help='<Optional> List of files (e.g: tel/yyyy-mm-dd/name))')
    parser.add_argument('--discover-missing-files', action='store_true', help='<Optional> Discover files in remote archives that are not present in archive')
    parser.add_argument('--list-local-files', action='store_true', help='<Optional> Discover local files to process them')
    parser.add_argument('--list-files-only', action='store_true', help='<Optional> If given, the built list of filelocs will be printed but not processed')
    parser.add_argument('--no-check-db-files',  dest='keep_files_in_db', action='store_true', help='<Optional> Process discovered files even if they existed in archive')

    # astrosource processing options
    parser.add_argument('--astrosource', type=str, default=None, help='<Optional> Select files only of this source')
    parser.add_argument('--recompute', action='store_true', help='<Optional> Recompute photometry and polarimetry results')
    parser.add_argument('--retry-failed', action='store_true', help='<Optional> Retry failed reduced fits')
    parser.add_argument('--only-sources-in-field', action='store_true', help='<Optional> Only process files that have this source in sources_in_field')
    parser.add_argument('--check-sources-in-field-again', action='store_true', help='<Optional> Check sources_in_field again')

    # other options
    parser.add_argument('--skip-remote-file-list', action='store_true', help='<Optional> Skip remote file list check')
    parser.add_argument("--force-rebuild", action="store_true", help="<Optional> Force re-building of files (pass force_rebuild=True)")

    # range
    parser.add_argument('--date-start', '-s', type=str, default=None, help='<Optional> Start date (YYYY-MM-DD)')
    parser.add_argument('--date-end', '-e', type=str, default=None, help='<Optional> End date (YYYY-MM-DD)')

    args = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_usage()
        sys.exit(0)

    # Configuration:

    global iop4conf
    import iop4lib.config
    iop4conf = iop4lib.Config(config_path=args.config_file, config_db=True)

    print("Using config file:", iop4conf.config_path)
    
    from iop4lib.db import Epoch, RawFit
    from iop4lib.telescopes import Telescope

    if args.config_overrides:
        # get the config options as a dict
        opts = parse_config_overrides(args.config_overrides)
        # update the config
        iop4conf.update(opts)

    if not iop4conf.is_valid():
        print("There are some missing keys in the config file. All keys in the example config file must be present. Add them to your config file (you can set them to null) and try again. Aborting.")
        sys.exit(-1)
    elif args.check_config:
        print("Config file is OK.")
        sys.exit(0)

    ## configure logging

    ROOT_LOGGER = logging.getLogger()
    
    ROOT_LOGGER.setLevel(iop4conf.log_level)

    logger_h1 = logging.FileHandler(iop4conf.log_file, mode="w")
    logger_h1.setFormatter(coloredlogs.ColoredFormatter(iop4conf.log_format, datefmt=iop4conf.log_date_format))

    logger_h2 = logging.StreamHandler(sys.stdout)
    logger_h2.setFormatter(coloredlogs.ColoredFormatter(iop4conf.log_format, datefmt=iop4conf.log_date_format))

    ROOT_LOGGER.handlers.clear()
    ROOT_LOGGER.addHandler(logger_h1)
    ROOT_LOGGER.addHandler(logger_h2)

    # Epochs
    
    epochnames_to_process = set()

    if args.list_local:
        logger.info("Listing local epochs...")
        local_epochs = list_local_epochnames()
        epochnames_to_process = epochnames_to_process.union(local_epochs)
        logger.info(f"Listed {len(local_epochs)} local epochs.")

    if args.discover_missing:
        logger.info("Discovering missing epochs...")
        missing_epochs = discover_missing_epochs()
        epochnames_to_process = epochnames_to_process.union(missing_epochs)
        logger.info(f"Discovered {len(missing_epochs)} missing epochs.")

    if args.epochname_list is not None:
         logger.info("Adding epochs from command line...")
         epochnames_to_process = epochnames_to_process.union(args.epochname_list)
         logger.info(f"Added {len(args.epochname_list)} epochs from command line.")

    if len(epochnames_to_process) > 0 and not args.keep_epochs_in_db:
        logger.info("Removing epochs already in the DB...")
        epochnames_in_db = set([epoch.epochname for epoch in Epoch.objects.all()])
        epochnames_to_process = epochnames_to_process.difference(epochnames_in_db)
        logger.info(f"Left {len(epochnames_to_process)} epochs to process.")

    logger.info(f"Gathered {len(epochnames_to_process)} epochs to process between {args.date_start} and {args.date_end}.")

    if args.date_start is not None or args.date_end is not None:
        logger.info("Filtering epochs by date.")
        epochnames_to_process = filter_epochname_by_date(epochnames_to_process, args.date_start, args.date_end)
        logger.info(f"Filtered to {len(epochnames_to_process)} epochs to process between {args.date_start} and {args.date_end}.")

    logger.debug(f"{epochnames_to_process=}")

    if not args.list_only:
        if len(epochnames_to_process) > 0:
            logger.info("Processing epochs.")
            epochnames_to_process, _ = list(zip(*sorted([(epochname, Epoch.epochname_to_tel_night(epochname)[1].strftime("%Y-%m-%d")) for epochname in epochnames_to_process], key=lambda x: x[1], reverse=True)))
            process_epochs(epochnames_to_process, args)
    else:
        logger.info("Invoked with --list-only!")

    # Files

    filelocs_to_process = set()

    if args.list_local_files:
        logger.info("Listing local files...")
        filelocs_local = list_local_filelocs()
        filelocs_to_process = filelocs_to_process.union(filelocs_local)
        logger.info(f"Listed {len(filelocs_local)} local files.")
    else:
        filelocs_local = set()

    if args.discover_missing_files:
        logger.info("Discovering missing files...")
        filelocs_missing = discover_missing_filelocs()
        filelocs_to_process = filelocs_to_process.union(filelocs_missing)
        logger.info(f"Discovered {len(filelocs_missing)} missing files.")
    else:
        filelocs_missing = set()

    if args.fileloc_list is not None:
        logger.info("Adding files from command line...")
        filelocs_to_process = filelocs_to_process.union(args.file_list)
        logger.info(f"Added {len(args.file_list)} files from command line.")

    if len(filelocs_to_process) > 0 and not args.keep_files_in_db:
        logger.info(f"Removing files already in the DB ({RawFit.objects.count()}).")
        filelocs_in_db = set([rawfit.fileloc for rawfit in RawFit.objects.all()])
        filelocs_to_process = filelocs_to_process.difference(filelocs_in_db)
        logger.info(f"Left {len(filelocs_to_process)} files to process.")

    logger.info(f"Gathered {len(filelocs_to_process)} files to process.")

    if args.date_start is not None or args.date_end is not None:
        logger.info("Filtering files by date...")
        filelocs_to_process = filter_filelocs_by_date(filelocs_to_process, args.date_start, args.date_end)    
        logger.info(f"Filtered to {len(filelocs_to_process)} filelocs_to_process  between {args.date_start} and {args.date_end}.")

        filelocs_missing = set(filelocs_to_process).intersection(filelocs_missing)

    logger.debug(f"{filelocs_to_process=}")

    if not args.list_files_only:

        if args.discover_missing_files and len(filelocs_missing) > 0:
            logger.info("Downloading missing files.")
            for telname, filelocs in group_filelocs_by_telescope(filelocs_missing).items():
                Telescope.by_name(telname).download_rawfits(filelocs)

        if len(filelocs_to_process) > 0:
            logger.info("Registering files.")
            for fileloc in filelocs_to_process:
                rawfit = RawFit.create(fileloc=fileloc)

    else:
        logger.info("Invoked with --list-files-only")

    # Astrosource

    if args.astrosource is not None:
        process_astrosource(args)

    # Start interactive shell if indicated

    if args.interactive:
        logger.info("Jumping to IPython shell.")
        import IPython
        _ns = dict(globals())
        _ns.update(locals())
        IPython.embed(header="Start IOP4ing!", module=sys.modules['__main__'], user_ns=_ns)

    sys.exit(0)



if __name__ == '__main__':
    main()
