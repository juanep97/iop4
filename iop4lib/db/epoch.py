import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

import django
from django.db import models
from django.db.models import Count, Q
from django.db.models.functions import Concat

# other imports

import re
import os
import datetime
import psutil
import gc
import tracemalloc
from pympler import tracker, summary, muppy
import multiprocessing, concurrent.futures
import time
import io
import contextlib
import time
import signal
import numpy as np
import itertools
from astropy.coordinates import SkyCoord
from pathlib import Path
import astropy.io.fits as fits
import glob

# iop4 imports

from iop4lib.enums import *
from iop4lib.telescopes import Telescope
from .astrosource import AstroSource
from .fields import FlagChoices, FlagBitField
from iop4lib.utils import  get_mem_parent_from_child, get_total_mem_from_child, get_mem_current, get_mem_children

# logging

import logging
logger = logging.getLogger(__name__)


class HybridProperty():
    def __init__(self, fget=None, fset=None, fexp=None):
        self.fget = fget
        self.fset = fset
        self.fexp = fexp

    def getter(self, fget):
        return type(self)(fget=fget, fset=self.fset, fexp=self.fexp)
    
    def setter(self, fset):
        return type(self)(fget=self.fget, fset=fset, fexp=self.fexp)
    
    def expression(self, fexp):
        return type(self)(fget=self.fget, fset=self.fset, fexp=fexp)
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return self.fget(instance)

    def __set__(self, instance, value):
        if self.fset is not None:
            self.fset(instance, value)
        else:
            raise AttributeError("Can't set attribute")

def hybrid_property(fget):
    res =  HybridProperty(fget=fget)
    if fget.__name__ is not None:
        res.__name__ = fget.__name__
    return res


class HybridManager(models.Manager):
    def get_queryset(self):
        qs = super().get_queryset()
        print(f"get_queryset: {qs=}, {qs.model=}")
        for name, value in vars(qs.model).items():
            if isinstance(value, HybridProperty) and value.expression is not None:
                print(f"Getting qs of {name} with {value=}")
                qs = qs.annotate(**{name: value.fexp(qs.model)})
                print("Got qs")
        return qs
    
class Epoch(models.Model):
    """A class representing an epoch.
    
    Identified by the telescope and date of the night. Provides method for fetching the data
    from the telescope archives and reducing the data.
    """

    objects = models.Manager()

    custom = HybridManager()
    @hybrid_property
    def epochname_(self):
        print(self)
        return f"{self.telescope}/{self.night}"

    @epochname_.expression
    def cush(self):
        return Concat('telescope', models.Value('/'), 'night', output_field=models.CharField())



    # Database fields and information

    # identifiers 
    
    telescope = models.CharField(max_length=255, choices=TELESCOPES.choices)
    night = models.DateField()

    # flags

    class FLAGS(FlagChoices):
        ERROR = 1 << 1
        ERROR_LISTING = 1 << 2
        ERROR_LISTING_REMOTE = 1 << 3
        ERROR_LISTING_LOCAL = 1 << 4
        LISTED = 1 << 5
        
    flags = FlagBitField(choices=FLAGS.choices(), default=0)

    # db info and constraints

    class Meta:
        app_label = 'iop4api'
        verbose_name = "Epoch"
        verbose_name_plural = "Epochs"
        constraints = [
            models.UniqueConstraint(fields=['telescope', 'night'], name='telescope_night_unique')
        ]
    
    # properties of epoch

    @property
    def epochname(self):
        """Returns a human readable string that uniquely identifies the epoch, as an alternative to the id."""
        return f"{self.telescope}/{self.night.strftime('%Y-%m-%d')}"

    @property
    def rawfitsdir(self):
        """Returns the path to the directory where the raw fits of this epoch are stored."""
        return os.path.join(iop4conf.datadir, "raw", self.epochname)
    
    @property
    def calibrationdir(self):
        """Returns the path to the directory where the calibration files of this epoch are stored."""
        return os.path.join(iop4conf.datadir, "calibration", self.epochname)
    
    @property
    def masterbiasdir(self):
        """Returns the path to the directory where the masterbias files of this epoch are stored."""
        return os.path.join(iop4conf.datadir, "masterbias", self.epochname)

    @property
    def masterflatdir(self):
        """Returns the path to the directory where the masterflat files of this epoch are stored."""
        return os.path.join(iop4conf.datadir, "masterflat", self.epochname)

    @property
    def yyyymmdd(self):
        """Returns the date of the epoch in the format YYYYMMDD."""
        return self.night.strftime("%Y%m%d")
    
    @property
    def yymmdd(self):
        """Returns the date of the epoch in the format YYMMDD."""
        return self.night.strftime("%y%m%d")
    
    # helper property to return the jyear of the epoch at noon (12:00)

    @property
    def jyear(self):
        """Returns the jyear of the epoch at noon (before the night)."""

        from astropy.time import Time
        from datetime import datetime, date, time

        return Time(datetime.combine(self.night, time(hour=12))).jyear

    # helper properties that return counts to files of this epoch.

    @property
    def rawcount(self):
        from iop4lib.db import RawFit
        return self.rawfits.count()

    @property
    def rawbiascount(self):
        from iop4lib.db import RawFit
        return self.rawfits.filter(imgtype=IMGTYPES.BIAS).count()
    
    @property
    def rawflatcount(self):
        from iop4lib.db import RawFit
        return self.rawfits.filter(imgtype=IMGTYPES.FLAT).count()
    
    @property
    def rawlightcount(self):
        from iop4lib.db import RawFit
        return self.rawfits.filter(imgtype=IMGTYPES.LIGHT).count()

    @property
    def reducedcount(self):
        from iop4lib.db import ReducedFit
        return ReducedFit.objects.filter(rawfit__epoch=self.id).count()

    @property
    def calibratedcount(self):
        from iop4lib.db import ReducedFit
        return ReducedFit.objects.filter(rawfit__epoch=self.id, flags__has=ReducedFit.FLAGS.BUILT_REDUCED).count()

    # repr and str

    def _repr_html_(self):
        return (f"Epoch(id {self.id}):<br>"
                f" - telescope: {self.telescope}<br>"
                f" - night: {self.night}<br>"
                f" - {self.rawcount} rawfits: {self.rawbiascount} bias, {self.rawflatcount} flat, {self.rawlightcount} light<br>"
                f" - summary status: {self.get_summary_rawfits_status()}")

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text(f"{self!r}")
        else:
            with p.group(4, '<Epoch(', ')>'):
                p.text(f"id: {self.id},")
                p.breakable()
                p.text(f"telescope: {self.telescope},")
                p.breakable()
                p.text(f"night: {self.night},")
                p.breakable()
                p.text(f"rawcount: {self.rawcount}, rawbiascount: {self.rawbiascount}, rawflatcount: {self.rawflatcount}, rawlightcount: {self.rawlightcount},")
                p.breakable()
                p.text(f"summary status: {self.get_summary_rawfits_status()}")


    def __repr__(self):
        return f"Epoch.objects.get(id={self.id!r})"
    
    def __str__(self):
        return f"<Epoch {self.id} | {self.epochname}>"

    # creator

    @staticmethod
    def epochname_to_tel_night(epochname):
        """Parses an epochname to a telescope and night."""

        matches = re.findall(r"([a-zA-Z0-9]+)/([0-9]{2,4}-?[0-9]{2}-?[0-9]{2})$", epochname)

        if len(matches) != 1:
            raise Exception(f"Epoch name {epochname} is not TEL/YYMMDD, TEL/YYYYMMDD or TEL/YYYY-MM-DD.")

        telescope = matches[0][0]
        if Telescope.is_known(telescope):
            telescope = Telescope.by_name(telescope).name
        else:
            raise Exception(f"Telescope {telescope} not known.")

        if len(matches[0][1]) == 6: ## yymmdd
            logger.debug("Epoch name was specified in YYMMDD format, it will automatically casted to 20YYMMDD format.")
            yymmdd = matches[0][1]
            yyyymmdd = f"20{yymmdd}"
        elif len(matches[0][1]) == 8: # yyyymmdd
            yyyymmdd = matches[0][1]
            yymmdd = matches[0][1][:-6]
        elif len(matches[0][1]) == 10: ## yyyy-mm-dd
            yyyymmdd = "".join(matches[0][1].split("-"))
            yymmdd = yyyymmdd[2:]
        

        night = datetime.datetime.strptime(yyyymmdd, "%Y%m%d").date()

        return telescope, night

    @classmethod
    def create(cls, 
                 telescope=None, night=None, epochname=None,
                 auto_link_rawfits=True,
                 check_remote_list=False,
                 force_redownload=False,
                 fallback_to_local=True,
                 auto_merge_to_db=True,
                 rawfits_kwargs=dict()):
        """Create an Epoch object for the given telescope and night, reusing an existing DB entry if it exists.

        Parameters
        ----------
        epochname : str
            TEL/YYYY-MM-DD, TEL/YYYYMMDD or TEL/YYMMDD ot (converted to  TEL/20YYMMDD)

        """

        if telescope is not None and night is not None:
            pass
        elif epochname is not None:
            telescope, night = Epoch.epochname_to_tel_night(epochname)
        else:
            raise ValueError("Either telescope and night or epochname must be specified.")
        
        # create entry

        if (epoch := Epoch.objects.filter(telescope=telescope, night=night).first()) is  None:
            logger.debug(f"Creating DB entry for {telescope}/{night}.")
            epoch = cls(telescope=telescope, night=night)
            epoch.save()
        else:
            logger.debug(f"Epoch for {telescope}/{night} already exists in the DB, it will be used instead.")

        # clear flags on creation
        #epoch.clear_flags()

        # instance only attributes
        epoch.auto_link_rawfits = auto_link_rawfits
        epoch.fallback_to_local = fallback_to_local
        epoch.check_remote_list = check_remote_list
        epoch.force_redownload = force_redownload
        epoch.rawfits_kwargs = rawfits_kwargs
        epoch.auto_merge_to_db = auto_merge_to_db

        if auto_merge_to_db:
            epoch.save()
        
        if auto_link_rawfits:
            epoch.link_rawfits()

        return epoch

    @classmethod 
    def from_db(cls, db, *args, **kwargs):
        """Provides some defaults for attributes that are not stores in DB.

        Should be the same as default kwargs in .create() for consistency.
        This method is executed when getting the instance from DB with a query.
        """
        instance = super(Epoch, cls).from_db(db, *args, **kwargs)
        instance.auto_link_rawfits = True
        instance.check_remote_list = False
        instance.force_redownload = False
        instance.fallback_to_local = True
        instance.auto_merge_to_db = True
        instance.rawfits_kwargs = dict()
        return instance
      
    # methods

    def list_remote_raw_fnames(self):
        """Checks remote file list and builds self.rawfits from them.
        """
        return Telescope.by_name(self.telescope).list_remote_raw_fnames(self)

            
    def link_rawfits(self):
        """ Links rawfits to this epoch. """

        from iop4lib.db import RawFit

        self.unset_flag(Epoch.FLAGS.LISTED)

        if not os.path.isdir(self.rawfitsdir):
            logger.debug(f"Dir {self.rawfitsdir} does not exist, creating it.")
            os.makedirs(self.rawfitsdir) 

        local_file_L = os.listdir(self.rawfitsdir)

        if self.check_remote_list:
            logger.info(f"{self}: fetching file list for epoch.")
            try:
                self.fnameL = self.list_remote_raw_fnames()
                logger.info(f"{self}: listed {len(self.fnameL)} files in remote.")
                self.unset_flag(Epoch.FLAGS.ERROR_LISTING_REMOTE)
                self.set_flag(Epoch.FLAGS.LISTED)
            except Exception as e:
                logger.warning(f"{self}: error listing remote dir, maybe it is an old epoch?: {e}.")
                self.set_flag(Epoch.FLAGS.ERROR_LISTING_REMOTE)


        if (self.has_flag(Epoch.FLAGS.ERROR_LISTING_REMOTE) or not self.check_remote_list) and self.fallback_to_local:
            if len(local_file_L) == 0:
                logger.error(f"{self}: used fallback to local, but there was an error listing local dir (0 files found)")
                self.set_flag(Epoch.FLAGS.ERROR_LISTING_LOCAL)
            else:
                self.fnameL = local_file_L
                self.unset_flag(Epoch.FLAGS.ERROR_LISTING_LOCAL)
                self.set_flag(Epoch.FLAGS.LISTED)

        if self.has_flag(Epoch.FLAGS.LISTED):

            # we could allow download on create but better to do it on bulk, it is more efficient (less logins). Also CAHA will block you if you login 2 fast 2 furious.

            rawfits = [RawFit.create(epoch=self, filename=filename, auto_procure_local=False, **self.rawfits_kwargs) for filename in self.fnameL]

            missing_files = set(self.fnameL) - set(local_file_L)

            if self.force_redownload:
                Telescope.by_name(self.telescope).download_rawfits(rawfits)
            elif len(missing_files)>0:
                Telescope.by_name(self.telescope).download_rawfits([rawfit for rawfit in rawfits if rawfit.filename in missing_files])

            for rawfit in rawfits:
                rawfit.procure_local_file()
                rawfit.classify()

            self.rawfits.set(rawfits)
        else:
            logger.error("Epoch not listed, not linking rawfits.")

        if self.auto_merge_to_db:
            self.save()



    def get_summary_rawfits_status(self):
        """
            Returns Flag() object if common to all fits
        """
        from iop4lib.db import RawFit

        flag_vals_present = self.rawfits.all().values_list('flags', flat=True).distinct()
        flag_label_set = set(itertools.chain.from_iterable([RawFit.FLAGS.get_labels(val) for val in flag_vals_present]))
        return ", ".join(flag_label_set)



    def build_master_biases(self, force_rebuild=False):
        from iop4lib.db import RawFit, MasterBias
        import itertools

        # define keywords to be used to build master biases
        kw_L = MasterBias.mbargs_kwL

        # for each keyword, get the set of values in the rawfits of this epoch
        kw_set_D = {kw:set(self.rawfits.filter(imgtype=IMGTYPES.BIAS).values_list(kw, flat=True).distinct()) for kw in kw_L}

        # create a list of dictionaries with all the combinations of values for each keyword
        mbargs_L = [dict(zip(kw_L, prod)) for prod in itertools.product(*kw_set_D.values())]
        
        # create master biases

        try:
            for mbargs in mbargs_L:
                mbargs['epoch'] = self
                logger.debug(f"{mbargs=}")
                if self.rawfits.filter(imgtype=IMGTYPES.BIAS, **mbargs).count() > 0:
                    logger.info(f"Building masterbias for {MasterBias.mbargs2str(mbargs)}.")
                    MasterBias.create(**mbargs, force_rebuild=force_rebuild)
                else:
                    logger.debug(f"No masterbias will be built for this mbargs since there are no files for it.")
        except Exception as e:
            logger.error(f"Error building masterbias for {self.epochname}: {e}.")
            self.set_flag(Epoch.FLAGS.ERROR)
            
        if self.auto_merge_to_db:
            self.save()



    def build_master_flats(self, force_rebuild=False):
        from iop4lib.db import RawFit, MasterFlat
        import itertools

        # define keywords to be used to build master flats
        kw_L = MasterFlat.mfargs_kwL

        # for each keyword, get the set of values in the rawfits of this epoch
        kw_set_D = {kw:set(self.rawfits.filter(imgtype=IMGTYPES.FLAT).values_list(kw, flat=True).distinct()) for kw in kw_L}

        # create a list of dictionaries with all the combinations of values for each keyword
        mfargs_L = [dict(zip(kw_L, prod)) for prod in itertools.product(*kw_set_D.values())]
        
        # create master flats

        try:
            for mfargs in mfargs_L:
                mfargs['epoch'] = self
                logger.debug(f"{mfargs=}")
                if self.rawfits.filter(imgtype=IMGTYPES.FLAT, **mfargs).count() > 0:
                    logger.info(f"Building masterflat for {MasterFlat.mfargs2str(mfargs)}.")
                    MasterFlat.create(**mfargs, force_rebuild=force_rebuild)
                else:
                    logger.debug(f"No masterflat will be built for this mfargs since there are no files for it.")
        except Exception as e:
            logger.error(f"Error building masterflats for {self.epochname}: {e}.")
            self.set_flag(Epoch.FLAGS.ERROR)
            
        if self.auto_merge_to_db:
            self.save()



    def reduce(self, force_rebuild=False):
        """ Reduces all (LIGHT) rawfits of this epoch. 
        
        If force_rebuild is False, only rawfits that have not been reduced yet (that do not have 
        the BUILT_REDUCED flag) will be reduced, else all rawfits of this epoch of type LIGHT be 
        reduced.

        If iop4conf.max_concurrent_threads > 1, the reduction will be done in parallel processes.
        """

        Epoch.reduce_rawfits(self.rawfits.filter(imgtype=IMGTYPES.LIGHT), force_rebuild=force_rebuild, epoch=self)
        
        if self.auto_merge_to_db:
            self.save()


    @staticmethod
    def reduce_rawfits(rawfits, force_rebuild=False, epoch=None):
        """ Bulk reduces a list of RawFit in a multiprocessing pool. `rawfits` can be an iterable such as a QuerySet. 
        
        If force_rebuild is False, only rawfits that have not been reduced yet (that do not have 
        the BUILT_REDUCED flag) will be reduced, else all rawfits in the list will be reduced.

        If iop4conf.max_concurrent_threads > 1, the reduction will be done in parallel processes.

        Parameters
        ----------
        rawfits : iterable
            An iterable of RawFit objects to be reduced.

        Other Parameters
        ----------------
        epoch : Epoch, optional
            If provided, it is used only to print the epoch in the log messages of the main thread.
                
        """

        from iop4lib.db import ReducedFit

        reduced_L = [ReducedFit.create(rawfit=rf, auto_build=False, force_rebuild=False) for rf in rawfits]

        if not force_rebuild:
            reduced_L = [redf for redf in reduced_L if not redf.has_flag(ReducedFit.FLAGS.BUILT_REDUCED)]
            
        Epoch.reduce_reducedfits(reduced_L, epoch=epoch)

    
    @staticmethod
    def reduce_reducedfits(reduced_L, epoch=None):
        """ Bulk reduces a list of ReducedFit in a multiprocessing pool. 
        
        If iop4conf.max_concurrent_threads > 1, the reduction will be done in parallel processes.

        Parameters
        ----------
        reduced_L : list
            A list of ReducedFit objects to be reduced. The .build_file() method of each object is always called, 
            independently of the value of the BUILT_REDUCED flag.

        Other Parameters
        ----------------
        epoch : Epoch, optional
            If provided, it is used only to print the epoch in the log messages of the main thread.

        """

        if len(reduced_L) > 0:
            if iop4conf.max_concurrent_threads > 1:
                epoch_bulkreduce_ray(reduced_L)
                #epoch_bulkreduce_multiprocesing(reduced_L, epoch=epoch)
            else:
                epoch_bulkreduce_onebyone(reduced_L, epoch=epoch)
        else:
            logger.info("No files to build.")


    @classmethod
    def by_epochname(cls, epochname):
        telescope, night = cls.epochname_to_tel_night(epochname)
        return cls.objects.filter(telescope=telescope, night=night).get()




    # REDUCTION METHODS



    def compute_relative_photometry(self):
        from .reducedfit import ReducedFit

        redf_qs = ReducedFit.objects.filter(epoch=self, obsmode=OBSMODES.PHOTOMETRY, flags__has=ReducedFit.FLAGS.BUILT_REDUCED).order_by('juliandate').all()

        for redf in redf_qs:
            redf.compute_relative_photometry()



    def make_polarimetry_groups(self):
        """
        To reduce the polarimetry data, we need to group the reduced fits corresponding to rotated polarization angles.

        To group the reduced fits, we make sure they are close (<10min from the julian date field), and that they correspond to the same object, band and exposure time.

        To make sure the correspond to the make object we make use of the OBJECT keyword in the FITS header; in OSN these are the object names (not necessarily the name we use in IOP4 catalog)
        and in CAHA these are the object name + the angle of the polarizer (e.g. "1652+398 0.0 deg"). So we just split the string by ' ' and use the first element.
        TODO: might be better to use the sources_in_field but right now the catalog is pretty incomplete.
        """

        from .reducedfit import ReducedFit

        if self.telescope == TELESCOPES.OSN_T090:
            N_per_cluster = 4
        elif self.telescope == TELESCOPES.CAHA_T220:
            N_per_cluster = 4
        else:
            raise Exception

        redf_qs = ReducedFit.objects.filter(epoch=self, obsmode=OBSMODES.POLARIMETRY, flags__has=ReducedFit.FLAGS.BUILT_REDUCED).order_by('juliandate').all()

        clusters_L = list()
        groupkeys_L = list()

        for i, redf in enumerate(redf_qs):
            
            groupkeys = dict(kwobj=redf.rawfit.header['OBJECT'].split(" ")[0], band=redf.band, exptime=redf.exptime)
            
            if i == 0:
                groupkeys_L.append(groupkeys)
                clusters_L.append([redf])
                continue
            
            if groupkeys == groupkeys_L[-1] and len(clusters_L[-1]) < N_per_cluster and np.abs(redf.juliandate-np.amax([redf_in_clusters.juliandate for redf_in_clusters in clusters_L[-1]])) < 10/60/24: # only if group keys match and it is not more than ~10min than the latest redf in the cluster (max exptime in DB is 300s -> 5min)
                clusters_L[-1].append(redf)
            else:
                groupkeys_L.append(groupkeys)
                clusters_L.append([redf])

        return clusters_L, groupkeys_L
    





    def compute_relative_polarimetry(self):
        from .reducedfit import ReducedFit

        clusters_L, groupkeys_L = self.make_polarimetry_groups()

        logger.info(f"{self}: computing relative polarimetry over {len(groupkeys_L)} polarimetry groups.")
        logger.debug(f"{self}: {groupkeys_L=}")

        if self.telescope == TELESCOPES.OSN_T090:
            return list(map(ReducedFit.compute_relative_polarimetry_osnt090, clusters_L))
        elif self.telescope == TELESCOPES.CAHA_T220:
            return list(map(ReducedFit.compute_relative_polarimetry_caha, clusters_L))
        else:
            raise Exception








# BULK REDUCTION ONE BY ONE

def epoch_bulkreduce_onebyone(reduced_L, epoch=None):
    """ Reduces a list of ReducedFit instances one by one."""
    logger.info(f"{epoch}: building {len(reduced_L)} reduced files one by one. This may take a while.")
    for i, redf in enumerate(reduced_L):
        logger.info(f"{epoch}: building reduced file {i+1}/{len(reduced_L)}: {redf}.")
        redf.build_file()








# BUILD REDUCED FILES IN MULTIPROCESSING POOL

def epoch_bulkreduce_multiprocesing(reduced_L, epoch=None):
    """ Reduces a list of ReducedFit instances in a multiprocessing pool.

    Invokes the reduction of a list of ReducedFit instances in a multiprocessing pool, with a maximum number of
    concurrent processes defined by iop4conf.max_concurrent_threads. It can be invoked with a list of ReducedFit 
    from different epochs.

    Parameters
    ----------
    reduced_L : list of ReducedFit
        List of ReducedFit instances to be reduced.

    Other Parameters
    ----------------
    epoch : Epoch, optional
        If provided, it is used only to print the epoch in the log messages of the main thread.
    """

    queue = multiprocessing.Manager().Queue(-1)  # create queue listener for multiprocessing logging (-1 so no limit on size)
    listener = logging.handlers.QueueListener(queue, *logging.getLogger().handlers)
    listener.start()

    counter = multiprocessing.Value('i', 0)

    logger.info(f"{epoch}: starting {iop4conf.max_concurrent_threads} threads to build {len(reduced_L)} reduced files. Current memory usage: {get_mem_current()/1024**3:.2f} GB.")

    with multiprocessing.get_context('spawn').Pool(processes=iop4conf.max_concurrent_threads, 
                              initializer=_epoch_bulkreduce_multiprocessing_init,
                              initargs=(counter, queue, len(reduced_L), iop4conf), 
                              maxtasksperchild=20) as pool:

        tasks = [pool.apply_async(_epoch_bulkreduce_multiprocessing_worker, args=(redf,)) for redf in reduced_L]    
    
        _epoch_bulkreduce_multiprocesing_mainloop(tasks, counter, reduced_L, epoch=epoch)
        
        pool.close() # no more tasks to be submitted
        pool.join() # wait for all tasks to finish (should already be finished anyway)
        pool.terminate()
    
    listener.stop()

    logger.info(f"{epoch}: all threads finished. Current memory usage: {get_mem_current()/1024**3:.2f} GB.")



def _epoch_bulkreduce_multiprocessing_init(counter, queue, Nredf, iop4conf):
    """ helper func of bulkreduce_multiprocessing to configure the child processes.
    
    Configures the root logger to send messages to a common queue and to allow only messages 
    from iop4lib (and not from the astrometry solver). iopconf has to be passed as an argument, 
    as if method is spawn, the debug level indicated in the cli is not kept.
    
    Makes _counter (shared) and _Nredf global variables so they can be used by the child processes to keep track of the progress.

    Sets up django ORM so it is usable from the child. 
    """

    # counter and total to show progress

    global _counter
    _counter = counter

    global _Nredf
    _Nredf = Nredf

    # set up django ORM to be available

    django.db.connections.close_all()
    django.setup()

    # set up logging to send messages to a common queue

    h = logging.handlers.QueueHandler(queue)    

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(h)
    root.addFilter(logging.Filter("iop4lib"))

    """
    class AstrometryNoLogFilter(logging.Filter):
    def filter(self, record):
        if record.name == "root" and record.levelno == logging.INFO:
            return False
        else:
            return True
        
    root.addFilter(AstrometryNoLogFilter())
    """

    root.setLevel(iop4conf.log_level)


class TimeoutException(Exception):
    pass

def _epoch_bulkreduce_multiprocessing_worker_timeout_handler(signum, frame):
    raise TimeoutException

def _epoch_bulkreduce_multiprocessing_worker(reduced_fit):
    """ helper func to invoke .build() method of ReducedFit on instances from a multiprocessing pool. 

    It needs to be a top-level function, not a method of the class, so it can be pickled and sent to 
    the pool.
    """
    from .reducedfit import ReducedFit

    logger = logging.getLogger(__name__)
    
    logger.debug(f"Epoch {reduced_fit.epoch.id}, child pid {os.getpid()}, ReducedFit {reduced_fit.id} (parent: {get_mem_parent_from_child()/1024**2:.0f} MB, this child: {get_mem_current()/1024**2:.0f} MB, total: {get_total_mem_from_child()/1024**3:.2f} GB total): started building file {_counter.value} / {_Nredf} ({100*_counter.value/_Nredf:.0f}%).")


    try:
        # Start a timer that will send SIGALRM in 20 minutes
        signal.signal(signal.SIGALRM, _epoch_bulkreduce_multiprocessing_worker_timeout_handler)
        signal.alarm(20*60) 
        reduced_fit.build_file()
        signal.alarm(0) # cancel the alarm
    except Exception as e:
        logger.error(f"Epoch {reduced_fit.epoch.id}, child pid {os.getpid()}, ReducedFit {reduced_fit.id}: ERROR. Exception was: {e}")
    else:
        logger.debug(f"Epoch {reduced_fit.epoch.id}, child pid {os.getpid()}, ReducedFit {reduced_fit.id}: finished building file {_counter.value} / {_Nredf} ({100*_counter.value/_Nredf:.0f}%).")

    with _counter.get_lock():
        _counter.value += 1

def _epoch_bulkreduce_multiprocesing_mainloop(tasks, counter, reduced_L, epoch=None):

    time_start = time.time()

    # the following code will print periodically information about the memory usage of the program
    # while tasks are finished. Most of it is commented as so much detail is not needed anymore,
    # feel free to uncomment it if you are having problems with memory usage, or to copy it -with
    # the necessary modifications- to the worker function.

    # # MEMORY USAGE TRACKING

    # # tracemalloc
    # tracemalloc.start()
    # snapshot1 = tracemalloc.take_snapshot()

    # # pympler
    # tr = tracker.SummaryTracker()
    # all_objects = muppy.get_objects()
    # sum1 = summary.summarize(all_objects)

    # output_buff = io.StringIO()

    while not all([task.ready() for task in tasks]):

        time_elapsed = time.time() - time_start
        avg_pace = time_elapsed / counter.value if counter.value > 0 else float('inf')

        # # pympler results
        # sum2 = summary.summarize(muppy.get_objects())
        # with contextlib.redirect_stdout(output_buff):
        #     #tr.print_diff()
        #     summary.print_(summary.get_diff(sum1, sum2))
        # pympler_res_str = output_buff.getvalue()
        # output_buff.seek(0)
        # output_buff.truncate(0)

        # # tracemalloc results
        # snapshot2 = tracemalloc.take_snapshot()
        # top_stats = snapshot2.statistics('lineno')
        # snapshot1 = snapshot2

        try:
            childs = [{'pid':child.pid, 'mem':child.memory_info().rss} for child in psutil.Process().children(recursive=True)]
        except psutil.NoSuchProcess:
            # one of the child processes might have dissapeared or restarted in the meantime (because of maxtasksperchild or 
            # because the tasks are close to being finished), so we just ignore it.
            pass
        else:
            logger.info(
                                "\n" + "*"*40 + "\n" +
                                "\n" +
                                (f"[ {epoch} - Main Thread (pid {os.getpid()}) ]\n" if epoch is not None else f"[ Main Thread (pid {os.getpid()}) ]\n") +
                                "\n" +
                                f"Progress: {counter.value} of {len(reduced_L)} files processed in {time_elapsed:.0f} seconds ({avg_pace:.1f} s / file).\n" 
                                "\n" +
                                f"Parent memory: {get_mem_current()/1024**2:5.0f} MB\n" +
                                f"Children memory: {get_mem_children()/1024**3:.2f} GB\n" +
                                "\n".join([f"  - pid {child['pid']:5} : {child['mem']/1024**2:5.0f} MB" for child in childs]) + "\n" +
                                "\n" + 
                                # "*"*40 + "\n" +
                                # "\n" +
                                # f"[ Memory statistics per line - Main Thread (pid {os.getpid()})]\n" +
                                # "\n" +
                                # "\n".join(map(str, top_stats[:20])) + "\n" +
                                # "\n" +
                                # "*"*40 + "\n" +
                                # "\n" +
                                # f"[ Memory statistics per object type - Main Thread (pid {os.getpid()})]\n" + pympler_res_str +
                                # "\n" + 
                                "*"*40
                            )

        time.sleep(5)

    # When task have finished, return
    return








# BULK REDUCTION IN RAY CLUSTER


def epoch_bulkreduce_ray(reduced_L):
    """Bulk reduction of a list of ReducedFits in a Ray cluster."""
    
    import os
    import ray
    from ray.util.multiprocessing import Pool
    import socket
    from dataclasses import dataclass
    from iop4lib.db import ReducedFit

    logger.info(f"Starting bulk reduction of {len(reduced_L)} ReducedFits in Ray cluster.")

    logger.info("Syncing raw files and DB to Ray cluster.")
    os.system(f"ray rsync-up priv.rayconfig.yaml")
    os.system(fr"rsync -v ~/iop4data/iop4.db {iop4conf.ray_cluster_address}:'~/iop4data/iop4.db'")
    os.system(fr"rsync -va --update ~/iop4data/raw/ {iop4conf.ray_cluster_address}:'~/iop4data/raw/'")
    os.system(fr"rsync -va --update --delete ~/iop4data/masterflat/ {iop4conf.ray_cluster_address}:'~/iop4data/masterflat/'")
    os.system(fr"rsync -va --update --delete ~/iop4data/masterbias/ {iop4conf.ray_cluster_address}:'~/iop4data/masterbias/'")

    logger.info("Connecting to Ray cluster at localhost:25000. Remember to attach to the cluster with 'ray attach priv.rayconfig.yaml -p 25000' and start the head node with 'ray stop' and 'ray start --head --ray-client-server-port 25000 --num-cpus=128'. Additionaly worker nodes can be started with 'ray start --address:head_address:port', port is usually 6379.")
    ray.init("ray://localhost:25000", ignore_reinit_error=True)

    def _init_func():
        import iop4lib
        iop4lib.Config(config_db=True, reconfigure=True)
        import django
        django.db.connections.close_all()
        print(f"Node {socket.gethostname()} initialized.")

    def _buildfile(redf_id):
        """Builds a ReducedFit file in the Ray cluster. 
        
        It syncs the raw files, masterbias and master flat from local, builds the file and performs astrometric calibration remotely,
        and gets the header and summary images back.
        """

        from iop4lib.db import ReducedFit

        print(f"Node {socket.gethostname()} starting to build ReducedFit {redf_id}.")

        try:
            # simplified version of ReducedFit.build_file(), avoids writing to the DB
            redf = ReducedFit.objects.get(id=redf_id)
            redf.apply_masterbias()
            redf.apply_masterflat()
            redf.astrometric_calibration()
        except Exception as e:
            logger.error(f"ReducedFit {redf.id} in Ray cluster -> exception during build_file(): {e}")
            return redf_id, False, None, None
        else:
            #return [redf_id, True, None, None]
        
            res =  [redf_id, True, redf.header, dict()]

            print(f"Node {socket.gethostname()} finished ReducedFit {redf_id}.")

            try:
                for fpath in glob.glob(f"{redf.filedpropdir}/astrometry_*"):
                    with open(fpath, "rb") as f:
                        res[3][os.path.basename(fpath)] = f.read()
            except Exception as e:
                logger.error(f"ReducedFit {redf.id} in Ray cluster -> exception during reading of astrometry plots: {e}")
                return redf_id, False, None, None

            return tuple(res)
           
    
    time_start = time.time()

    with Pool(initializer=_init_func) as pool:

        for i, res in enumerate(pool.imap_unordered(_buildfile, [redf.id for redf in reduced_L], chunksize=4)):

            redf_id, success, header, files = res

            time_elapsed = time.time() - time_start
            avg_pace = time_elapsed / i if i > 0 else float('inf')

            if success:
                logger.info(f"ReducedFit {redf_id} was successfully calibrated astrometrically by the RAY cluster.\nProgress: {i} of {len(reduced_L)} files processed in {time_elapsed:.0f} seconds ({avg_pace:.1f} s / file).")
            else:
                logger.error(f"ReducedFit {redf_id} could not be calibrated astrometrically by the RAY cluster.\nProgress: {i} of {len(reduced_L)} files processed in {time_elapsed:.0f} seconds ({avg_pace:.1f} s / file).")

            redf = ReducedFit.objects.get(id=redf_id)
        
            if not redf.fileexists:
                try:
                    redf.apply_masterbias()
                    redf.apply_masterflat()
                except Exception as e:
                    logger.error(f"{redf}: exception during .apply_masterbias(), .apply_masterflat(): {e}")
                    pass
            
            if success:

                with fits.open(redf.filepath, 'update') as hdul:
                    hdul[0].header.update(header)

                for filename, data in files.items():
                    logger.debug(f"Writting {filename} to {redf.filedpropdir}.")
                    with open(Path(redf.filedpropdir) / filename, "wb") as f:
                        f.write(data)

                redf.sources_in_field.set(AstroSource.get_sources_in_field(fit=redf))

                redf.unset_flag(ReducedFit.FLAGS.ERROR_ASTROMETRY)
                redf.set_flag(ReducedFit.FLAGS.BUILT_REDUCED)
                
            else:
                redf.set_flag(ReducedFit.FLAGS.ERROR_ASTROMETRY)
                redf.unset_flag(ReducedFit.FLAGS.BUILT_REDUCED)

            redf.save()

        pool.close() 
        pool.join()
        pool.terminate()


    logger.warning("Not syncing files from Ray cluster! (anyway, all results should have come back already)")
    #os.system(f"rsync -va {iop4conf.ray_cluster_address}:'~/iop4data/calibration' ~/iop4data/calibration")

    n_in_error = ReducedFit.objects.filter(flags__has=ReducedFit.FLAGS.ERROR_ASTROMETRY).count()
    duration_hms_str = time.strftime('%H h %M min %S s', time.gmtime(time.time()-time_start))
    logger.info(f"Finished bulk reduction of {len(reduced_L)} ReducedFits in Ray cluster. {n_in_error} files could not be calibrated astrometrically. Took {duration_hms_str}.")

