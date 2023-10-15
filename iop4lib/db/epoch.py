import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

import django
from django.db import models
from django.db.models import Count, Q
from django.db.models.functions import Concat

# other imports
from collections.abc import Sequence
import re
import os
import datetime
import time
import numpy as np
import itertools
from astropy.coordinates import SkyCoord
from pathlib import Path
import astropy.io.fits as fits
import glob
from astropy.time import Time

# iop4 imports

from iop4lib.enums import *
from iop4lib.telescopes import Telescope
from .astrosource import AstroSource
from .fields import FlagChoices, FlagBitField
from iop4lib.utils import  get_mem_parent_from_child, get_total_mem_from_child, get_mem_current, get_mem_children
from iop4lib.utils.parallel import epoch_bulkreduce_multiprocesing, epoch_bulkreduce_ray

# logging

import logging
logger = logging.getLogger(__name__)


    
class Epoch(models.Model):
    """A class representing an epoch.
    
    Identified by the telescope and date of the night. Provides method for fetching the data
    from the telescope archives and reducing the data.
    """

    objects = models.Manager()

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

    def __init__(self, *args, **kwargs):
        """Provides some defaults for attributes that are not stores in DB.

        Should be the same as default kwargs in .create() for consistency.
        """
        super().__init__(*args, **kwargs)
        self.auto_link_rawfits = True
        self.check_remote_list = False
        self.force_redownload = False
        self.fallback_to_local = True
        self.auto_merge_to_db = True
        self.rawfits_kwargs = dict()
      
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

        else:
            logger.error("Epoch not listed, not linking rawfits.")

        if self.auto_merge_to_db:
            self.save()



    def get_summary_rawfits_status(self):
        """
            Returns list of flags present in the rawfits of this epoch.
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
            if iop4conf.ray_use_cluster:
                epoch_bulkreduce_ray(reduced_L)
            elif iop4conf.max_concurrent_threads > 1:
                epoch_bulkreduce_multiprocesing(reduced_L, epoch=epoch)
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

        logger.debug(f"{self}: grouping observations for polarimetry...")

        from .reducedfit import ReducedFit

        redf_qs = ReducedFit.objects.filter(epoch=self, obsmode=OBSMODES.POLARIMETRY, flags__has=ReducedFit.FLAGS.BUILT_REDUCED).order_by('juliandate').all()

        # Create a groups with the same keys (object, band, exptime)

        groups_D = dict()

        for i, redf in enumerate(redf_qs):
            
            keys = (
                    ('kwobj', redf.rawfit.header['OBJECT'].split(" ")[0]), 
                    ('instument', redf.instrument),
                    ('band', redf.band), 
                    ('exptime', redf.exptime)
            )
            
            if keys not in groups_D.keys():
                groups_D[keys] = list()
                
            groups_D[keys].append(redf)

        # output some debug info about the groups made
        for key_T, redf_L in groups_D.items():
            key_D = dict(key_T)
            logger.debug(f"{len(redf_L)=}; {key_D.values()}, {set([redf.rotangle for redf in redf_L])}")
    
        # split the groups into subgroups such that every subgroup has at most 4 elements and all rotangles are present in the subgroup

        split_groups_keys = list()
        split_groups = list()

        for key_T, redf_L in groups_D.items():
            key_D = dict(key_T)

            rotangles_S = set([redf.rotangle for redf in redf_L]) # rotangles available in the redf_L
            
            split_rotangle_D = {rotangle:[redf for redf in redf_L if redf.rotangle==rotangle] for rotangle in rotangles_S}  # to access the redfs in the redfL by rotangle
            
            while any([len(split_rotangle_D[rotangle])>0 for rotangle in rotangles_S]): # while there are redfs for some rotangle, create groups by popping one of each rotangle
                split_groups.append([split_rotangle_D[rotangle].pop() for rotangle in rotangles_S if len(split_rotangle_D[rotangle]) > 0])
                split_groups_keys.append(key_D)

        # sort the groups by min(juliandate)
         
        t1_L = [min([redf.juliandate for redf in redf_L]) for redf_L in split_groups]

        split_groups_keys = [x[1] for x in sorted(zip(t1_L, split_groups_keys))]
        split_groups = [x[1] for x in sorted(zip(t1_L, split_groups))]

        # some debug info about the final sorted groups:

        if iop4conf.log_level == logging.DEBUG:
            for key_D, redf_L in zip(split_groups_keys, split_groups):
                
                t1 = Time(min([redf.juliandate for redf in redf_L]), format="jd").datetime.strftime("%H:%M:%S")
                t2 = Time(max([redf.juliandate for redf in redf_L]), format="jd").datetime.strftime("%H:%M:%S")
                
                print(f"{len(redf_L)=}; {key_D.values()}, {set([redf.rotangle for redf in redf_L])}   ({t1}, {t2})")
                
                for redf in redf_L:
                    print(f"     -> {redf.rotangle}: {Time(redf.juliandate, format='jd').datetime.strftime('%H:%M:%S')}")

        # return the groups and their keys:

        return split_groups, split_groups_keys


    def compute_relative_polarimetry(self, *args, **kwargs):
        from .reducedfit import ReducedFit

        clusters_L, groupkeys_L = self.make_polarimetry_groups()

        logger.info(f"{self}: computing relative polarimetry over {len(groupkeys_L)} polarimetry groups.")
        logger.debug(f"{self}: {groupkeys_L=}")

        f = lambda x: Telescope.by_name(self.telescope).compute_relative_polarimetry(x, *args, **kwargs)
        
        return list(map(f, clusters_L))







# BULK REDUCTION ONE BY ONE

def epoch_bulkreduce_onebyone(reduced_L: Sequence['ReducedFit'], epoch: Epoch = None) -> None:
    """ Reduces a list of ReducedFit instances one by one."""
    logger.info(f"{epoch}: building {len(reduced_L)} reduced files one by one. This may take a while.")
    for i, redf in enumerate(reduced_L):
        logger.info(f"{epoch}: building reduced file {i+1}/{len(reduced_L)}: {redf}.")
        redf.build_file()







