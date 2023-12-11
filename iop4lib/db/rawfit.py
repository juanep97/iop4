# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# django imports
from django.db import models

# iop4lib imports
from ..enums import *
from iop4lib.telescopes import Telescope
from iop4lib.instruments import Instrument
from .fitfilemodel import FitFileModel
from .fields import FlagChoices, FlagBitField

# other imports
import re
import os
import stat
import datetime
import numpy as np

# logging 
import logging
logger = logging.getLogger(__name__)



class RawFit(FitFileModel):

    # Database fields and information

    ## Identifiers
    
    epoch = models.ForeignKey('Epoch', on_delete=models.CASCADE, related_name="%(class)ss") # rawfits
    filename = models.CharField(max_length=255)

    # flags

    class FLAGS(FlagChoices):
        ERROR = 1 << 1
        ERROR_DOWNLOAD = 1 << 2
        DOWNLOADED = 1 << 3
        ERROR_CLASSIFY = 1 << 4
        CLASSIFIED = 1 << 5
        ERROR_ASTROMETRY = 1 << 6
        BUILT_REDUCED = 1 << 7
        
    flags = FlagBitField(choices=FLAGS.choices(), default=0)

    ## FITS related fields

    instrument = models.CharField(max_length=20, choices=INSTRUMENTS.choices, default=INSTRUMENTS.NONE, help_text="Instrument used for the observation.")
    obsmode = models.CharField(max_length=20, choices=OBSMODES.choices, default=OBSMODES.NONE, help_text="Whether the observation was photometry or polarimetry.")
    juliandate = models.FloatField(null=True, help_text="Julian date of observation, from the date and time indicated in the FITS header.")
    imgsize = models.CharField(max_length=12, null=True, help_text="String with the format _width_x_height_ for the image, as in 1024x1024.") # as in 1024x1024, 2048x2048, etc
    imgtype = models.CharField(max_length=20, choices=IMGTYPES.choices, default=IMGTYPES.NONE, help_text="Whether the raw image is a bias (BIAS), flat (FLAT), or science (LIGHT).")
    band = models.CharField(max_length=10, choices=BANDS.choices, default=BANDS.NONE, help_text="Band of the observation, as in the filter used (R, V, etc).")
    rotangle = models.FloatField(null=True, help_text="Rotation angle of the polarizer in degrees.")
    exptime = models.FloatField(null=True, help_text="Exposure time in seconds.")

    # DB information and constraints

    class Meta:
        app_label = 'iop4api'
        verbose_name = "RawFit"
        verbose_name_plural = "Raw FITS files"
        constraints = [
            models.UniqueConstraint(fields=['epoch', 'filename'], name='epoch_filename_unique')
        ]

    # Properties 

    @property
    def filepath(self):
        """ Returns the full path to the FITS file.
        The raw FITS file are stored in the rawfitsdir of the epoch.
        """
        return os.path.join(self.epoch.rawfitsdir, self.filename)
    
    @property
    def filedpropdir(self):
       """ Returns the full path to the directory where related files are stored.
       The directory is in the calibrationdir of the epoch, different from the rawfitsfit,
       to keep a clean archive of raw data (see the `Epoch` class). Reduced FITS files will be
       stored here also, see the `ReducedFit` class.
       """
       return os.path.join(self.epoch.calibrationdir, 
                            self.filename +  '.d', 
                            self.__class__.__name__)
    
    @property
    def fileloc(self):
        """ Returns an human readable string which uniquely identifies the file, as an alternative to the id."""
        return f"{self.epoch.epochname}/{self.filename}"
    
    @property
    def telescope(self):
        return self.epoch.telescope
    
    @property
    def night(self):
        return self.epoch.night


    # repr and str

    def __repr__(self):
        return f"{self.__class__.__name__}.objects.get(id={self.id!r})"
    
    def __str__(self):
        return f"<{self.__class__.__name__} {self.id!r} | {self.fileloc}>"

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text(f"{self!r}")
        else:
            with p.group(4, f'<{self.__class__.__name__}(', ')>'):
                p.text(f"id: {self.id},")
                p.breakable()
                p.text(f"telescope: {self.epoch.telescope},")
                p.breakable()
                p.text(f"night: {self.epoch.night},")
                p.breakable()
                p.text(f"filename: {self.filename},")
                p.breakable()
                p.text(f"instrument: {self.instrument},")
                p.breakable()
                p.text(f"imgtype: {self.imgtype},")
                p.breakable()
                p.text(f"size: {self.imgsize},")
                p.breakable()
                p.text(f"obsmode: {self.obsmode},")
                p.breakable()
                p.text(f"rotangle: {self.rotangle},")
                p.breakable()
                p.text(f"band: {self.band},")
                p.breakable()
                p.text(f"exptime: {self.exptime},")
                p.breakable()
                p.text(f"flags: {', '.join(self.flag_labels)}")

    def _repr_html_(self):
        return (f"{self.__class__.__name__}(id={self.id!r}):<br>\n"
                f" - telescope: {self.epoch.telescope}<br>\n"
                f" - night: {self.epoch.night}<br>\n"
                f" - filename: {self.filename}<br>\n"
                f" - instrument: {self.instrument}<br>\n"
                f" - imgtype: {self.imgtype}<br>\n"
                f" - size: {self.imgsize}<br>\n"
                f" - obsmode: {self.obsmode}<br>\n"
                f" - band: {self.band}<br>\n"
                f" - exptime: {self.exptime}<br>\n"
                f" - flags: {(','.join(self.flag_labels))}<br>\n")     
    
    # Constructor

    @staticmethod 
    def fileloc_to_tel_night_filename(fileloc):
        """Parses a fileloc to telescope, night and filename."""
        from .epoch import Epoch

        matches = re.findall(r"(([a-zA-Z0-9]+)/([0-9]{2,4}-?[0-9]{2}-?[0-9]{2}))/([^/\\]+)$", fileloc)

        if len(matches) != 1:
            raise Exception(f"fileloc {fileloc} is not EPOCHNAME/filename")
        
        epochname = matches[0][0]

        telescope, night = Epoch.epochname_to_tel_night(epochname)
        
        filename = matches[0][-1]

        return telescope, night, filename
      
    @classmethod
    def create(cls, 
                 epoch=None, filename=None, fileloc=None,
                 auto_procure_local=True,
                 reset_flags=False, 
                 auto_classify=True,
                 force_redownload=False,
                 auto_merge_to_db=True):
        """
            Create a RawFit object from fileloc or epoch and filename.

            Flags are cleared when .create() is called.

            Parameters
            ----------
            fileloc : str
                File location in the format TELESCOPE/YYYYMMDD/FILENAME.
            epoch : Epoch
                Epoch object.
            filename : str
                Filename.
            auto_procure_local : bool
                Whether automatically procure the local file (linking it to a local file).
            auto_classify : bool
                If auto_procure_local is True, whether to automatically classify the file.
            force_redownload : bool
                If auto_procure_local is True, whether to force redownload of the file even if it already exists.
            auto_merge_to_db : bool
                Whether automatically merge the object to the DB when making any changes, including creation.
        """

        from iop4lib.db import Epoch

        if epoch is not None and filename is not None:
            fileloc = os.path.join(epoch.epochname, filename)
        elif fileloc is not None: # get epoch and filename from fileloc

            telescope, night, filename = RawFit.fileloc_to_tel_night_filename(fileloc)

            if (epoch := Epoch.objects.filter(telescope=telescope, night=night).first()) is  None:
                logger.warning("No epoch in DB; it will be created.")
                epoch = Epoch.create(telescope=telescope, night=night, auto_merge_to_db=auto_merge_to_db, auto_link_rawfits=auto_procure_local)
            
            logger.info(f"Setting epoch to {epoch}.")
        else:
            raise Exception("Either epoch and filename or fileloc must be provided.")

        # create entry

        if (rawfit := RawFit.objects.filter(epoch=epoch, filename=filename).first()) is None:
            logger.debug(f"Creating DB entry for {fileloc}.")
            rawfit = cls(epoch=epoch, filename=filename)
            rawfit.save()
        else:
            logger.debug(f"DB entry for {fileloc} already exists, it will be used instead.")

        # intance only attributes
        rawfit.auto_merge_to_db = auto_merge_to_db
        rawfit.auto_classify = auto_classify
        rawfit.auto_procure_local = auto_procure_local
        rawfit.reset_flags = reset_flags
        rawfit.force_redownload = force_redownload

        # link to local file
        if auto_procure_local:
            rawfit.procure_local_file()

        # merge to db
        if auto_merge_to_db:
            rawfit.save()

        return rawfit
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.auto_procure_local=True
        self.auto_classify=True
        self.reset_flags=False
        self.force_redownload=False
        self.auto_merge_to_db=True
    
    # Methods

    def download(self):
        if not os.path.isdir(self.epoch.rawfitsdir):
            logger.debug("Creating rawfitsdir")
            os.makedirs(self.epoch.rawfitsdir)
        
        Telescope.by_name(self.epoch.telescope).download_rawfits([self])
               
    def procure_local_file(self):

        logger.debug(f"{self}: procuring local file.")

        if self.fileexists and self.reset_flags:
            self.clear_flags()

        if not self.fileexists or self.force_redownload:
            try:
                logger.debug(f"Downloading file {self.fileloc}.")
                self.download()
                self.set_flag(RawFit.FLAGS.DOWNLOADED)
            except Exception as e:
                logger.error(f"Error downloading file {self.filename}, skipping download: {e}.")
                self.set_flag(RawFit.FLAGS.ERROR_DOWNLOAD)
        else:
            self.set_flag(RawFit.FLAGS.DOWNLOADED)

        if self.fileexists:
            if iop4conf.set_rawdata_readonly:
                # this will remove the write permission from the file for all users
                current_perms = os.stat(self.filepath).st_mode
                new_perms = current_perms & (~stat.S_IWUSR) & (~stat.S_IWGRP) & (~stat.S_IWOTH)
                os.chmod(self.filepath, new_perms)

            if self.auto_classify:
                self.classify()

        if self.auto_merge_to_db:
            self.save()
        
    def classify(self):
        """
            Get some parameters from the fit file to the DB entry.
        """
        logger.debug(f"{self}: classifying")

        try:
            self.unset_flag(RawFit.FLAGS.ERROR_CLASSIFY)
            self.unset_flag(RawFit.FLAGS.CLASSIFIED)
            Telescope.by_name(self.epoch.telescope).classify_rawfit(self)
            self.set_flag(RawFit.FLAGS.CLASSIFIED)
        except Exception as e:
            logger.error(f"Error classifying {self.fileloc}: {e}.")
            self.set_flag(RawFit.FLAGS.ERROR_CLASSIFY)

        if self.auto_merge_to_db:
            self.save()

    def request_master(self, model, other_epochs=False):
        return Instrument.by_name(self.instrument).request_master(self, model, other_epochs=other_epochs)

    def request_masterbias(self, *args, **kwargs):
        from iop4lib.db import MasterBias
        return self.request_master(MasterBias, *args, **kwargs)
    
    def request_masterflat(self, *args, **kwargs):
        from iop4lib.db import MasterFlat
        return self.request_master(MasterFlat, *args, **kwargs)
    
    def request_masterdark(self, *args, **kwargs):
        from iop4lib.db import MasterDark
        return self.request_master(MasterDark, *args, **kwargs)

    @property
    def header_hintcoord(self):
        """ Returns a SkyCoord according to the headers of the FITS file."""
        return Instrument.by_name(self.instrument).get_header_hintcoord(self)

    @property
    def header_hintobject(self):
        """ Returns the AstroSource according to the OBJECT keyword in the header of the FITS file. """
        return Instrument.by_name(self.instrument).get_header_hintobject(self)
    
    # Class methods    

    @classmethod
    def by_fileloc(cls, fileloc):
        telescope, night, filename = cls.fileloc_to_tel_night_filename(fileloc)
        return cls.objects.filter(epoch__telescope=telescope, epoch__night=night, filename=filename).get()