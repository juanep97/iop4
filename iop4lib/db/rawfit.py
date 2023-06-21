import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

from django.db import models

# other imports

import re
import os
import stat
import datetime

import numpy as np

from ..enums import *
from iop4lib.telescopes import Telescope
from .fitfilemodel import FitFileModel
from .fields import FlagChoices, FlagBitField

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
        telescope, night = Epoch.epochname_to_tel_night(matches[0][0])
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

    @classmethod 
    def from_db(cls, db, *args, **kwargs):
        instance = super(RawFit, cls).from_db(db, *args, **kwargs)
        instance.auto_procure_local=True,
        instance.auto_classify=True,
        instance.reset_flags=False, 
        instance.force_redownload=False
        instance.auto_merge_to_db=True,
        return instance
    
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
                os.chmod(self.filepath, stat.S_IREAD)

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
            self.unset_flag(RawFit.FLAGS.CLASSIFIED)
            Telescope.by_name(self.epoch.telescope).check_telescop_kw(self)
            Telescope.by_name(self.epoch.telescope).classify_instrument_kw(self)
            Telescope.by_name(self.epoch.telescope).classify_juliandate_rawfit(self)
            Telescope.by_name(self.epoch.telescope).classify_imgtype_rawfit(self)
            Telescope.by_name(self.epoch.telescope).classify_band_rawfit(self)
            Telescope.by_name(self.epoch.telescope).classify_obsmode_rawfit(self)
            Telescope.by_name(self.epoch.telescope).classify_imgsize(self)
            Telescope.by_name(self.epoch.telescope).classify_exptime(self)
            self.set_flag(RawFit.FLAGS.CLASSIFIED)
        except Exception as e:
            logger.error(f"Error classifying {self.fileloc}: {e}.")
            self.set_flag(RawFit.FLAGS.ERROR_CLASSIFY)

        if self.auto_merge_to_db:
            self.save()



    def request_masterbias(self, other_epochs=False):
        """ Returns the master bias for this raw fit.

        Notes
        -----
        See also iop4lib.db.MasterBias.request_masterbias().
        """
        from iop4lib.db import MasterBias

        rf_vals = RawFit.objects.filter(id=self.id).values().get()
        args = {k:rf_vals[k] for k in rf_vals if k in MasterBias.mbargs_kwL}
        args["epoch"] = self.epoch # from values we only get epoch__id 

        mb = MasterBias.objects.filter(**args).first()
        
        if mb is None and other_epochs == True:
            args.pop("epoch")

            mb_other_epochs = np.array(MasterBias.objects.filter(**args).all())

            if len(mb_other_epochs) == 0:
                logger.debug(f"No master bias for {args} in DB, None will be returned.")
                return None

            mb_other_epochs_jyear = np.array([mb.epoch.jyear for mb in mb_other_epochs])
            mb = mb_other_epochs[np.argsort(np.abs(mb_other_epochs_jyear - self.epoch.jyear))[0]]

            if (mb.epoch.jyear - self.epoch.jyear) > 7/365:
                #logger.debug(f"Master bias from epoch {mb.epoch} is more than 1 week away from epoch {self.epoch}, None will be returned.")
                #return None
                logger.warning(f"Master bias from epoch {mb.epoch} is more than 1 week away from epoch {self.epoch}.")
            
        return mb



    def request_masterflat(self, other_epochs=False):
        """ Searchs in the DB and returns an appropiate masterflat for this rawfit. 
        
        Notes
        -----
        It takes into account the parameters (band, size, etc) defined in MaserFlat.mfargs_kwL; except 
        for exptime, which is not taken into account (flats with different extime can and must be used). 
        By default, it looks for masterflats in the same epoch, but if other_epochs is set to True, it
        will look for masterflats in other epochs. If more than one masterflat is found, it returns the
        one from the closest night. It will print a warning even with other_epochs if it is more than 1
        week away from the rawfit epoch.
        
        If no masterflat is found, it returns None.
        """

        from iop4lib.db import MasterFlat

        rf_vals = RawFit.objects.filter(id=self.id).values().get()
        args = {k:rf_vals[k] for k in rf_vals if k in MasterFlat.mfargs_kwL}
        
        args.pop("exptime", None)
        args["epoch"] = self.epoch # from values we only get epoch__id 

        mf = MasterFlat.objects.filter(**args).first()
        
        if mf is None and other_epochs == True:
            args.pop("epoch")

            mf_other_epochs = np.array(MasterFlat.objects.filter(**args).all())

            if len(mf_other_epochs) == 0:
                logger.debug(f"No master flat for {args} in DB, None will be returned.")
                return None
            
            mf_other_epochs_jyear = np.array([mf.epoch.jyear for mf in mf_other_epochs])
            mf = mf_other_epochs[np.argsort(np.abs(mf_other_epochs_jyear - self.epoch.jyear))[0]]
            
            if (mf.epoch.jyear - self.epoch.jyear) > 7/365:
                #logger.debug(f"Master flat from epoch {mf.epoch} is more than 1 week away from epoch {self.epoch}, None will be returned.")
                #return None
                logger.warning(f"Master flat from epoch {mf.epoch} is more than 1 week away from epoch {self.epoch}.")
                        
        return mf

    @property
    def header_hintcoord(self):
        """ Returns a SkyCoord according to the headers of the FITS file."""
        return Telescope.by_name(self.epoch.telescope).get_header_hintcoord(self)

    @property
    def header_objecthint(self):
        """ Returns the AstroSource according to the OBJECT keyword in the header of the FITS file. """
        return Telescope.by_name(self.epoch.telescope).get_header_objecthint(self)
    
    # Class methods    

    @classmethod
    def by_fileloc(cls, fileloc):
        telescope, night, filename = cls.fileloc_to_tel_night_filename(fileloc)
        return cls.objects.filter(epoch__telescope=telescope, epoch__night=night, filename=filename).get()