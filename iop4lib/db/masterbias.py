import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

from django.db import models

import os
import datetime
import numpy as np
import astropy.io.fits as fits
import datetime

from ..enums import *
from .fitfilemodel import FitFileModel
from .fields import FlagChoices, FlagBitField

import logging
logger = logging.getLogger(__name__)

class MasterBias(FitFileModel):
    """A class representing a master bias for an epoch."""

    margs_kwL = ['epoch', 'instrument', 'imgsize']

    imgtype = IMGTYPES.BIAS

    # Database fields
    rawfits = models.ManyToManyField('RawFit', related_name='built_masterbias')

    # flags

    class FLAGS(FlagChoices):
        IGNORE = 1 << 1
        
    flags = FlagBitField(choices=FLAGS.choices(), default=0)

    # fields corresponding to MasterBias kw arguments (margs_kwL)
    
    epoch = models.ForeignKey('Epoch', on_delete=models.CASCADE, related_name='masterbias') 
    instrument = models.CharField(max_length=20, choices=INSTRUMENTS.choices)
    imgsize = models.CharField(max_length=12, null=True)

    class Meta:
        app_label = 'iop4api'
        verbose_name = "Master Bias"
        verbose_name_plural = "Master Biases"

    # Properties

    @property
    def filename(self):
        return f"masterbias_id{self.id}.fits"
    
    @property
    def filepath(self):
        return os.path.join(self.epoch.masterbiasdir, self.filename)

    @property
    def fileloc(self):
        return f"{self.epoch.epochname}/{self.filename}"
    
    # Filed Properties location

    @property
    def filedpropdir(self):
       return os.path.join(self.epoch.calibrationdir, 
                            self.filename +  '.d', 
                            self.__class__.__name__)
        
    @property
    def margs(self):
        """Return a dict of the arguments used to build this MasterFlat.
        """

        return {'epoch':self.epoch, 'instrument':self.instrument, 'imgsize':self.imgsize}
    
    # repr and str

    @classmethod
    def margs2str(cls, margs):
       """Class method to build a nice string rep of the arguments of a MasterFlat.
       """

       return f"{margs['epoch'].epochname} | {margs['instrument']} | {margs['imgsize']}"

    def __repr__(self):
        return f"MasterBias.objects.get(id={self.id!r})"
    
    def __str__(self):
        return f"<MasterBias {self.id!r} | {MasterBias.margs2str(self.margs)}>"
    
    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text(f"{self!r}")
        else:
            with p.group(4, f'<{self.__class__.__name__}(', ')>'):
                p.text(f"id: {self.id},")
                for k,v in self.margs.items():
                    p.breakable()
                    p.text(f"{k}: {v}")
                

    
    # Constructor

    @classmethod
    def create(cls,
               rawfits=None,
               auto_merge_to_db=True,
               force_rebuild=False,
               **kwargs):
        """Create a MasterBias object for the given epoch, reusing an existing DB entry if it exists.

        It will overwrite any existing MasterBias for that epoch, or create a new one if it doesn't exist.
    
        Parameters
        ----------
        rawfits : list of RawFit, optional
            List of RawFit objects used to build the MasterBias.
        auto_merge_to_db : bool, optional
            If True, automatically merge the created MasterBias object to the database.
        force_rebuild : bool, optional
            If True, force rebuilding the MasterBias even if it already exists.

        Other Parameters
        ----------------
        `**kwargs` :
            Keyword arguments passed to the constructor.

        """

        margs = {k:kwargs[k] for k in MasterBias.margs_kwL if k in kwargs}

        from iop4lib.db import RawFit

        if (mb := MasterBias.objects.filter(**margs).first()) is not None:
            logger.debug(f"DB entry for {mb} already exists, using it instead.")
        else:
            logger.debug(f"A DB entry for MasterBias {MasterBias.margs2str(margs)} will be created.")
            mb = cls(**margs)
            mb.save()

        if rawfits is None:
            rawfits = RawFit.objects.filter(imgtype=IMGTYPES.BIAS, **margs)
            logger.debug(f"Found {len(rawfits)} bias raw files for {mb}.")
        else:
            logger.debug(f"Using {len(rawfits)} bias raw files for {mb}.")

        mb.rawfits.set(rawfits)

        # Build the file
        if not mb.fileexists or force_rebuild:
            logger.info(f"Building file")
            try:
                mb.build_file()
            except Exception as e:
                logger.error("An error occurred while building the masterbias, deleting it and raising Exception")
                mb.delete()
                raise Exception(f"An error occurred while building the MasterBias for {margs}: {e}")
            else:
                logger.info("Built masterbias successfully")

        # merge to DB

        if auto_merge_to_db:
            mb.save()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.auto_merge_to_db = True
    
    
    # Methods

    def build_file(self):

        logger.debug(f"Getting data from files")

        if self.rawfits.count() == 0:
            logger.error(f"No rawfits for {self}")
            raise Exception(f"No rawfits for {self}")
        
        data_L = []
        for rf in self.rawfits.all():
            with fits.open(rf.filepath) as hdul:
                data_L.append(hdul[0].data)

        logger.debug(f"Computing median")

        data = np.nanmedian(data_L, axis=0)

        logger.debug(f"Building header")

        header = fits.Header()
        header['TELESCOP'] = self.epoch.telescope
        header['NIGHT'] = self.epoch.yyyymmdd
        header['EPOCH'] = self.epoch.epochname
        header['IMGSIZE'] = self.imgsize
        header['IMGTYPE'] = 'masterbias'
        header['DATECREA'] = datetime.datetime.utcnow().isoformat(timespec="milliseconds")
        header['NRAWFITS'] = self.rawfits.count()

        logger.debug(f"Building HDU")

        if not os.path.isdir(self.epoch.masterbiasdir):
            logger.debug(f"Creating directory {self.epoch.masterbiasdir}")
            os.makedirs(self.epoch.masterbiasdir)

        hdu = fits.PrimaryHDU(data, header=header)

        logger.debug(f"Writing MasterBias to {self.filepath}")
        hdu.writeto(self.filepath, overwrite=True)