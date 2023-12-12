import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

from django.db import models

import os
import datetime
import numpy as np
import astropy.io.fits as fits

from .fitfilemodel import FitFileModel
from ..enums import *
from .fields import FlagChoices, FlagBitField

import logging
logger = logging.getLogger(__name__)

class MasterFlat(FitFileModel):
    """
    A class representing a master flat for an epoch.
    """

    margs_kwL = ['epoch', 'instrument', 'imgsize', 'band', 'obsmode', 'rotangle', 'exptime']

    imgtype = IMGTYPES.FLAT
    
    # Database fields
    rawfits = models.ManyToManyField('RawFit', related_name='built_masterflats')

    # flags

    class FLAGS(FlagChoices):
        IGNORE = 1 << 1
        
    flags = FlagBitField(choices=FLAGS.choices(), default=0)

    # fields corresponding to MasterFlat kw arguments (margs_kwL)

    epoch = models.ForeignKey('Epoch', on_delete=models.CASCADE, related_name='masterflats') 
    instrument = models.CharField(max_length=20, choices=INSTRUMENTS.choices)
    imgsize = models.CharField(max_length=12, null=True)
    band = models.CharField(max_length=10, choices=BANDS.choices, default=BANDS.NONE)
    obsmode = models.CharField(max_length=20, choices=OBSMODES.choices, default=OBSMODES.NONE)
    rotangle = models.FloatField(null=True)
    exptime = models.FloatField(null=True)

    masterbias = models.ForeignKey('MasterBias', null=True, on_delete=models.CASCADE, related_name='masterflats')
    masterdark = models.ForeignKey('MasterDark', null=True, on_delete=models.CASCADE, related_name='masterflats')

    class Meta:
        app_label = 'iop4api'
        verbose_name = "Master Flat"
        verbose_name_plural = "Master Flats"

    # Properties

    @property
    def filename(self):
        return f"masterflat_id{self.id}.fits"
    
    @property
    def filepath(self):
        return os.path.join(self.epoch.masterflatdir, self.filename)
    
    @property
    def fileloc(self):
        return f"{self.epoch.epochname}/{self.filename}"

    # Filed Properties location
    
    @property
    def filedpropdir(self):
       return os.path.join(self.epoch.calibrationdir, 
                            self.filename +  '.d', 
                            self.__class__.__name__)
    
    # some helper properties

    @property
    def margs(self):
        """
        Return a dict of the arguments used to build this MasterFlat.
        """
        return {'epoch':self.epoch, 'instrument':self.instrument, 'imgsize':self.imgsize, 'band':self.band, 'obsmode':self.obsmode, 'rotangle':self.rotangle, 'exptime':self.exptime}
    
    # repr and str

    @classmethod
    def margs2str(cls, margs):
       """
       Class method to build a nice string rep of the arguments of a MasterFlat.
       """
       return f"{margs['epoch'].epochname} | {margs['instrument']} | {margs['imgsize']} | {margs['band']} | {margs['obsmode']} | {margs['rotangle']} º | {margs['exptime']}s"

    def __repr__(self):
        return f"MasterFlat.objects.get(id={self.id!r})"
    
    def __str__(self):
        return f"<MasterFlat {self.id!r} | {MasterFlat.margs2str(self.margs)}>"
    
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
               masterbias=None,
               masterdark=None,
               auto_merge_to_db=True,
               force_rebuild=True,
               **kwargs):
        """
            Create a MasterFlat object for Epoch epoch.

            Parameters
            ----------
            epoch: Epoch
            instrument: str
            imgsize: "WidthxHeight" str
            band: str
            obsmode: str
            rotangle: float
            exptime: flat

            Other Parameters
            ----------------
            rawfits: list of RawFit (optional)
            masterbias: MasterBias (optional)
            auto_merge_to_db: bool (optional) Default: True
        """

        #margs = {k:kwargs[k] for k in MasterFlat.margs_kwL if k in kwargs}
        margs = {k:kwargs[k] for k in MasterFlat.margs_kwL} # so it gives error if some margs kw missing

        from iop4lib.db import RawFit

        if (mf := MasterFlat.objects.filter(**margs).first()) is not None:
            logger.info(f"{mf} exists, using it instead.")
        else:
            logger.info(f"A MasterFlat entry for {MasterFlat.margs2str(margs)} will be created.")
            mf = cls(**margs)
            logger.debug("Saving MasterFlat to DB so that it has an id.")
            mf.save()

        if rawfits is None:
            rawfits = RawFit.objects.filter(imgtype=IMGTYPES.FLAT, **margs)
            logger.debug(f"Found {len(rawfits)} flat raw files for {mf}.")
        else:
            logger.debug(f"Using {len(rawfits)} flat raw files for {mf}.")

        mf.rawfits.set(rawfits)

        if masterbias is None:
            from .masterbias import MasterBias
            margs = {k:kwargs[k] for k in MasterBias.margs_kwL if k in kwargs}
            masterbias = MasterBias.objects.filter(**margs).first()    
            logger.debug(f"Using {masterbias} as MasterBias for {mf}.")
            mf.masterbias = masterbias        

        if masterdark is None:
            from .masterdark import MasterDark
            margs = {k:kwargs[k] for k in MasterDark.margs_kwL if k in kwargs if k != 'exptime'} # exptime is a build parameter, but darks with different exptime can be used
            masterdark = MasterDark.objects.filter(**margs).first()    
            logger.debug(f"Using {masterdark} as MasterDark for {mf}.")
            mf.masterdark = masterdark

        # Build the file
        if not mf.fileexists or force_rebuild:
            logger.info(f"Building file")
            try:
                mf.build_file()
            except Exception as e:
                logger.error(f"An error ocurred while building the MasterFlat, deleting it and raising Exception.")
                mf.delete()
                raise Exception(f"An error ocurred while building the MasterFlat for {margs}: {e}")
            else:
                logger.info("MasterFlat created successfully.")

        # merge to DB

        if auto_merge_to_db:
            mf.save()

        return mf

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.auto_merge_to_db=True
    
    # Methods

    def build_file(self):
        logger.debug(f"Getting data from files")

        if self.rawfits.count() == 0:
            raise Exception(f"No rawfits for {self}")

        data_L = []
        for rf in self.rawfits.all():
            with fits.open(rf.filepath) as hdul:

                if self.masterdark is not None:
                    data = hdul[0].data - self.masterbias.data - self.masterdark.data * self.exptime
                else:
                    logger.warning(f"No MasterDark for {self}, is this a CCD and it is cold?")
                    data = (hdul[0].data - self.masterbias.data) 
                
                data = data / np.nanmedian(data)
                data_L.append(data)

        data = np.nanmedian(data_L, axis=0)
        data = data / np.nanmedian(data)

        logger.debug(f"Building header")

        header = fits.Header()

        header['TELESCOP'] = self.epoch.telescope
        header['NIGHT'] = self.epoch.yyyymmdd
        header['EPOCH'] = self.epoch.epochname
        header['IMGSIZE'] = self.imgsize
        header['IMGTYPE'] = 'masterflat'
        header['DATECREA'] = datetime.datetime.utcnow().isoformat(timespec="milliseconds")
        header['NRAWFITS'] = self.rawfits.count()
        header['BAND'] = self.band
        header['INSTRUME'] = self.instrument
        header['OBSMODE'] = self.obsmode
        header['ROTANGLE'] = self.rotangle
        header['EXPTIME'] = self.exptime

        logger.debug(f"Building HDU")

        if not os.path.isdir(self.epoch.masterflatdir):
            logger.debug(f"Creating directory {self.epoch.masterflatdir}")
            os.makedirs(self.epoch.masterflatdir)

        hdu = fits.PrimaryHDU(data, header=header)

        logger.debug(f"Writing MasterFlat to {self.filepath}")
        
        hdu.writeto(self.filepath, overwrite=True)