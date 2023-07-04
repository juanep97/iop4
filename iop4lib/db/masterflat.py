import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

from django.db import models

import os
import datetime
import numpy as np
import astropy.io.fits as fits

from .fitfilemodel import FitFileModel
from ..enums import *

import logging
logger = logging.getLogger(__name__)

class MasterFlat(FitFileModel):
    """
    A class representing a master flat for an epoch.
    """

    mfargs_kwL = ['epoch', 'instrument', 'imgsize', 'band', 'obsmode', 'rotangle', 'exptime']

    # Database fields
    rawfits = models.ManyToManyField('RawFit', related_name='built_masterflats')

    # fields corresponding to MasterFlat kw arguments (mfargs_kwL)

    epoch = models.ForeignKey('Epoch', on_delete=models.CASCADE, related_name='masterflats') 
    instrument = models.CharField(max_length=20, choices=INSTRUMENTS.choices)
    imgsize = models.CharField(max_length=12, null=True)
    band = models.CharField(max_length=10, choices=BANDS.choices, default=BANDS.NONE)
    obsmode = models.CharField(max_length=20, choices=OBSMODES.choices, default=OBSMODES.NONE)
    rotangle = models.FloatField(null=True)
    exptime = models.FloatField(null=True)

    masterbias = models.ForeignKey('MasterBias', null=True, on_delete=models.CASCADE, related_name='masterflats')

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
    def mfargs(self):
        """
        Return a dict of the arguments used to build this MasterFlat.
        """
        return {'epoch':self.epoch, 'instrument':self.instrument, 'imgsize':self.imgsize, 'band':self.band, 'obsmode':self.obsmode, 'rotangle':self.rotangle, 'exptime':self.exptime}
    
    # repr and str

    @classmethod
    def mfargs2str(cls, mfargs):
       """
       Class method to build a nice string rep of the arguments of a MasterFlat.
       """
       return f"{mfargs['epoch'].epochname} | {mfargs['instrument']} | {mfargs['imgsize']} | {mfargs['band']} | {mfargs['obsmode']} | {mfargs['rotangle']} º | {mfargs['exptime']}s"

    def __repr__(self):
        return f"MasterFlat.objects.get(id={self.id!r})"
    
    def __str__(self):
        return f"<MasterFlat {self.id!r} | {MasterFlat.mfargs2str(self.mfargs)}>"
    
    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text(f"{self!r}")
        else:
            with p.group(4, f'<{self.__class__.__name__}(', ')>'):
                p.text(f"id: {self.id},")
                for k,v in self.mbargs.items():
                    p.breakable()
                    p.text(f"{k}: {v}")

    
    # Constructor
    @classmethod
    def create(cls, 
               rawfits=None, 
               masterbias=None,
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

        #mfargs = {k:kwargs[k] for k in MasterFlat.mfargs_kwL if k in kwargs}
        mfargs = {k:kwargs[k] for k in MasterFlat.mfargs_kwL} # so it gives error if some mfargs kw missing

        from iop4lib.db import RawFit

        if (mf := MasterFlat.objects.filter(**mfargs).first()) is not None:
            logger.info(f"{mf} exists, using it instead.")
        else:
            logger.info(f"A MasterFlat entry for {MasterFlat.mfargs2str(mfargs)} will be created.")
            mf = cls(**mfargs)
            logger.debug("Saving MasterFlat to DB so that it has an id.")
            mf.save()

        if rawfits is None:
            rawfits = RawFit.objects.filter(imgtype=IMGTYPES.FLAT, **mfargs)
            logger.debug(f"Found {len(rawfits)} flat raw files for {mf}.")
        else:
            logger.debug(f"Using {len(rawfits)} flat raw files for {mf}.")

        mf.rawfits.set(rawfits)

        if masterbias is None:
            from .masterbias import MasterBias
            mbargs = {k:kwargs[k] for k in MasterBias.mbargs_kwL if k in kwargs}
            masterbias = MasterBias.objects.filter(**mbargs).first()    
            logger.debug(f"Using {masterbias} as MasterBias for {mf}.")
            mf.masterbias = masterbias        

        # Build the file
        if not mf.fileexists or force_rebuild:
            logger.info(f"Building file")
            try:
                mf.build_file()
            except Exception as e:
                logger.error(f"An error ocurred while building the MasterFlat, deleting it and raising Exception.")
                mf.delete()
                raise Exception(f"An error ocurred while building the MasterFlat for {mfargs}: {e}")
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