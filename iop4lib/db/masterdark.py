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

class MasterDark(FitFileModel):
    """
    A class representing a master dark for an epoch.
    """

    margs_kwL = ['epoch', 'instrument', 'imgsize', 'exptime']

    imgtype = IMGTYPES.DARK

    # Database fields
    rawfits = models.ManyToManyField('RawFit', related_name='built_masterdarks')

    # flags

    class FLAGS(FlagChoices):
        IGNORE = 1 << 1
        
    flags = FlagBitField(choices=FLAGS.choices(), default=0)

    # fields corresponding to MasterDark kw arguments (margs_kwL)

    epoch = models.ForeignKey('Epoch', on_delete=models.CASCADE, related_name='masterdarks') 
    instrument = models.CharField(max_length=20, choices=INSTRUMENTS.choices)
    imgsize = models.CharField(max_length=12, null=True)
    exptime = models.FloatField(null=True)

    masterbias = models.ForeignKey('MasterBias', null=True, on_delete=models.CASCADE, related_name='masterdarks')

    class Meta:
        app_label = 'iop4api'
        verbose_name = "Master Dark"
        verbose_name_plural = "Master Darks"

    # Properties

    @property
    def filename(self):
        return f"masterdark_id{self.id}.fits"
    
    @property
    def filepath(self):
        return os.path.join(self.epoch.masterdarkdir, self.filename)
    
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
        Return a dict of the arguments used to build this MasterDark.
        """
        return {'epoch':self.epoch, 'instrument':self.instrument, 'imgsize':self.imgsize, 'exptime':self.exptime}
    
    # repr and str

    @classmethod
    def margs2str(cls, margs):
       """
       Class method to build a nice string rep of the arguments of a MasterDark.
       """
       return f"{margs['epoch'].epochname} | {margs['instrument']} | {margs['imgsize']} | {margs['exptime']}s"

    def __repr__(self):
        return f"MasterDark.objects.get(id={self.id!r})"
    
    def __str__(self):
        return f"<MasterDark {self.id!r} | {MasterDark.margs2str(self.margs)}>"
    
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
               auto_merge_to_db=True,
               force_rebuild=True,
               **kwargs):
        """
            Create a MasterDark object for Epoch epoch.

            Parameters
            ----------
            epoch: Epoch
            instrument: str
            imgsize: "WidthxHeight" str
            band: str
            obsmode: str
            rotangle: float
            exptime: float

            Other Parameters
            ----------------
            rawfits: list of RawFit (optional)
            masterbias: MasterBias (optional)
            auto_merge_to_db: bool (optional) Default: True
        """

        #margs = {k:kwargs[k] for k in MasterDark.margs_kwL if k in kwargs}
        margs = {k:kwargs[k] for k in MasterDark.margs_kwL} # so it gives error if some margs kw missing

        from iop4lib.db import RawFit

        if (md := MasterDark.objects.filter(**margs).first()) is not None:
            logger.info(f"{md} exists, using it instead.")
        else:
            logger.info(f"A MasterDark entry for {MasterDark.margs2str(margs)} will be created.")
            md = cls(**margs)
            logger.debug("Saving MasterDark to DB so that it has an id.")
            md.save()

        if rawfits is None:
            rawfits = RawFit.objects.filter(imgtype=IMGTYPES.DARK, **margs)
            logger.debug(f"Found {len(rawfits)} dark raw files for {md}.")
        else:
            logger.debug(f"Using {len(rawfits)} dark raw files for {md}.")

        md.rawfits.set(rawfits)

        if masterbias is None:
            from .masterbias import MasterBias
            margs = {k:kwargs[k] for k in MasterBias.margs_kwL if k in kwargs}
            masterbias = MasterBias.objects.filter(**margs).first()    
            logger.debug(f"Using {masterbias} as MasterBias for {md}.")
            md.masterbias = masterbias        

        # Build the file
        if not md.fileexists or force_rebuild:
            logger.info(f"Building file")
            try:
                md.build_file()
            except Exception as e:
                logger.error(f"An error ocurred while building the MasterDark, deleting it and raising Exception.")
                md.delete()
                raise Exception(f"An error ocurred while building the MasterDark for {margs}: {e}")
            else:
                logger.info("MasterDark created successfully.")

        # merge to DB

        if auto_merge_to_db:
            md.save()

        return md

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
                data = (hdul[0].data - self.masterbias.data) / self.exptime
                data_L.append(data)

        data = np.nanmedian(data_L, axis=0)

        logger.debug(f"Building header")

        header = fits.Header()

        header['TELESCOP'] = self.epoch.telescope
        header['NIGHT'] = self.epoch.yyyymmdd
        header['EPOCH'] = self.epoch.epochname
        header['IMGSIZE'] = self.imgsize
        header['IMGTYPE'] = 'masterdark'
        header['DATECREA'] = datetime.datetime.utcnow().isoformat(timespec="milliseconds")
        header['NRAWFITS'] = self.rawfits.count()
        header['INSTRUME'] = self.instrument
        header['EXPTIME'] = self.exptime

        logger.debug(f"Building HDU")

        if not os.path.isdir(self.epoch.masterdarkdir):
            logger.debug(f"Creating directory {self.epoch.masterdarkdir}")
            os.makedirs(self.epoch.masterdarkdir)

        hdu = fits.PrimaryHDU(data, header=header)

        logger.debug(f"Writing MasterDark to {self.filepath}")
        
        hdu.writeto(self.filepath, overwrite=True)