from django.db import models

class IMGTYPES(models.TextChoices):
    """
    Enum for the type of a rawfit file.
    """

    NONE = None, "None"
    ERROR = "ERROR", "Error"
    FLAT = 'FLAT', "Flat"
    DARK = 'DARK', "Dark"
    BIAS = 'BIAS', "Bias"
    LIGHT = 'LIGHT', "Light"

class BANDS(models.TextChoices):
    """
    Enum for the band of a rawfit file.
    """

    NONE = None, "None"
    ERROR = "ERROR", "Error"
    I = "I", "I"
    R = 'R', "R"
    V = 'V', "V"
    U = 'U', "U"
    B = 'B', "B"

class OBSMODES(models.TextChoices):
    """
    Enum for possible observation modes (photometry, polarimetry, etc)
    """

    NONE = None, "None"
    ERROR = "ERROR", "Error"
    PHOTOMETRY = 'PHOTOMETRY', "Photometry"
    POLARIMETRY = 'POLARIMETRY', "Polarimetry"

class INSTRUMENTS(models.TextChoices):
    """
    Enum for possible instruments
    """

    NONE = None, "None"
    CAFOS = 'CAFOS2.2', "CAFOS2.2"
    RoperT90 = 'RoperT90', "RoperT90"
    AndorT90 = 'AndorT90', "AndorT90"
    AndorT150 = 'AndorT150', "AndorT150"
    DIPOL = 'DIPOL', "DIPOL"

class TELESCOPES(models.TextChoices):
    """
    Enum for possible telescopes: (name, name)
    """
    CAHA_T220 = 'CAHA-T220', "CAHA-T220"
    OSN_T090 = 'OSN-T090', "OSN-T090"
    OSN_T150 = 'OSN-T150', "OSN-T150"

class SRCTYPES(models.TextChoices):
    """
    Enum for possible types of sources
    """
    BLAZAR = 'blazar', "Blazar"
    STAR = 'star', "Star"
    CALIBRATOR = 'calibrator', "Calibrator"

class REDUCTIONMETHODS(models.TextChoices):
    """
    Enum for possible reduction methods
    """
    RELPHOT = 'relphot', "Relative photometry"
    RELPOL = 'relpol', "Relative photo-polarimetry"
    ABSPHOT = 'absphot', "Absolute photometry"

class PAIRS(models.TextChoices):
    O = "O", "Ordinary"
    E = "E", "Extraordinary"