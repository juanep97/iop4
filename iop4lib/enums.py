from django.db import models

class IMGTYPES(models.TextChoices):
    """
    Enum for the type of a rawfit file.
    """

    NONE = None, "None"
    ERROR = "ERROR", "Error"
    FLAT = 'FLAT', "Flat"
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
    AndorT90 = 'AndorT90', "AndorT90"

class TELESCOPES(models.TextChoices):
    """
    Enum for possible telescopes: (name, name)
    """
    CAHA_T220 = 'CAHA-T220', "CAHA-T220"
    OSN_T090 = 'OSN-T090', "OSN-T090"

class SRCTYPES(models.TextChoices):
    """
    Enum for possible types of sources
    """
    BLAZAR = 'blazar', "Blazar"
    STAR = 'star', "Star"
    CALIBRATOR = 'calibrator', "Calibrator"
    UNPOLARIZED_FIELD_STAR = 'unpolarized_field_star', "Unpolarized field star"

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