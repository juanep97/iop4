import astropy.units as u

from typing import TYPE_CHECKING

from typing import Iterable, Any, List, Tuple, Union, NamedTuple, TypedDict, Literal

if TYPE_CHECKING:

    from iop4lib.db import (
        Epoch,
        RawFit,
        ReducedFit,
        AperPhotResult,
        PhotoPolResult,
        AstroSource,
    )

    from iop4lib.utils.polarization import (
        PolarimetryGroup,
    )

    from iop4lib.utils.astrometry import (
        BuildWCSResult,
    )

class FwhmStatsTuple(NamedTuple):
    mean: u.Quantity
    std: u.Quantity
    median: u.Quantity

class SourcePairTuple(NamedTuple):
    astrosource: 'AstroSource'
    pair: Literal['O','E']

class Centroid(NamedTuple):
    x_px: float
    y_px: float

class CentroidFwhmTuple(NamedTuple):
    centroid: Centroid
    fwhm: u.Quantity

class CentroidsAndFwhmResultTuple(NamedTuple):
    centroids_and_fwhms: dict[SourcePairTuple, CentroidFwhmTuple]
    fwhm_stats: FwhmStatsTuple
    detected_fwhms: u.Quantity | None

class CommonAperturesTuple(NamedTuple):
    r_ap: u.Quantity
    r_in: u.Quantity
    r_out: u.Quantity
    ap_fwhm: u.Quantity # the one used to compute aperture and annulus radii
    fwhm_stats: FwhmStatsTuple # aggregate for all reduced fits in the group
    centroids_and_fwhms: dict['ReducedFit', CentroidsAndFwhmResultTuple]

class InstrumentalPolarizationDict(TypedDict):
    q_inst: float
    dq_inst: float
    u_inst: float
    du_inst: float
    CPA: float
    dCPA: float
