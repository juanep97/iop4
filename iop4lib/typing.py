import astropy.units as u

from typing import TYPE_CHECKING

from typing import Any, List, Tuple, Union, NamedTuple, TypedDict, Literal

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
