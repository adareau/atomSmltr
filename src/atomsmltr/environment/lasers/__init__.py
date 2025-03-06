"""The ``atomsmltr.environment.lasers``\ subpackage provides definitions for lasers

It defines two main classes :

+ ``LaserBeam``    > to manage laser intensity, propagation, etc.
+ ``Polarization`` > to specifically manage the polarization

Examples
---------

Setup a Gaussian beam

.. code-block:: python

    from atomsmltr.environment.lasers import GaussianLaserBeam
    from atomsmltr.environment.lasers.polarization import CircularLeft


    beam = GaussianLaserBeam(
        wavelength=399e-9,
        waist=50e-6,
        power=30e-3,
        waist_position=(0, 0, 0),
        direction=(0, 0, 1),
        polarization=CircularLeft(),
    )

See also
--------
atomsmltr.environment.lasers.beams
atomsmltr.environment.lasers.polarization

"""

__all__ = [
    "GaussianLaserBeam",
    "PlaneWaveLaserBeam",
    "LaserBeam",
    "Polarization",
    "Vertical",
    "Horizontal",
    "CircularLeft",
    "CircularRight",
    "Vector",
    "Linear",
]

from .beams import GaussianLaserBeam, PlaneWaveLaserBeam, LaserBeam
from .polarization import (
    Polarization,
    Vertical,
    Horizontal,
    CircularLeft,
    CircularRight,
    Vector,
    Linear,
)
