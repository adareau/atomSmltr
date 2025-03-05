"""The ``atomsmltr.environment`` subpackage provides classes to define the atom environment (lasers, mag. fields, zones).

Content
------------------

| ``atomsmltr.environment.fields``  : vector fields, currently only magnetic
| ``atomsmltr.environment.lasers``  : laser beams
| ``atomsmltr.environment.zones``   : zones in position or speed space
"""

from .fields.magnetic import MagneticField
from .lasers.beams import LaserBeam
from .zones.generic import Zone
