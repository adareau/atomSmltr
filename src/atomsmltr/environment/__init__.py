"""The ``atomsmltr.environment`` subpackage provides classes to define the atom environment (lasers, mag. fields, zones).

Content
------------------

| ``atomsmltr.environment.fields``  : vector fields (magnetic fields & forces)
| ``atomsmltr.environment.lasers``  : laser beams
| ``atomsmltr.environment.zones``   : zones in position or speed space
"""

from .fields.magnetic import *
from .fields.force import *
from .lasers import *
from .zones import *
