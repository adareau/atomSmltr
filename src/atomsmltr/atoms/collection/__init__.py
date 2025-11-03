"""The ``atomsmltr.atoms.collection`` subpackage provides definitions for common atomic species

Examples
---------

Import an ytterbium atom

.. code-block:: python

    from atomsmltr.atoms import Ytterbium  # import `Ytterbium` class
    yb = Ytterbium()  # init an object
    yb.print_info()  # print atom informations

"""

__all__ = [
    "Ytterbium",
    "Strontium",
    "Rubidium",
]

from .ytterbium import Ytterbium
from .strontium import Strontium
from .rubidium import Rubidium
