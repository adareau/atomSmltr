"""The ``atomsmltr.atoms`` subpackage provides classes to handle atomic species and their transitions.

Examples
---------

Using an existing atom

.. code-block:: python

    from atomsmltr.atoms import Ytterbium  # import `Ytterbium` class
    yb = Ytterbium()  # init an object
    yb.print_info()  # print atom informations

One can also define a new atom from scratch

.. code-block:: python

    from atomsmltr.atoms import Atom
    from atomsmltr.atoms.transitions import DummyTransition
    from scipy import constants as csts
    atom = Atom(mass=4 * csts.m_u, name="Helium")
    transition = DummyTransition("transition", Gamma=1, wavelength=1)
    atom.add_transition(transition, tag="transition")

"""

__all__ = [
    "Atom",
    "Ytterbium",
    "Strontium",
    "Rubidium",
    "DummyTransition",
    "J0J1Transition",
]

from .generic import Atom
from .collection import Ytterbium, Strontium, Rubidium
from .transitions import DummyTransition, J0J1Transition
