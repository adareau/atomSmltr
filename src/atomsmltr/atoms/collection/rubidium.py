"""Rubidium
=============
Implements a dedicated class for Rubidium atoms

>>> from atomsmltr.atoms import Rubidium
"""

# % IMPORTS
import scipy.constants as csts

# % LOCAL IMPORTS
from ...atoms.generic import Atom
from ...atoms.transitions import J0J1Transition

# % CONSTANTS

RUBIDIUM_87_MASS = 87 * csts.m_u  # kg
"""float: Rubidium 87 mass (kg)"""


# - D2 transition
RUBIDIUM_D2_WAVELENGTH = 780.241e-9  # m
"""float: Rubidium D2 transition wavelength (m)"""
RUBIDIUM_D2_GAMMA = 38.11e6  # rad/s
"""float: Rubidium D2 transition natural linewidth (rad/s)"""
RUBIDIUM_D2_LANDE_FACTOR = 1.0
"""float: Rubidium D2 transition Lande factor"""

# %% TRANSITIONS


class MainLine(J0J1Transition):
    """The D2 (780nm) transition of Rubidium"""

    def __init__(self):
        super().__init__(
            lande_factor=RUBIDIUM_D2_LANDE_FACTOR,
            wavelength=RUBIDIUM_D2_WAVELENGTH,
            Gamma=RUBIDIUM_D2_GAMMA,
            tag="main",
        )


# %% ATOM


class Rubidium(Atom):
    """Rubidium 87 atomic class"""

    def __init__(self):

        # init super class
        super().__init__(
            mass=RUBIDIUM_87_MASS,
            name="Rubidium",
        )
        # add transitions
        D2 = MainLine()
        self.add_transition(D2)
