"""Strontium
=============
Implements a dedicated class for Strontium atoms

>>> from atomsmltr.atoms import Strontium
"""

# % IMPORTS
import scipy.constants as csts

# % LOCAL IMPORTS
from ...atoms.generic import Atom
from ...atoms.transitions import J0J1Transition

# % CONSTANTS

STRONTIUM_88_MASS = 88 * csts.m_u  # kg
"""float: Strontium 88 mass (kg)"""

# - main transition
STRONTIUM_MAIN_WAVELENGTH = 460.862e-9  # m
"""float: Strontium main transition wavelength (m)"""
STRONTIUM_MAIN_GAMMA = 2 * csts.pi * 32.0e6  # rad/s
"""float: Strontium main transition natural linewidth (rad/s)"""
STRONTIUM_MAIN_LANDE_FACTOR = 1.0
"""float: Strontium main transition Lande factor"""

# - intercombination transition
STRONTIUM_INTERCOMBINATION_WAVELENGTH = 689.449e-9  # m
"""float: Strontium intercombination transition wavelength (m)"""
STRONTIUM_INTERCOMBINATION_GAMMA = 2 * csts.pi * 7.46e3  # rad/s
"""float: Strontium intercombination transition natural linewidth (rad/s)"""
STRONTIUM_INTERCOMBINATION_LANDE_FACTOR = 1.5
"""float: Strontium intercombination transition Lande factor"""

# %% TRANSITIONS


class MainLine(J0J1Transition):
    """The main (461nm) transition of Strontium"""

    def __init__(self):
        super().__init__(
            lande_factor=STRONTIUM_MAIN_LANDE_FACTOR,
            wavelength=STRONTIUM_MAIN_WAVELENGTH,
            Gamma=STRONTIUM_MAIN_GAMMA,
            tag="main",
        )


class IntercombinationLine(J0J1Transition):
    """The intercombination (689nm) transition of Strontium"""

    def __init__(self):
        super().__init__(
            lande_factor=STRONTIUM_INTERCOMBINATION_LANDE_FACTOR,
            wavelength=STRONTIUM_INTERCOMBINATION_WAVELENGTH,
            Gamma=STRONTIUM_INTERCOMBINATION_GAMMA,
            tag="intercombination",
        )


# %% ATOM


class Strontium(Atom):
    """Strontium 88 atomic class"""

    def __init__(self):

        # init super class
        super().__init__(
            mass=STRONTIUM_88_MASS,
            name="Strontium",
        )
        # add transitions
        main = MainLine()
        intercomb = IntercombinationLine()
        self.add_transition(main)
        self.add_transition(intercomb)
