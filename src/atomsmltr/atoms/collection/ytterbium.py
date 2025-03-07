"""Ytterbium
=============
Implements a dedicated class for Ytterbium atoms

>>> from atomsmltr.atoms import Ytterbium
"""

# % IMPORTS
import scipy.constants as csts

# % LOCAL IMPORTS
from ...atoms.generic import Atom
from ...atoms.transitions import J0J1Transition

# % CONSTANTS

YTTERBIUM_174_MASS = 173.94 * csts.m_u  # kg
"""float: Ytterbium 88 mass (kg)"""

YTTERBIUM_MAIN_WAVELENGTH = 398.911e-9  # m
"""float: Ytterbium main transition wavelength (m)"""
YTTERBIUM_MAIN_GAMMA = 2 * csts.pi * 28.9e6  # rad/s
"""float: Ytterbium main transition natural linewidth (rad/s)"""
YTTERBIUM_MAIN_LANDE_FACTOR = 1.035
"""float: Ytterbium main transition Lande factor"""

YTTERBIUM_INTERCOMBINATION_WAVELENGTH = 555.802e-9  # m
"""float: Ytterbium intercombination transition wavelength (m)"""
YTTERBIUM_INTERCOMBINATION_GAMMA = 2 * csts.pi * 182e3  # rad/s
"""float: Ytterbium intercombination transition natural linewidth (rad/s)"""
YTTERBIUM_INTERCOMBINATION_LANDE_FACTOR = 1.493
"""float: Ytterbium intercombination transition Lande factor"""

# %% TRANSITIONS


class MainLine(J0J1Transition):
    """The main (399nm) transition of Ytterbium"""

    def __init__(self):
        super().__init__(
            lande_factor=YTTERBIUM_MAIN_LANDE_FACTOR,
            wavelength=YTTERBIUM_MAIN_WAVELENGTH,
            Gamma=YTTERBIUM_MAIN_GAMMA,
            tag="main",
        )


class IntercombinationLine(J0J1Transition):
    """The intercombination (556nm) transition of Ytterbium"""

    def __init__(self):
        super().__init__(
            lande_factor=YTTERBIUM_INTERCOMBINATION_LANDE_FACTOR,
            wavelength=YTTERBIUM_INTERCOMBINATION_WAVELENGTH,
            Gamma=YTTERBIUM_INTERCOMBINATION_GAMMA,
            tag="intercombination",
        )


# %% ATOM


class Ytterbium(Atom):
    """Strontium 174 atomic class"""

    def __init__(self):

        # init super class
        super().__init__(
            mass=YTTERBIUM_174_MASS,
            name="Ytterbium",
        )
        # add transitions
        main = MainLine()
        intercomb = IntercombinationLine()
        self.add_transition(main)
        self.add_transition(intercomb)
