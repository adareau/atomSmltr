# -*- coding: utf-8 -*-
"""Pre-made classes for Ytterbium
"""

# % IMPORTS
import scipy.constants as csts

# % LOCAL IMPORTS
from ...atoms.generic import Atom
from ...atoms.transitions import J0J1Transition

# % CONSTANTS

YTTERBIUM_174_MASS = 173.94 * csts.m_u  # kg

YTTERBIUM_MAIN_WAVELENGTH = 398.911e-9  # m
YTTERBIUM_MAIN_GAMMA = 2 * csts.pi * 28.9e6  # rad/s
YTTERBIUM_MAIN_LANDE_FACTOR = 1.035

YTTERBIUM_INTERCOMBINATION_WAVELENGTH = 555.802e-9  # m
YTTERBIUM_INTERCOMBINATION_GAMMA = 2 * csts.pi * 182e3  # rad/s
YTTERBIUM_INTERCOMBINATION_LANDE_FACTOR = 1.493

# %% TRANSITIONS


class MainLine(J0J1Transition):
    """The main (blue) transition of ytterbium"""

    def __init__(self):
        super().__init__(
            lande_factor=YTTERBIUM_MAIN_LANDE_FACTOR,
            lbda=YTTERBIUM_MAIN_WAVELENGTH,
            Gamma=YTTERBIUM_MAIN_GAMMA,
            tag="main",
        )


class IntercombinationLine(J0J1Transition):
    """The main (blue) transition of ytterbium"""

    def __init__(self):
        super().__init__(
            lande_factor=YTTERBIUM_INTERCOMBINATION_LANDE_FACTOR,
            lbda=YTTERBIUM_INTERCOMBINATION_WAVELENGTH,
            Gamma=YTTERBIUM_INTERCOMBINATION_GAMMA,
            tag="intercombination",
        )


# %% ATOM


class Ytterbium(Atom):

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
