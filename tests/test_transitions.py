import pytest


def test_J0J1Transition():
    from atomsmltr.atoms.transitions import J0J1Transition

    # - Init
    trans = J0J1Transition(lande_factor=1, tag="transition", Gamma=0, wavelength=1)


def test_Ytterbium_transitions():
    from atomsmltr.atoms.collection.ytterbium import MainLine, IntercombinationLine

    main = MainLine()
    print(main.Isat_mW_per_cm2)
    print(main.lande_factor)

    intercomb = IntercombinationLine()
    print(intercomb.Isat_mW_per_cm2)
    print(intercomb.lande_factor)


if __name__ == "__main__":
    test_J0J1Transition()
    test_Ytterbium_transitions()
