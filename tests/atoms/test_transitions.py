import pytest


def test_J0J1Transition():
    from atomsmltr.atoms.transitions import J0J1Transition

    # - Init
    trans = J0J1Transition(lande_factor=1, tag="transition", Gamma=0, wavelength=1)


def test_Ytterbium_transitions():
    from atomsmltr.atoms.collection.ytterbium import MainLine, IntercombinationLine

    main = MainLine()
    main.print_info()

    intercomb = IntercombinationLine()
    intercomb.print_info()


def test_Strontium_transitions():
    from atomsmltr.atoms.collection.strontium import MainLine, IntercombinationLine

    main = MainLine()
    main.print_info()

    intercomb = IntercombinationLine()
    intercomb.print_info()


def test_Rubidium_transitions():
    from atomsmltr.atoms.collection.rubidium import MainLine

    main = MainLine()
    main.print_info()


if __name__ == "__main__":
    test_J0J1Transition()
    test_Ytterbium_transitions()
    test_Strontium_transitions()
