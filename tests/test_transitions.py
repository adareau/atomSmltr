import pytest


def test_J0J1Transition():
    from atomsmltr.atoms.transitions import J0J1Transition

    # - Init
    trans = J0J1Transition(lande_factor=1, tag="transition", Gamma=0, lbda=1)


if __name__ == "__main__":
    test_J0J1Transition()
