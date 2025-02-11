import pytest


def test_atom_import():
    from atomsmltr.atoms import generic


def test_atom_transitions():
    from atomsmltr.atoms.generic import Atom
    from atomsmltr.atoms.transitions import DummyTransition
    from scipy import constants as csts

    # - Initialize an atom
    atom = Atom(mass=4 * csts.m_u, name="Helium")
    transition = DummyTransition("transition", Gamma=1, lbda=1)
    # -- Adding transitions
    # - wrong transitions
    with pytest.raises(TypeError) as excinfo:
        atom.add_transition(5)
    # - same tag
    atom.add_transition(transition)
    with pytest.raises(AssertionError) as excinfo:
        atom.add_transition(transition)
    # - same tag, v2
    atom.add_transition(transition, tag="transition2")
    with pytest.raises(AssertionError) as excinfo:
        atom.add_transition(transition, tag="transition2")
    # -- listing transitions
    assert sorted(["transition2", "transition"]) == sorted(atom.list_transitions())
    # -- removing transitions
    atom.rm_transition("transition")
    assert atom.list_transitions() == [
        "transition2",
    ]
    # - now returns a key error
    with pytest.raises(KeyError) as excinfo:
        atom.rm_transition("transition")


def test_ytterbium_atom():
    from atomsmltr.atoms.collection import Ytterbium

    yb = Ytterbium()
    print(yb.mass_au)


if __name__ == "__main__":
    test_atom_import()
    test_atom_transitions()
    test_ytterbium_atom()
