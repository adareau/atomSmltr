"""The `atomsmltr.atoms` subpackage provides classes to handle atomic species and their transitions.
"""

__all__ = [
    "Atom",
    "Ytterbium",
    "Strontium",
]

from .generic import Atom
from .collection import Ytterbium, Strontium
