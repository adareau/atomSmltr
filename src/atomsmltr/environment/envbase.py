"""
environment objects
=======================

This module implements the generic ``EnvObject`` class. All objects aimed at defining
the atom's environment will derive from this class.

This will help ensure that those objects share common properties and methods, which will
ease their use inside a ``Configuration`` class.
"""

# % IMPORTS
from abc import abstractmethod
from copy import copy
import numpy as np

# % LOCAL IMPORTS
from ..utils.plotter import Plottable
from ..utils.misc import random_word, check_position_array

# % ABSTRACT CLASSES


class EnvObject(Plottable):
    """Defines a generic (abstract) class for environment objects.

    Notably, this makes sure that:

      (1) they are ``Plottable`` objects
      (2) they feature a ``get_value()`` method that satisfies our vectorization convention
      (3) they have a ``tag`` property.

    Parameters
    ----------
    tag : str, optional (default=None)
        a tag identifying the object. If `None` is given, a random tag
        will be generated

    See Also
    ----------
    atomsmltr.utils.plotter.plotter.Plottable
    """

    def __init__(self, tag: str = None):
        # init tag with random word if None
        if tag is None:
            tag = random_word()
        self.tag = tag
        super(EnvObject, self).__init__()

    # -- UNIFIED GET_VALUE FUNCTION

    @abstractmethod
    def get_value(self, position: np.ndarray, nocheck=False) -> np.ndarray:
        """Returns the ``EnvObject`` 'value' at a given position

        Parameters
        ----------
        position : array of shape (3,) or (n1, n2, ..., 3)
            cartesian coordinates in the lab frame
        nocheck : bool, optional
            if set to True, function will not check that the shape of position
            matches requirements, by default False

        Returns
        -------
        intensity : float or array of shape (n1, n2, ..., 1)
            laser intensity at position

        Notes
        -------
        position is an array_like object, with shape (3,) or (n1, n2, .., 3).
        In all cases, the last dimension contains cordinates (x, y, z), in meter and in the lab frame

        Note
        -----
            this is an **abstract** method, and the actual method has to be implemented
            for each child class.
        """

        # -- check position array
        pass

    def _check_position_array(self, position: np.ndarray, nocheck=False) -> np.ndarray:
        """Checks that a position array matches our vectorization convention.

        It raises an error if the shape is not good.

        Parameters
        ----------
        position : array
            the array to check
        nocheck : bool, optional
            if set to True, the function is bypasses

        Returns
        -------
        position
            the array

        Notes
        ------
        positions array should have a shape (3,) or (n1, n2, .., 3).

        In all cases, the last dimension contains cordinates (x, y, z),
        in meter and in the lab frame
        """
        return check_position_array(position, nocheck)

    # -- TAG
    @property
    def tag(self) -> str:
        """str: the object tag"""
        return self._tag

    @tag.setter
    def tag(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("'tag' should be a string")
        self._tag = value

    # -- COPY
    def copy(self, new_tag: str = None):
        """returns a copy of the object.

        Parameters
        ----------
        new_tag : str, optional
            a tag for the copy, if None is given a random tag is generated.

        Returns
        -------
        object_copy : EnvObject
            a copy of the object
        """
        object_copy = copy(self)
        if new_tag is None:
            new_tag = f"{self.tag}-{random_word()}"
        object_copy.tag = new_tag
        return object_copy

    # -- INFO STRING / OBJECT MANAGEMENT
    @abstractmethod
    def gen_infostring_obj(self):
        """generates an ``InfoString`` object.

        Returns
        -------
        InfoString
            an ``InfoString`` object

        See also
        --------
        atomsmltr.utils.infostring.InfoString
        """
        pass

    def gen_info_string(self, **kwargs):
        """generates an info string

        Returns
        -------
        info_string: str
            a string with information on the atom
        """
        return self.gen_infostring_obj().generate(**kwargs)

    def print_info(self):
        """prints the atom infostring"""
        print(self.gen_info_string())

    @property
    @abstractmethod
    def type():
        """str: a description of the object type"""
        pass

    @property
    @abstractmethod
    def vector():
        """bool: True it the object value is vectorial, False if scalar"""
        pass
