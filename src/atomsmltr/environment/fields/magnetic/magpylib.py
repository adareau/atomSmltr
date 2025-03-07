"""
magpylib wrapper
=======================

Defines a class allowing to wrap a magpylib object into a ``MagneticField`` object
compatible with ``atomsmltr``

see: https://magpylib.readthedocs.io

"""

# % IMPORTS
import numpy as np


# % LOCAL IMPORTS
from .magnetic import MagneticField
from ....utils.infostring import InfoString
import magpylib as magpy


class MagpylibWrapper(MagneticField):
    """A wrapper for magpylib objects

    Parameters
    ----------
    magpy_object : magpy_object
        the object to wrap
    tag : str, optional
        the field tag, by default None
    """

    def __init__(self, magpy_object, tag: str = None):
        super(MagpylibWrapper, self).__init__(tag)
        self.magpy_object = magpy_object

    @property
    def type(self):
        return "magpylib object"

    # -- requested methods for Field
    # pylint : disable=method_hidden
    def _field_value_func(self, position):
        """Returns field value at point position

        position should be an array of shape (3,) or (n1,n2,..,3)
        last axis contains coordinates x, y, z

        NB: position is already checked and converted to an array in the
            `Field` class
        """
        # let's call the magpy get_B function
        B = magpy.getB(self.magpy_object, position, squeeze=False)
        # sqeeeeeeze
        value = B.T
        while value.ndim > position.ndim:
            value = np.squeeze(value, axis=-1)
        return value.T

    def gen_infostring_obj(self):
        """Generates an info string object"""
        unit = self.unit
        title = self.type
        title = title[:1].upper() + title[1:]  # capitalize first letter
        info = InfoString(title=title)
        info.add_section("Parameters")
        info.add_element("type", "magpylib object")
        info.add_element("tag", self.tag)
        # TODO can we have more info ?
        return info
