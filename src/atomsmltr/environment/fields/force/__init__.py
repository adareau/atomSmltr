"""The ``atomsmltr.environment.fields.force`` subpackage provides definitions for forces

Those are mostly direct implementations of generic `Fields` objects
from `atomsmltr.environment.fields.generic`

Examples
---------

Setup a gravitational force

.. code-block:: python

    TODO

See also
--------
atomsmltr.environment.fields.generic

"""

__all__ = [
    "GradientForce",
    "ConstantForce",
    "Force",
]

from .force import (
    GradientForce,
    ConstantForce,
    Force,
)
