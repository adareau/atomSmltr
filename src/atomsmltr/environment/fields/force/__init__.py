"""The ``atomsmltr.environment.fields.force`` subpackage provides definitions for forces

Those are mostly direct implementations of generic `Fields` objects
from `atomsmltr.environment.fields.generic`

Examples
---------

Setup a gravitational force

.. code-block:: python

    import numpy as np
    from atomsmltr.environment import ConstantForce
    from atomsmltr.atoms import Ytterbium

    m = Ytterbium().mass  # kg
    g = 9.81  # m/s^2
    direction = np.array([0, 0, -1])  # along -z
    grav_force = m * g * direction

    gravity = ConstantForce(field_value=grav_force, tag="gravity")

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
