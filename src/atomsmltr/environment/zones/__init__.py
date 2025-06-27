"""The ``atomsmltr.environment.zones`` subpackage provides definitions for spatial zones

Those zones are then used in ``Configuration`` objects to define position and speed zones
in which some actions have to be taken - for instance, stopping the simulation

Create a zone
---------------

Define two zones, for x and vx

.. code-block:: python

    from atomsmltr.environment.zones import Limits

    xlim = Limits(min=-1, max=1, axis=0, target="position", action="stop", tag="xlim")
    vxlim = Limits(min=0, max=500, axis=0, target="speed", action="stop", tag="vxlim")


Combining Zones
---------------

Once a series of ``Zone`` are defined, it is possible to combine them using
basic python operators. The result will be a ``ZoneCollection`` with different
combination rules, depending on the operator.

Note that a ``ZoneCollection`` is a particular kind of ``Zone``, so operators
also work on ``ZoneCollection``.


ANDCollection
~~~~~~~~~~~~~~~~

A ``ANDCollection`` can be generated using the ``&`` operator :

>>> and_collection = zone1 & zone2

The created ``and_collection`` object then has two zones in its ``.zones`` list,
namely ``zone1`` and ``zone2``. Its ``in_zone()`` method will return the result of
``zone1.get_value() and zone2.get_value()``


ORCollection
~~~~~~~~~~~~~~~~

A ``ORCollection`` can be generated using the ``|`` operator :

>>> or_collection = zone1 | zone2

XORCollection
~~~~~~~~~~~~~~~~

A ``XORCollection`` can be generated using the ``^`` operator :

>>> xor_collection = zone1 ^ zone2

Adding zones to a collection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Zones can be added to a collection using the ``+=`` operator :

>>> xor_collection = zone1 ^ zone2
>>> xor_collection += zone3

Zones of a same kind can also be appended

>>> and_coll1 = zone1 & zone2
>>> and_coll2 = zone3 & zone4
>>> and_coll3 = and_coll1 + and_coll2


See also
--------
atomsmltr.environment.zones.generic
atomsmltr.environment.zones.limits
atomsmltr.environment.zones.volumes

"""

__all__ = [
    "UpperLimit",
    "LowerLimit",
    "Limits",
    "Box",
    "Cylinder",
    "Zone",
    "SuperZone",
]

from .limits import UpperLimit, LowerLimit, Limits
from .volumes import Box, Cylinder
from .generic import Zone, SuperZone
