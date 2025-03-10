.. _documentation-laser-propagation:

Laser propagation conventions
===============================

Laser direction
----------------

When creating a ``LaserBeam`` object, one has to provide its ``direction``, which can be defined in two ways:

.. grid:: 2
    :gutter: 2

    .. grid-item-card::

        1) in the form of a **vector**, defining the propagation axis of the laser. Note that this vector *does not* have to be *a unit vector*: the ``LaserBeam`` object will take care of normalizing it when needed.

    .. grid-item-card::

        2) in the form of a **(θ, 𝜙)** tuple, where **θ** and **𝜙** are respectively the **polar** and **azimuthal** angles defining the orientation of the laser propagation axis.

Here is an example of how to use those two conventions to declare a laser beam propagation along the **+z** direction.

.. code-block:: python

    from atomsmltr.environment import GaussianLaserBeam
    # beam propagating along z
    # option 1 > unit_vector = (0,0,1)
    beam = GaussianLaserBeam(direction=(0,0,1), direction_type="vector")
    # option 1 > θ = 0, 𝜙 =0
    beam = GaussianLaserBeam(direction=(0,0), direction_type="thetaphi")


Inclure illustration ici


Laser frame
------------
