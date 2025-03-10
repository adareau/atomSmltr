.. _documentation-spatial-coordinates:

Spatial coordinates conventions
================================

In ``atomsmltr`` objects high-level methods, positions are given **cartesian coordinates** in the **laboratory frame**. This is the case for the ``get_intensity()`` methold of the ``LaserBeam`` class, or the ``value()`` method of the ``MagneticField`` class.


.. note::

    In order to take advantage of numpy's array paralellisation, methods taking spatial coordinates as an input follow a **common vectorization convention**.

    In our convention, spatial coordinates are passed as an **array with a shape (,3) or (n, m, ..., 3)**, where the **last axis** is always of **dimension 3** and contains the cartesian coordinates (x, y, z) in the lab frame.

    In return, the array will be an **array of shape (,i) or (n, m, ..., i)**, where the **last axis of dimension i** contains the value of the function to evaluate. For a scalar function, i=1 ; for a vector function, i=3.


.. hint::

    Here is how to generate a position vector with the proper dimensions, and to retrieve the x, y, z components for a vector function. We illustrate this using a ``MagneticField`` object:

    .. code-block:: python

        from atomsmltr.environment import MagneticOffset
        import numpy as np

        # - create a magnetic field object (offset)
        mag_field = MagneticOffset((0,1,0))

        # - prepare the position vector
        # initiate a x,y,z grid with
        # x : -10:10, 100 points
        # y : -5:5,   10 points
        # z : -1:1,   5 points
        # grid.shape = (3, 100, 10, 5)
        grid = np.mgrid[-10:10:100j, -5:5:10j, -1:1:5j]
        # extract X, Y, Z for later (plot for instance)
        X, Y, Z = grid  # X.shape = (100, 10, 5)
        # transpose grid to have a position vector
        # matching our convention
        position = grid.T  # position.shape = (5, 10, 100, 3)

        # - get magnetic field value
        # compute
        B = mag_field.value(position)  # B.shape = (5, 10, 100, 3)
        # get components
        Bx, By, Bz = B.T  # Bx.shape = (100, 10, 5)
