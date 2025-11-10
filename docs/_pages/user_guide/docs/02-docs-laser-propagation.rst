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



.. grid:: 2
    :gutter: 2

    .. grid-item::

        **Fig: Laser propagation axis definition**

        the laser propagation is defined by a direction vector :math:`\vec{u}`. The vector can be parametrized using two angles, θ and 𝜙, corresponding to the polar and azimuthal angles of :math:`\vec{u}` in the lab frame.


    .. grid-item::

        .. image:: /_static/images/docs_laser_direction.svg
            :alt: laser direction illustration
            :align: center
            :height: 200px


Below is an example of how to use those two conventions to declare a laser beam propagation along the **+z** direction.

.. code-block:: python

    from atomsmltr.environment import GaussianLaserBeam
    # beam propagating along z
    # option 1 > unit_vector = (0,0,1)
    beam = GaussianLaserBeam(direction=(0,0,1), direction_type="vector")
    # option 1 > θ = 0, 𝜙 =0
    beam = GaussianLaserBeam(direction=(0,0), direction_type="thetaphi")




Laser frame
------------

Definition
^^^^^^^^^^

Some laser properties, such as the laser :ref:`polarization<documentation-polarization>`, are defined **in the laser frame**. In the following, we denote :math:`\mathcal{R}` the lab frame and :math:`\mathcal{R}^{\prime\prime}` the laser frame. Our laser frame definition is such that **the laser propagation unit vector coincides with the** :math:`z^{\prime\prime}` **direction of the laser frame**, *i.e.* :math:`\vec{e}_{z}^{\prime\prime} \equiv \vec{u}`. We denote :math:`(x^{\prime\prime}, y^{\prime\prime}, z^{\prime\prime})|_{\mathcal{R}^{\prime\prime}}` the coordinates in the laser frame.


.. grid:: 2
    :gutter: 2

    .. grid-item::

        **Fig: Laser frame definition**

        In the laser frame, the laser propagates along the :math:`z^{\prime\prime}` axis. We then call :math:`x^{\prime\prime}` the "vertical" axis and :math:`y^{\prime\prime}` the "horizontal" axis.


    .. grid-item::

        .. image:: /_static/images/docs_laser_frame.svg
            :alt: laser direction illustration
            :height: 200px
            :align: center


Since this definition is not sufficient to unambiguously define the laser frame :math:`\mathcal{R}^{\prime\prime}`, here is how we go from the lab frame :math:`\mathcal{R}` to :math:`\mathcal{R}^{\prime\prime}` in two steps:

.. grid:: 1
    :gutter: 2

    .. grid-item-card::

        **STEP 1 :** :math:`\mathcal{R} \to \mathcal{R}^{\prime}`

        First, we perform a rotation around :math:`\vec{e}_z` with an angle :math:`\phi`. The coordinates :math:`(x^{\prime}, y^{\prime}, z^{\prime})|_{\mathcal{R}^{\prime}}` in the new frame can be expressed as

        .. math::

            \left\{
                \begin{array}{lrlrl}
                x^\prime = & & x\cos(\phi) & + &y \sin(\phi)\\
                y^\prime = & - &x\sin(\phi) & + &y \cos(\phi)\\
                z^\prime = & & z & &\\
                \end{array}
            \right.

    .. grid-item-card::

        **STEP 2 :** :math:`\mathcal{R}^{\prime} \to \mathcal{R}^{\prime\prime}`

        Second, we perform a rotation around :math:`\vec{e}_y^\prime` with an angle :math:`\theta`. After this second rotation, :math:`\vec{e}_z^{\prime\prime}` coincides with :math:`\vec{u}`. The coordinates :math:`(x^{\prime\prime}, y^{\prime\prime}, z^{\prime\prime})|_{\mathcal{R}^{\prime\prime}}` in the new frame can be expressed as

        .. math::

            \left\{
                \begin{array}{lrlrl}
                x^{\prime\prime} = & & x^\prime\cos(\theta) & - &z^\prime \sin(\theta)\\
                y^{\prime\prime} = & & y^\prime & &\\
                z^{\prime\prime} = & &x^\prime\sin(\theta) & + &z^\prime \cos(\theta)\\
                \end{array}
            \right.



So, in the end, we have the following relation between the coordinates :math:`(x^{\prime\prime}, y^{\prime\prime}, z^{\prime\prime})|_{\mathcal{R}^{\prime\prime}}` in the :math:`\mathcal{R}^{\prime\prime}` (laser) frame and :math:`(x, y, z)|_{\mathcal{R}}` in the :math:`\mathcal{R}` (lab) frame:

.. grid:: 1
    :gutter: 2

    .. grid-item-card::

        **STEP 1 & 2:**  :math:`\mathcal{R} \to \mathcal{R}^{\prime\prime}`

        .. math::

            \left\{
                \begin{array}{lrlrlrl}
                x^{\prime\prime} = & & x\cos(\theta)\cos(\phi) & + & y \cos(\theta) \sin(\phi) &-& z \sin(\theta)\\
                y^{\prime\prime} = & - & x\sin(\phi) & + & y \cos(\phi) &&\\
                z^{\prime\prime} = & & x\sin(\theta)\cos(\phi) & + & y \sin(\theta) \sin(\phi) &+& z \cos(\theta)\\
                \end{array}
            \right.

        or, likewise

        .. math::

            \left\{
                \begin{array}{lrlrlrl}
                x = & & x^{\prime\prime}\cos(\theta)\cos(\phi) & - & y^{\prime\prime} \sin(\phi) &+& z^{\prime\prime} \sin(\theta)\cos(\phi)\\
                y = &  & x^{\prime\prime}\cos(\theta)\sin(\phi) & + & y^{\prime\prime} \cos(\phi) &+& z^{\prime\prime} \sin(\theta)\sin(\phi) \\
                z = & -& x^{\prime\prime}\sin(\theta) &  &  &+& z^{\prime\prime} \cos(\theta)\\
                \end{array}
            \right.




Illustration in `atomsmltr`
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the following, we define a laser beam with vertical :ref:`polarization<documentation-polarization>`, meaning that its polarization vector is aligned with the :math:`x^{\prime\prime}` axis in the laser frame. Using the methods ``get_polarization_vector_in_lab_frame()`` and ``get_polarization_vector_in_lab_frame()``, we illustrate the frame transformation:

.. code-block:: python

    # initialize a beam with "vertical" polarization
    from atomsmltr.environment import GaussianLaserBeam, Vertical
    beam = GaussianLaserBeam()
    beam.polarization = Vertical()

    # beam propagating along z
    #   > laser frame = lab frame
    beam.direction = (0, 0, 1)
    beam.get_polarization_vector_in_lab_frame() # > (1, 0, 0)
    beam.get_polarization_vector_in_laser_frame() # > (1, 0, 0)


    # beam propagating along -z
    #   > x" → -x
    beam.direction = (0, 0, -1)
    beam.get_polarization_vector_in_lab_frame() # > (-1, 0, 0)
    beam.get_polarization_vector_in_laser_frame() # > (1, 0, 0)

    # beam propagating along y
    #   > 𝜙 = π/2 ; θ = π/2
    #   > x" → -z
    beam.direction = (0, 1, 0)
    beam.get_polarization_vector_in_lab_frame() # > (0, 0, -1)
    beam.get_polarization_vector_in_laser_frame() # > (1, 0, 0)

The ``LaserBeam`` class also includes a set of hidden methods to convert coordinates or vectors between the lab and laser frames:

- ``_convert_coordinates_to_laser_frame()_convert_coordinates_to_laser_frame()``
- ``_convert_vector_to_lab_frame()``
- ``_convert_vector_to_laser_frame()``
