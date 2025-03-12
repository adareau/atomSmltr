.. _documentation-polarization:

Polarization conventions
=========================

Laser polarization naming
-----------------------------

As explained in :ref:`the laser propagation section<documentation-laser-propagation>`, the polarization is defined in the **laser frame**. For **linear** polarizations, we define the **vertical** as aligned with the :math:`x^{\prime\prime}` axis, and **horizontal** as aligned with the :math:`y^{\prime\prime}`. For **circular** polarizations, we take the **source point of view convention**, as illustrated below.

.. grid:: 2 2 4 4

    .. grid-item-card::

        **Horizontal (H)**

        Polarization is aligned with the **horizontal** axis :math:`y^{\prime\prime}` in the :ref:`laser frame<documentation-laser-propagation>`.

        .. image:: /_static/images/docs_laser_polarization_horizontal.svg
            :alt: laser horizontal polarization illustration
            :align: center



    .. grid-item-card::

        **Vertical (V)**

        Polarization is aligned with the **vertical** axis :math:`x^{\prime\prime}` in the :ref:`laser frame<documentation-laser-propagation>`.

        .. image:: /_static/images/docs_laser_polarization_vertical.svg
            :alt: laser vertical polarization illustration
            :align: center


    .. grid-item-card::

        **Circular Right (R)**

        Polarization rotates **clockwise** from the **source point of view**.

        .. image:: /_static/images/docs_laser_polarization_circular_right.svg
            :alt: laser circular right polarization illustration
            :align: center


    .. grid-item-card::

        **Circular Left (L)**

        Polarization rotates **anti-clockwise** from the **source point of view**.

        .. image:: /_static/images/docs_laser_polarization_circular_left.svg
            :alt: laser circular left polarization illustration
            :align: center



Quantization convention
---------------------------

In ``atomsmltr``, we always define the quantization axis as aligned with the local magnetic field. This allows to project the laser's polarization state (H, V, R, L), that is defined independently from the magnetic field, onto the π, σ+ and σ- components. In next section, we provide a formalism that allows to calculate this decomposition for an arbitrary set of magnetic field direction and laser polarization, but we start by giving some examples for circular polarizations in the cases where :math:`\vec{u}` and :math:`\vec{B}` are aligned.

When :math:`\vec{u}` and :math:`\vec{B}` are aligned and co-propagating, we have:

.. grid:: 2

    .. grid-item-card::
        :text-align: center

        laser (R) | atom (σ+)

        .. image:: /_static/images/docs_laser_quant_RSP.svg
            :align: center
            :width: 150px


    .. grid-item-card::
        :text-align: center

        laser (L) | atom (σ-)

        .. image:: /_static/images/docs_laser_quant_LSM.svg
            :align: center
            :width: 150px


When :math:`\vec{u}` and :math:`\vec{B}` are aligned and counter-propagating, we have:

.. grid:: 2

    .. grid-item-card::
        :text-align: center

        laser (R) | atom (σ-)

        .. image:: /_static/images/docs_laser_quant_RSM.svg
            :align: center
            :width: 150px

    .. grid-item-card::
        :text-align: center

        laser (L) | atom (σ+)

        .. image:: /_static/images/docs_laser_quant_LSP.svg
            :align: center
            :width: 150px




Polarization formalism
------------------------

Here we describe the general polarization formalism we use in ``atomsmltr`` to handle arbitrary laser polarization and its application to the derivation of polarization projection for a given quantization axis.

Polarization vector formalism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. grid:: 2

    .. grid-item::

        .. image:: /_static/images/docs_polarization_vector_sphere.svg
            :align: center
            :width: 200px

    .. grid-item::

        We define a generic laser polarization state using a Bloch-sphere-like formalism. In this formalism, laser polarization is described via a vector :math:`\vec{p}`. This vector is defined using its **polar** and **azimuthal** angles u & v **in the laser frame** (see :ref:`documentation<documentation-laser-propagation>`). This sphere-based representation is illustrated on the left. We remind that in the laser frame, the z axis is aligned with the laser wave-vector :math:`\vec{k}`.


The vector polarization is defined as follow:

.. grid:: 1

    .. grid-item-card::

        **Circular Right polarization** :math:`\ket{R}` is described with a polarization vector pointing along the laser propagation direction, i.e., :math:`\vec{p}` aligned with :math:`\vec{k}`, or in the north pole of the sphere.

    .. grid-item-card::

        **Circular Left polarization** :math:`\ket{L}` is described with a polarization vector :math:`\vec{p}` pointing opposite to the laser propagation direction, i.e. towards the south pole of the sphere.

    .. grid-item-card::

        **Linear polarizations** correspond to a polarization vector :math:`\vec{p}` lying in the equator of the sphere-based

    .. grid-item-card::

        **Horizontal polarization** :math:`\ket{H}` corresponding to a polarization vector aligned with the y axis

    .. grid-item-card::

        **Vertical polarization** :math:`\ket{V}` corresponding to a polarization vector aligned with the x axis.

To be self-consistent, this formalism comes with the following relations for the polarization state :math:`\ket{p}`

.. grid:: 1

    .. grid-item-card::

        .. math::

            \ket{p} = e^{-iv} \cos(u/2) \ket{R} + e^{iv} \sin(u/2)\ket{L}

    .. grid-item-card::

        .. math::

            \ket{V} = \frac{1}{\sqrt{2}}\left( \ket{R} + \ket{L}\right), ~~~~~ \ket{H} = \frac{i}{\sqrt{2}}\left( \ket{L} - \ket{R}\right)

Deriving projections
~~~~~~~~~~~~~~~~~~~~

With this formalism, we can use the :math:`\ket{p}` to compute the decomposition of the polarization on π, σ+ and σ- components using a given magnetic field :math:`\ket{B}` as our quantization axis. In the following we place ourselves in the **laser frame**, in which the z axis is aligned with the laser wave-vector :math:`\vec{k}` ; vector transformation from the lab from to the laser frame are described in the :ref:`laser propagation<documentation-laser-propagation>` section.

The respective orientation of the polarization :math:`\ket{p}` and magnetic field :math:`\ket{B}` vector are given by two set of angles (u, v) and (α, β), as illustrated below.


.. grid:: 2
    :gutter: 4
    :margin: 4

    .. grid-item-card::

        .. image:: /_static/images/docs_polarization_angle.svg
            :align: center
            :width: 200px

    .. grid-item-card::

        .. image:: /_static/images/docs_magfield_angle.svg
            :align: center
            :width: 200px
