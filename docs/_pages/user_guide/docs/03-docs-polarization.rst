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

**The problem is set as follow:**

* in the *laser basis* (x,y,z), the polarization state can be written :math:`\ket{p} = e^{-iv} \cos(u/2) \ket{R} + e^{iv} \sin(u/2)\ket{L}`, which can be decomposed onto :math:`\ket{x}=\ket{V}` and :math:`\ket{y}=\ket{H}` using :math:`\ket{R}=(\ket{x}+i\ket{y})/\sqrt{2}` and :math:`\ket{L}=(\ket{x}-i\ket{y})/\sqrt{2}`.


* in the *magnetic field* basis (x',y',z'), where z' is aligned with B, we can define the polarization states :math:`\ket{\pi} = \ket{z^\prime}` and :math:`\ket{\sigma\pm} = (\ket{x^\prime}\pm i\ket{y^\prime})/\sqrt{2}`


Now using the formulae above it is possible to (1) decompose the polarization state :math:`\ket{p}` onto :math:`\ket{x}` and :math:`\ket{x}` and (2) compute its projection onto the :math:`\ket{pi}` and :math:`\ket{\sigma\pm}` states.


Here is what we get:


.. grid:: 1

    .. grid-item-card::

        .. math::
            \begin{align}
            \ket{p} & =  \left\{ e^{-iv} \cos(u/2) + e^{iv} \sin(u/2) \right\} / \sqrt{2}\,\ket{x} \\
             & + i\left\{ e^{-iv} \cos(u/2) - e^{iv} \sin(u/2) \right\}  / \sqrt{2}\,\ket{y}
            \end{align}



    .. grid-item-card::

        .. math::
            \begin{align}
            \ket{x} & = \left( \cos\beta\cos\alpha + i \sin\beta\right) / \sqrt{2} \,\ket{\sigma+}\\
                    & + \left( \cos\beta\cos\alpha - i \sin\beta\right)/ \sqrt{2} \, \ket{\sigma-} \\
                    & + \cos\beta\sin\alpha \ket{\pi}
            \end{align}

    .. grid-item-card::

        .. math::
            \begin{align}
            \ket{y} & = \left( \sin\beta\cos\alpha - i \cos\beta\right) / \sqrt{2} \,\ket{\sigma+}\\
                    & + \left( \sin\beta\cos\alpha + i \cos\beta\right)/ \sqrt{2} \, \ket{\sigma-} \\
                    & + \sin\beta\sin\alpha \ket{\pi}
            \end{align}


With that, it is possible to compute :math:`\braket{p|\pi}`, :math:`\braket{p|\sigma+}` and :math:`\braket{p|\sigma-}`.


Implementation in atomsmltr
---------------------------

Here is a short code example illustring how all of the above is implemented in ``atomsmtlr``:

.. code-block:: python

    from numpy import pi
    from atomsmltr.environment import (
        GaussianLaserBeam,
        Vertical,
        Horizontal,
        Linear,
        CircularLeft,
        CircularRight,
        Vector,
    )

    # - Setting up polarizations
    beam = GaussianLaserBeam()
    beam.polarization = Vertical()
    beam.polarization = Horizontal()
    beam.polarization = Linear(angle=pi / 4)
    beam.polarization = CircularLeft()
    beam.polarization = CircularRight()
    beam.polarization = Vector((0, 0, -1))

    # - Getting info
    beam.polarization.print_info()
    beam.get_polarization_vector_in_lab_frame()
    beam.get_polarization_vector_in_laser_frame()


    # - Compute projections
    B = (0,1,0)
    # 〈Ψ|π⟩, 〈Ψ|σ+⟩ and 〈Ψ|σ-⟩ in an array form
    beam.get_polarization_quant_amplitude(B)
    # same, output as a dict
    beam.get_polarization_quant_amplitude_dict(B)
    # |〈Ψ|π⟩|**2 , |〈Ψ|σ+⟩|**2 and |〈Ψ|σ-⟩|**2 in an array form
    beam.get_polarization_quant(B)
    # same, output as a dict
    beam.get_polarization_quant_dict(B)
