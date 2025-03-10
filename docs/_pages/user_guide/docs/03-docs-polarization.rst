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

When :math:`\vec{u}` and :math:`\vec{B}` are aligned and co-propagating, we have:

.. grid:: 2
    :gutter: 4
    :margin: 4 4 0 0

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
    :gutter: 4
    :margin: 4 4 0 0

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




Polarization formalism in ``atomsmtlr``
---------------------------------------
