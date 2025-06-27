.. _documentation-start:

Overview
====================

In this section we provide a series of documents providing definitions and conventions used in ``atomsmtlr``. We also tried, as much as possible, to include those definitions in the docstrings of the associated functions.


.. grid:: 2
    :gutter: 2

    .. grid-item-card::
        :columns: 4

        .. image:: /_static/images/docs_spatial_coords.svg
            :align: center
            :height: 150px

    .. grid-item-card::
        :columns: 8

        :ref:`documentation-spatial-coordinates` - presentation of the convention we use for the definition of **spatial coordinates**, in particular in the context of function vectorization to harness the optimized array operations of Numpy.

    .. grid-item-card::
        :columns: 4

        .. image:: /_static/images/docs_laser_frame.svg
            :align: center
            :height: 150px

    .. grid-item-card::
        :columns: 8

        :ref:`documentation-laser-propagation` - presentation of the convention we use for the **laser propagation**, as implemented in the ``LaserBeam`` class

    .. grid-item-card::
        :columns: 4

        .. image:: /_static/images/docs_polarization_vector_sphere.svg
            :align: center
            :height: 150px

    .. grid-item-card::
        :columns: 8

        :ref:`documentation-polarization` - presentation of the convention we use to define **laser polarizations** and their projections on a given quantization axis.
