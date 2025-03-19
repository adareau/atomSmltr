.. _getting-started:
Installation 🚀
===================

pip
~~~~~~~~~~~~~~~~~~

you can install the latest stable release using ``pip``

🚨 **Not implemented yet > when we go public** 🚨


.. code-block:: bash

    pip install atomsmtlr


install from source
~~~~~~~~~~~~~~~~~~

You can also install ``atomsmltr`` from our github repository_. In our git development workflow, we have three main branches: `main` for stable relases, `testing` for development versions that should work most of the time and `devel` for the implementation of new, more experimental features. If you want to benefit from the latest features, we encourage you to use the `testing` branch.

.. _repository: https://github.com/adareau/atomSmltr

First, clone the repository

.. code-block:: bash

    git clone https://github.com/adareau/atomSmltr.git
    cd atomSmltr
    git checkout testing


We strongly encourage you to use a virtual environment

.. code-block:: bash

    python3 -m venv __venv__
    source ./__venv__/bin/activate


Then you can use pip

.. code-block:: bash

    pip install .


That's it, you should be good to go !!
