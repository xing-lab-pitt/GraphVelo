Installation guide
==================

************
Main package
************

The ``graphvelo`` package can be installed via pip:

.. code-block:: bash
    :linenos:

    pip install graphvelo

.. note::
    To avoid potential dependency conflicts, installing within a
    `conda environment <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`__
    is recommended.


*********************
Optional dependencies
*********************

Our ``graphvelo`` package serves as a seamlessly plugin in RNA velocity analyses pipeline 
and uses external tools such as ``scvelo``, ``cellrank``, ``dynamo``.

You may install ``scvelo`` following the official `guide <https://scvelo.readthedocs.io/en/stable/installation.html>`__.
You may install ``cellrank`` following the official `guide <https://cellrank.readthedocs.io/en/latest/installation.html>`__.
You may install ``dynamo`` following the official `guide <https://dynamo-release.readthedocs.io/en/latest/installation.html>`__.

Now you are all set. Proceed to `tutorials <graphvelo_notebooks/tutorials/index.rst>`__ for how to use the ``graphvelo`` package.