.. graphvelo documentation master file, created by
   sphinx-quickstart on Wed Dec  25 21:21:35 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the GraphVelo documentation.
======================================
**GraphVelo** is a graph-based machine learning procedure that uses RNA velocities inferred from existing methods as input and infers velocity vectors lie in the tangent space of the low-dimensional manifold formed by the single cell data.

Key Features
------------

- Refine the velocity vectors estimated by any methods (e.g. splicing-based, metabolic labeling-based, pseudotime-based, lineage tracing-based etc.) to the data manifold
- Infer modality dynamics goes beyond splicing events 
    - Transcription rate of gene without intron or undergoing alternative splicing
    - Change rate of chromatin openess
    - More to Be Explored
- Serve as a plugin that can be seamlessly integrated into existing RNA velocity analysis pipelines
- Analyze dynamical systems in the context of multi-modal single cell data

Getting Started with GraphVelo
-----------------------------
- Let's get start with our `quick_start_guide demo <graphvelo_notebooks/tutorials/tutorial_for_scvelo.ipynb>`_.
- Contribute to the project on `github`_.

Installation
------------
You need to have Python 3.8 or newer installed on your system. 

To create and activate a new environment:

.. code-block:: bash
    conda create -n graphvelo python=3.8
    conda activate graphvelo

Install graphvelo via pip:

.. code-block:: bash
    pip install graphvelo

.. toctree::
    :caption: General
    :maxdepth: 1
    :hidden:

    API
    about
    tutorials


.. toctree::
    :caption: Gallery
    :maxdepth: 2
    :hidden:

    graphvelo_notebooks/tutorials/index


.. _gitHub: https://github.com/xing-lab-pitt/GraphVelo