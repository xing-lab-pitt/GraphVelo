About GraphVelo
============

RNA velocities and generalizations emerge as powerful approaches for extracting time-resolved information from high-throughput snapshot single-cell data. Learning the vector field function from gene expression profile x and RNA velocity dx/dt 
can uncover the governing equation that drive cell fate transition `Qiu et al. (Cell) <https://www.sciencedirect.com/science/article/pii/S0092867421015774>`_. Yet, several inherent limitations restrict applying the
approaches to genes not suitable for RNA velocity inference due to complex transcriptional dynamics, low expression, lacking splicing dynamics, or data of non-transcriptomic modality. On the other hand, these hidden factors
will also mislead learning the governing function of cell state transition `Xing. (Physical Biology) <https://iopscience.iop.org/article/10.1088/1478-3975/ac8c16>`_. 

Taking various inferred single cell RNA velocity vectors, e.g. splicing-based, metabolic labeling-based, or lineage tracing-based, as input, GraphVelo takes advantage of the nature of the low-dimensional cell state manifold to: 

1) refine the estimated RNA velocity to satisfy the tangent space requirement; 

2) infer the velocities of non-transcriptomic modalities using RNA velocities.  

GraphVelo thus serves as a plugin that can be seamlessly integrated into existing RNA velocity analysis pipelines, and help process single cell data for downstream cellular dynamics analyses using methods 
such as dynamo `Qiu et al. (Cell) <https://www.sciencedirect.com/science/article/pii/S0092867421015774>`_ and CellRank `Lange et al. (Nature Methods, 2022) <https://www.nature.com/articles/s41592-021-01346-6>`_.