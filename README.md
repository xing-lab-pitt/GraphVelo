# GraphVelo allows for accurate inference of multimodal omics velocities and molecular mechanisms for single cells

[![Supported Python versions](https://img.shields.io/badge/python-3.8-blue)](https://python.org)

<img src="https://github.com/xing-lab-pitt/GraphVelo/blob/main/docs/source/_static/img/framework_fig.png" alt="GraphVelo" width="800" />

**GraphVelo** is a graph-based machine learning procedure that uses RNA velocities inferred from existing methods as input and infers velocity vectors that lie in the tangent space of the low-dimensional manifold formed by the single-cell data.

## Key Features

- Refine the velocity vectors estimated by any methods (e.g., splicing-based, metabolic labeling-based, pseudotime-based, lineage tracing-based, etc.) to the data manifold
- Infer modality dynamics that go beyond splicing events
    - Transcription rate of genes without introns or undergoing alternative splicing
    - Change rate of chromatin openness
    - More to be explored
- Serve as a plugin that can be seamlessly integrated into existing RNA velocity analysis pipelines
- Analyze dynamical systems in the context of multi-modal single-cell data

## Getting Started with GraphVelo

Check the pipeline of RNA velocity estimation and you will find the niche of `graphvelo`:

<img src="https://github.com/xing-lab-pitt/GraphVelo/blob/main/docs/source/_static/img/graphvelo_pipeline.png" alt="GraphVelo" width="800" />

Now let's get started with our [Tutorials](https://graphvelo.readthedocs.io/en/latest/index.html).

## Installation

You need to have Python 3.8 or newer installed on your system. 

To create and activate a new environment
```bash
conda create -n graphvelo python=3.8
conda activate graphvelo
```

Install via pip:
```bash
pip install graphvelo
```

## Citing GraphVelo

Please see our [manuscript](https://www.biorxiv.org/content/10.1101/2024.12.03.626638v1) for detailed explanation. 
If you find GraphVelo useful for your research, please consider citing our work as follows:

```
@article {Chen2024.12.03.626638,
	author = {Chen, Yuhao and Zhang, Yan and Gan, Jiaqi and Ni, Ke and Chen, Ming and Bahar, Ivet and Xing, Jianhua},
	title = {GraphVelo allows inference of multi-modal single cell velocities and molecular mechanisms},
	year = {2024},
	doi = {10.1101/2024.12.03.626638},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/12/07/2024.12.03.626638},
	journal = {bioRxiv}
}

```