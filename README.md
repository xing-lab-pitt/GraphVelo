# GraphVelo allows inference of multi-modal single cell velocities and molecular mechanisms

GraphVelo is a graph-based machine learning procedure that uses RNA velocities inferred from existing methods as input and infers velocity vectors lie in the tangent space of the low-dimensional manifold formed by the single cell data. GraphVelo preserves vector magnitude and direction information during transformations across different data representations. If you use our tool in your own work, please cite it as

```
@article {Chen2024.12.03.626638,
	author = {Chen, Yuhao and Zhang, Yan and Gan, Jiaqi and Ni, Ke and Chen, Ming and Bahar, Ivet and Xing, Jianhua},
	year = {2024},
	doi = {10.1101/2024.12.03.626638},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/12/07/2024.12.03.626638},
	journal = {bioRxiv}
}
```

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