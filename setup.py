import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="graphvelo",
    version="0.1.0",
    author="{Yuhao Chen, Yan Zhang, Jiaqi Gan, Ke Ni, Ming Chen, Ivet Bahar, Jianhua Xing",
    author_email="xing1@pitt.edu",
    description="Estimation of manifold-constrained velocity and transform vectors across representations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xing-lab-pitt/GraphVelo/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        'typing-extensions>=4.12.2'
    ],
    extras_require={
        'dev': [
            'pytest>=8.3.4',
        ]
    }
)