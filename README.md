# ARD-R2F

ARD-R2F: Accurate and Robust Dynamic Radio Radiance Fields

_Note: Current name is temporary and subject to change_

# Installation

Create a virtual environment using `mamba`:

```bash
mamba env create -n ard-r2f python=3.12
```

Activate the environment:

```bash
mamba activate ard-r2f
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Install submodules:

```bash
git submodule update --init --recursive

pip install -e submodules/simple-knn/
pip install -e submodules/diff-gaussian-rasterization/
```