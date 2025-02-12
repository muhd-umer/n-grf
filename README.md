# ARD-R2F

ARD-R2F: Accurate and Robust Dynamic Radio Radiance Fields

_Note: Current name is temporary and subject to change_

# Installation

- Create a virtual environment using `uv`. If you don't have `uv` installed, follow the instructions [here](https://docs.astral.sh/uv/getting-started/installation/).
    ```bash
    uv venv
    source .venv/bin/activate
    ```

- Install the dependencies:
    ```bash
    uv sync
    ```

- Install required submodules:
    ```bash
    git submodule update --init --recursive

    uv pip install -e submodules/simple-knn/ --config-settings editable_mode="compat"
    uv pip install -e submodules/diff-gaussian-rasterization/ --config-settings editable_mode="compat"
    ```