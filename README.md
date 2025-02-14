# ARD-R2F

ARD-R2F: Accurate and Robust Dynamic Radio Radiance Fields

_Note: Current name is temporary and subject to change_

# Installation

- Clone the repository:
    ```bash
    git clone https://github.com/muhd-umer/ard-r2f.git
    git submodule update --init --recursive
    cd ard-r2f
    ```

- Create a virtual environment using `uv`. If you don't have `uv` installed, follow the instructions [here](https://docs.astral.sh/uv/getting-started/installation/).
    ```bash
    uv venv
    source .venv/bin/activate
    ```

- Install the dependencies:
    ```bash
    uv sync
    ```

- If installation of submodules fails, install them manually as (remove the `--editable` flag if not needed):
    ```bash
    uv add --editable "simple-knn @ ./submodules/simple-knn"
    uv add --editable "diff-gaussian-rasterization @ ./submodules/diff-gaussian-rasterization"
    ```