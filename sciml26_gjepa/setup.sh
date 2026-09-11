#!/usr/bin/env bash
# Local one-command setup. On Colab you do not need this: run the first cell of the notebook.
set -euo pipefail

PY=${PYTHON:-python3}
command -v uv >/dev/null || { echo "installing uv..."; curl -LsSf https://astral.sh/uv/install.sh | sh; }

echo "==> creating .venv"
uv venv --python 3.10 .venv

echo "==> installing torch (CUDA 12.8; use --torch-backend=cpu if you have no GPU)"
VIRTUAL_ENV=.venv uv pip install torch --torch-backend="${TORCH_BACKEND:-cu128}"

echo "==> installing the rest"
VIRTUAL_ENV=.venv uv pip install -r requirements.txt

echo "==> registering the Jupyter kernel"
.venv/bin/python -m ipykernel install --user \
    --name sciml-gjepa --display-name "Python (sciml-gjepa)"

echo
echo "Done. Open notebooks/sciml_graph_jepa.ipynb and choose the"
echo "'Python (sciml-gjepa)' kernel:  Kernel > Change kernel."
