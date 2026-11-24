# Installation

## Python package

SynthForest requires Python 3.12 and a CUDA-capable GPU for training. Install the
package in editable mode from the repository root, choosing the optional
dependency groups you need:

```bash
conda create -n synthforest python=3.12 gdal
conda activate synthforest
pip install -e ".[training,geospatial]"     # training and evaluation
pip install -e ".[training,geospatial,publishing,dev]"   # everything
```

| Extra | Purpose |
| --- | --- |
| `training` | PyTorch, torchvision, Transformers (DINOv3), COCO evaluation, W&B |
| `geospatial` | rasterio, geopandas and the CanopyRS `geodataset` (tiling, raster evaluation, real-data preparation) |
| `publishing` | Hugging Face upload of the generated dataset |
| `dev` | pytest, ruff |

Install a PyTorch build that matches your CUDA version first if the default
wheel does not. The package versions used for the paper are listed in
[`environment/known-good-linux-py312.txt`](../environment/known-good-linux-py312.txt).

GDAL is easiest to obtain from conda-forge. If the system `libstdc++` is older
than the one in the conda environment, set
`export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"`.

## DINOv3

Download the DINOv3 ViT-L/16 (LVD-1689M) weights in Hugging Face format after
accepting the DINOv3 license, and place them at
`$SYNTHFOREST_MODEL_ROOT/dinov3-vitl16`.

## Blender and Botaniq (generation only)

Scene generation runs inside Blender 4.1.1 and requires the commercial Botaniq
library; see [generation.md](generation.md). Neither is needed to train or
evaluate models on the released dataset.

```bash
export SYNTHFOREST_BLENDER_BIN=/path/to/blender-4.1.1-linux-x64/blender
```

## Check the installation

```bash
synthforest doctor
pytest -q
```

The unit tests run on CPU with tiny synthetic fixtures and a fake backbone; they
do not need DINOv3 weights, Blender or any dataset.
