# Generating SynthForest

This page describes how the SynthForest scene collections (paper Sec. 3, Table 2)
are generated and converted into training samples (Sec. 5, supplementary Table 6).

> **Botaniq is required for generation.** Scenes are built from the commercial
> [Botaniq Tree and Grass Library](https://polygoniq.com/3d/botaniq/), which
> cannot be redistributed. Without it, the generation commands print an
> explanatory message and exit; all other parts of the repository (data
> preparation, training, evaluation) work without Botaniq or Blender.

## Requirements

- Blender 4.1.1, selected with `SYNTHFOREST_BLENDER_BIN`.
- A Botaniq installation at `$SYNTHFOREST_ASSET_ROOT/botaniq_full`
  (the directory containing `mapr_index.json`).
- Optional context assets (road, water and building textures) at
  `$SYNTHFOREST_ASSET_ROOT/city_rs/assets`. Without them, context objects fall
  back to procedural materials.

All generation configs live in [`configs/generation`](../configs/generation) and
accept Hydra `key=value` overrides on the command line.

```text
$SYNTHFOREST_ASSET_ROOT/
├── botaniq_full/                         # Botaniq library (not distributed)
├── botaniq_tree_asset_pool/              # step 1
├── terrain_textures/botaniq_particles_5cm/   # step 2
└── city_rs/assets/                       # optional context textures
```

## 1. Tree asset pool

For every Botaniq species and season, the generator pre-builds 50 geometric
variants with allometrically consistent height and crown dimensions. Scenes then
instance meshes from this pool, which makes generation much faster.

```bash
synthforest generate asset-pool
```

## 2. Forest-floor textures

Botaniq ground-cover particle presets are baked into 5 cm terrain textures:

```bash
synthforest generate textures -- --gsd 0.05
```

## 3. Scenes

Each run samples a forest configuration (11 types; see
`src/synthforest/tree_scatter/configs/biomes`) and one of three structural
profiles (`src/synthforest/tree_scatter/configs/target_profiles.yaml`):

| Profile key | Paper name |
| --- | --- |
| `small_medium_crowns` | Small/medium crowns (S/M) |
| `large_closed_canopy` | Large closed canopy (LCC) |
| `hard_negative_sparse_context` | Hard-negative sparse context (HN) |

The profile is selected with `generation.target_profile_weights`. The paper
collections correspond to the following batches (supplementary Table 1). All use
seed 42 and start at scene 1.

| Collection | Profile | Scenes | Config | Extent / GSD |
| --- | --- | ---: | --- | --- |
| Crown | HN | 40 | `tree_scatter_scene` | 124 m / 5 cm |
| Crown | LCC | 40 | `tree_scatter_scene` | 124 m / 5 cm |
| Crown | S/M | 80 | `tree_scatter_scene` | 124 m / 5 cm |
| Crown | S/M (legacy) | 75 | see below | 160 m / 5 cm |
| Height | HN | 150 | `tree_scatter_scene_height` | 256 m / 10 cm |
| Height | LCC | 150 | `tree_scatter_scene_height` | 256 m / 10 cm |
| Height | S/M | 400 | `tree_scatter_scene_height` | 256 m / 10 cm |

Example: the large-closed-canopy crown batch.

```bash
synthforest generate scenes \
  --config-name tree_scatter_scene \
  generation.num_scenes=40 seed=42 \
  'generation.target_profile_weights={large_closed_canopy:1}'
```

and the hard-negative height batch:

```bash
synthforest generate scenes \
  --config-name tree_scatter_scene_height \
  generation.num_scenes=150 seed=42 \
  'generation.target_profile_weights={hard_negative_sparse_context:1}'
```

For GPU rendering add `render.device=GPU render.gpu_backend=CUDA`.

The 75-scene legacy S/M crown batch was produced by an earlier revision of the
generator that sampled Botaniq assets directly instead of using the asset pool
(`scatter.tree_source_mode=botaniq_asset_pack`) and rendered 160 m scenes. It is
included in the released dataset as generated. The other six batches are
reproduced by the commands above.

Each scene directory contains aligned outputs:

| Output | Description |
| --- | --- |
| `images/` | RGB render |
| `height_maps/` | metric canopy height (GeoTIFF, metres) |
| `instance_id_maps_smoothed/` | visibility-processed tree-instance IDs (crown collection) |
| `annotations/` | COCO boxes and instance masks with species metadata (crown collection) |
| `run_summary.json` | per-scene forest type, profile, species and tree counts |

## 4. Training samples

**Crown delineation.** Scenes are tiled into 992 × 992 px tiles with 30 % overlap
(10 % for the legacy batch); empty tiles are kept. The 80/20 train/validation
split is made per scene before tiling:

```bash
synthforest prepare tiles --config-name convert_scenes_to_tiles \
  images_dir=<run>/images annotations_dir=<run>/annotations product_name=<run-name>
synthforest prepare split --tiles-root <tiles>/<run-name> --apply
```

**Canopy height (Syn-20).** Scenes are rendered at 10 cm and downsampled to
20 cm before tiling. RGB is area-averaged; height uses the 90th percentile of
positive canopy heights in each 2 × 2 block, which preserves crown tops. Tiles
are 256 × 256 px without overlap. Syn-20 is trained on the HN and LCC height
batches (300 scenes, ≈7.5k tiles):

```bash
synthforest prepare downsample source_root=<run> output_root=<run>/downsampled_20cm
synthforest prepare tiles --config-name convert_scenes_to_tiles_synthetic_height \
  images_dir=<run>/downsampled_20cm/images height_maps_dir=<run>/downsampled_20cm/height_maps \
  product_name=<run-name>
synthforest prepare split --tiles-root <tiles>/<run-name> --apply
```

The training configs expect the tiles under
`$SYNTHFOREST_OUTPUT_ROOT/synthforest/{crown_delineation,canopy_height}/tiles/<batch>`,
with `<batch>` one of `hard_negative`, `large_closed_canopy`, `small_medium`
(and `small_medium_legacy` for crowns), the layout of the released dataset
(see [data.md](data.md)). Use these names as `product_name` when tiling.

## Dataset release

`synthforest publish dataset --repo-id ORG/NAME` previews the upload of the
scene and tile folders to a Hugging Face dataset repository; add `--execute` to
upload.
