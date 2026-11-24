# Data, models and paths

## Location variables

All paths in the configs are expressed relative to a few environment variables.
Unset variables fall back to directories inside the repository.

| Variable | Contents | Default |
| --- | --- | --- |
| `SYNTHFOREST_DATA_ROOT` | real benchmark data and appearance-adaptation references | `data/` |
| `SYNTHFOREST_OUTPUT_ROOT` | SynthForest scenes and tiles, training runs, benchmarks | `outputs/` |
| `SYNTHFOREST_MODEL_ROOT` | DINOv3 weights and SynthForest checkpoints | `models/` |
| `SYNTHFOREST_ASSET_ROOT` | Botaniq library and generation assets (generation only) | `data/assets/` |
| `SYNTHFOREST_BLENDER_BIN` | Blender 4.1.1 executable (generation only) | — |

`synthforest doctor` prints the current values.

## Models

```text
$SYNTHFOREST_MODEL_ROOT/
├── dinov3-vitl16/                 # DINOv3 ViT-L/16 (LVD-1689M), Hugging Face format
└── synthforest/                   # SynthForest checkpoints
    ├── syn20.pth                  # Syn-20 height model
    ├── maskrcnn_syn.pth           # synthetic Mask R-CNN
    ├── maskrcnn_syn_hf.pth        # + height fusion
    ├── maskrcnn_syn_mt.pth        # + Mean Teacher
    └── maskrcnn_syn_hf_mt.pth     # + height fusion + Mean Teacher
```

DINOv3 weights are available from Meta under the DINOv3 license. The released
SynthForest checkpoints contain only the trained parameters (the frozen DINOv3
backbone is loaded from `dinov3-vitl16`):

```bash
hf download novoselova/synthforest --local-dir $SYNTHFOREST_MODEL_ROOT/synthforest
```

When training yourself, copy the final checkpoint of each experiment to these
names instead (see [reproduction.md](reproduction.md)).

## SynthForest

Download the released dataset from Hugging Face (or generate it, see
[generation.md](generation.md)):

```bash
hf download novoselova/synthforest --repo-type dataset \
  --local-dir $SYNTHFOREST_OUTPUT_ROOT/synthforest
```

```text
$SYNTHFOREST_OUTPUT_ROOT/synthforest/
├── crown_delineation/
│   ├── scenes/{hard_negative,large_closed_canopy,small_medium,small_medium_legacy}/
│   └── tiles/{hard_negative,large_closed_canopy,small_medium,small_medium_legacy}/
├── canopy_height/
│   ├── scenes/{hard_negative,large_closed_canopy,small_medium}/   # 10 cm
│   └── tiles/{hard_negative,large_closed_canopy,small_medium}/    # 20 cm
└── splits/{crown_delineation,canopy_height}.json
```

Batch folders are named after the structural profile (S/M, LCC, HN). Each tile
folder contains one sub-folder per scene with `tiles/{train,valid}/`, the COCO
annotations (crowns) or `height_maps/{train,valid}/` (canopy height); the
train/valid assignment is made per scene and listed in `splits/`.

## Real canopy-height benchmarks

| Dataset | Command | Prepared location |
| --- | --- | --- |
| France (IGN Ortho Express + LiDAR HD MNH) | `synthforest data prepare france discover`, then `... download` | `CHM_rasters_20cm/France_CHM_OrthoExpress-20cm_MNH-50cm/` |
| Virtual Shizuoka (2025 airborne LiDAR) | `synthforest data prepare virtual-shizuoka` | `CHM_rasters_20cm/virtual-shizuoka_CHM/` |
| NEON (AOP RGB + Ecosystem Structure CHM) | `synthforest data download neon`, then `synthforest data prepare neon` | `NEON_CHM_1m_real_all/` |
| São Paulo (GeoSampa nDSM) | place the rasters, then `synthforest data prepare sao-paulo` | `CHM_rasters_50cm/sao_paulo/benchmark/` |

The France and Virtual Shizuoka preparation writes full-scene rasters
(`scenes/`) and benchmark rasters (`benchmark/*_RGB.tif`, `*_CHM_matchRGB.tif`).
The scenes are then tiled into 256 px tiles and split at scene level into a
validation set and the test set of Table 3(a):

```bash
synthforest prepare tiles --config-name convert_scenes_to_tiles_france_height
synthforest prepare tiles --config-name convert_scenes_to_tiles_shizuoka_height
synthforest data split height --dataset-preset france --apply
synthforest data split height --dataset-preset shizuoka --apply
```

The split moves validation rasters to `validation/` and keeps test rasters in
`benchmark/`. NEON uses the same command with `--dataset-preset neon`.

The São Paulo benchmark is derived from the LiDAR products of the
[GeoSampa portal](https://geosampa.prefeitura.sp.gov.br/PaginasPublicas/_SBC.aspx)
of the City of São Paulo, following Song et al. (ISPRS J. Photogramm. Remote
Sens. 2026). Download the orthophoto and nDSM for the benchmark scene, export
them as 50 cm GeoTIFFs in EPSG:31983, and place the `*_RGB.tif` / `*_CHM.tif`
pair in the directory above.

### Appearance-adaptation references for Syn-20

Syn-20 adapts synthetic images towards unlabeled RGB tiles from all four height
benchmarks (up to 1,200 images in total; paper Sec. 5.1). Only RGB is used, never
the reference heights. Sample 300 tiles per domain:

```bash
D=$SYNTHFOREST_DATA_ROOT
synthforest prepare tiles --config-name convert_scenes_to_tiles_sao_paulo_height
synthforest data prepare references --src-root $D/CHM_tiles_20cm --split valid \
  --datasets france_mnh_orthoexpress_height_50cm --dst-root $D/domain_adaptation/france \
  --sample-size 300 --allow-fewer
synthforest data prepare references --src-root $D/CHM_tiles_20cm --split valid \
  --datasets virtual_shizuoka_height_50cm --dst-root $D/domain_adaptation/virtual_shizuoka \
  --sample-size 300 --allow-fewer
synthforest data prepare references --src-root $D --split valid \
  --datasets NEON_CHM_1m_real_eval --dst-root $D/domain_adaptation/neon \
  --sample-size 300 --allow-fewer
synthforest data prepare references --src-root $D/CHM_tiles_50cm --split train \
  --datasets sao_paulo --dst-root $D/domain_adaptation/sao_paulo \
  --sample-size 300 --allow-fewer
```

For the Real-NEON baseline, the NEON training sites are tiled with
`synthforest prepare tiles --config-name convert_scenes_to_tiles_neon_height`.

## Real crown-delineation benchmarks

The eight crown benchmarks (BAMForests, QuebecTrees, NeonTreeEvaluation,
OAM-TCD, SelvaMask, SelvaBox, BCI50ha, Detectree2) are used in the standardized
[CanopyRS](https://github.com/hugobaudchon/CanopyRS) preprocessing (tiles, COCO
annotations and folds). Place the CanopyRS datasets at
`$SYNTHFOREST_DATA_ROOT/canopyrs/` and download the ground-truth GeoPackages
needed for raster-level RF1 with

```bash
synthforest data download canopyrs --config crowns/nms/maskrcnn
synthforest data download canopyrs --datasets QuebecTrees Detectree2 NeonTreeEvaluation \
  SelvaBox OAM-TCD BCI50ha BAMForest SelvaMask --fold test
```

The unlabeled appearance-adaptation references for crown models are a random
sample of 2,000 tiles from the training folds of the crown benchmarks:

```bash
synthforest data prepare references   # canopyrs/*/*/tiles/train -> domain_adaptation/crowns
```

## Summary of the expected layout

```text
$SYNTHFOREST_DATA_ROOT/
├── canopyrs/                                   # CanopyRS crown benchmarks
├── domain_adaptation/                          # unlabeled RGB references
│   └── {crowns,france,neon,sao_paulo,virtual_shizuoka}/
├── CHM_rasters_20cm/
│   ├── France_CHM_OrthoExpress-20cm_MNH-50cm/{scenes,benchmark,validation}/
│   └── virtual-shizuoka_CHM/{scenes,benchmark,validation}/
├── CHM_rasters_50cm/sao_paulo/benchmark/
├── CHM_tiles_50cm/sao_paulo/                   # São Paulo reference tiles
├── CHM_tiles_20cm/                             # France / Shizuoka validation tiles
├── NEON_CHM_1m_real_all/{train,benchmark}/
├── NEON_CHM_1m_real_eval/                      # NEON validation tiles
└── NEON_CHM_1m_real_tiles/                     # NEON training tiles (Real-NEON)
```

All datasets remain under their original licenses and access conditions.
