# Reproducing the paper

This page lists the commands behind every result in the paper. Configs are
referenced by their path below [`configs/`](../configs) without the `.yaml`
suffix. Each run writes to
`$SYNTHFOREST_OUTPUT_ROOT/{heights,detectors}/training/<run-name>/` with the
resolved config, metrics, `checkpoints/{best,last}.pth` and, if enabled, the
post-training benchmark.

All models use a frozen DINOv3 ViT-L/16 backbone, AdamW, mixed precision, an
effective batch size of 4 and seed 42; each experiment runs on a single GPU.
W&B logging is enabled in the configs but optional; set
`WANDB_MODE=disabled` to turn it off.

## Experiment dependencies

Some experiments start from the final checkpoint (`last.pth`) of another one.
The configs read these checkpoints from `$SYNTHFOREST_MODEL_ROOT/synthforest/`;
copy (or link) each checkpoint there after the corresponding run, or use the
released weights. Real validation sets are evaluated during training for
monitoring only; they are never used to select checkpoints.

```text
height/train_syn20 ──────────────► syn20.pth (last.pth)
crowns/training/synthetic ───────► maskrcnn_syn.pth (last.pth) ──► crowns/training/synthetic_mt
crowns/training/synthetic_hf ────► maskrcnn_syn_hf.pth (last.pth) ─► crowns/training/synthetic_hf_mt
crowns/training/synthetic_hf_mt ─► maskrcnn_syn_hf_mt.pth (last.pth) ► crowns/finetuning/*
```

`syn20.pth` is required by every height-fusion config (`*_hf*`, `finetuning/*`).

## Table 4: canopy-height estimation

Train Syn-20 (15,000 iterations, 20 cm input, HM/PDA/FDA appearance adaptation):

```bash
synthforest train height --config height/train_syn20
cp <syn20-run>/checkpoints/last.pth $SYNTHFOREST_MODEL_ROOT/synthforest/syn20.pth
```

Evaluate on the four real benchmarks at the native grid of each reference height
product:

```bash
for benchmark in france_raster neon_raster sao_paulo_raster shizuoka_raster; do
  synthforest evaluate height --config height/benchmark/$benchmark \
    --run-folder <syn20-run> --checkpoint-name last.pth
done
```

The Real-NEON baseline is trained with `synthforest train height --config
height/train_real_neon` and evaluated in the same way. RS3DAda numbers are taken
from the SynRS3D release (Song et al., NeurIPS 2024).

| Dataset | Model | MAE | MAE>10 m | RMSE |
| --- | --- | ---: | ---: | ---: |
| France | Real-NEON | 3.59 | 7.65 | 5.33 |
| | RS3DAda | 5.29 | 11.51 | 7.02 |
| | **Syn-20** | **2.58** | **4.76** | **4.08** |
| NEON | Real-NEON | **2.69** | **5.98** | **3.77** |
| | RS3DAda | 4.14 | 9.19 | 5.27 |
| | Syn-20 | 4.33 | 7.80 | 5.65 |
| São Paulo | Real-NEON | **3.33** | **3.82** | **4.97** |
| | RS3DAda | 6.47 | 10.42 | 8.98 |
| | Syn-20 | 4.79 | 6.30 | 6.90 |
| Virtual Shizuoka | Real-NEON | 9.11 | 9.46 | 11.33 |
| | RS3DAda | 14.90 | 15.64 | 15.90 |
| | **Syn-20** | **8.26** | **8.43** | **9.90** |

## Table 5: crown delineation

All synthetic crown models train for 8,000 iterations on the SynthForest crown
tiles. Height fusion (HF) adds Syn-20 predictions at P8 and P16; the
detector-aligned height maps are computed automatically on first use
(`synthforest prepare height-maps` computes them explicitly). Mean Teacher (MT)
adapts a trained detector with unlabeled BAMForests and SelvaMask training
imagery.

```bash
# Synthetic supervision
synthforest train crowns --config crowns/training/synthetic
cp <run>/checkpoints/last.pth $SYNTHFOREST_MODEL_ROOT/synthforest/maskrcnn_syn.pth

# + height fusion
synthforest train crowns --config crowns/training/synthetic_hf
cp <run>/checkpoints/last.pth $SYNTHFOREST_MODEL_ROOT/synthforest/maskrcnn_syn_hf.pth

# + Mean Teacher, without and with height fusion
synthforest train crowns --config crowns/training/synthetic_mt
synthforest train crowns --config crowns/training/synthetic_hf_mt
cp <run>/checkpoints/last.pth $SYNTHFOREST_MODEL_ROOT/synthforest/maskrcnn_syn_hf_mt.pth

# Real-supervised baselines (6,500 iterations)
synthforest train crowns --config crowns/training/real_selvamask
synthforest train crowns --config crowns/training/real_bamforest
synthforest train crowns --config crowns/training/real_selvamask_bamforest
```

After training, each run is benchmarked with
`crowns/benchmark/maskrcnn` on the test folds of all eight datasets
(bounding-box mAP50:95 / mAR50:95, final checkpoint). Raster-level RF1@75 is
computed after merging overlapping tiles with NMS, whose thresholds are tuned on
QuebecTrees and SelvaMask validation rasters:

```bash
synthforest evaluate crowns-nms \
  --nms-search-config crowns/nms/maskrcnn --benchmark-config crowns/benchmark/maskrcnn \
  --run-folder <run> --checkpoint-name last.pth
```

| Model | Train | Macro mAP | MT-HO mAP | Macro RF1@75 |
| --- | --- | ---: | ---: | ---: |
| Mask R-CNN | SM | 13.89 | 13.67 | 21.88 |
| Mask R-CNN | BF | 19.98 | 19.40 | 24.02 |
| Mask R-CNN | BF+SM | 20.35 | 19.25 | 24.34 |
| Mask R-CNN | Syn | 15.11 | 15.36 | 17.28 |
| + HF | Syn | 15.49 | 15.81 | **17.77** |
| + MT | Syn + BF<sub>u</sub> + SM<sub>u</sub> | 15.65 | 15.76 | 16.39 |
| + HF + MT | Syn + BF<sub>u</sub> + SM<sub>u</sub> | **15.91** | **16.08** | 16.77 |

Per-dataset values are in Table 5 of the paper.

## Figure 6: label-efficient fine-tuning

Fine-tune the final HF+MT model (Synth-init) and train matched real-only models
on fractions of the labeled BAMForests and SelvaMask training sets. Each run uses
6,500 iterations with HM/PDA appearance adaptation; fractions are sampled with
seed 42, independently per dataset in the combined setting. Fine-tuning reuses the real-image height maps written
by the HF+MT run.

```bash
for source in real_bamforest real_selvamask real_selvamask_bamforest; do
  for fraction in 0.01 0.05 0.10 0.15 0.20 0.25 0.50 0.75 1.00; do
    synthforest finetune crowns --config crowns/finetuning/$source \
      --real-data-fraction $fraction --real-data-seed 42
    synthforest train crowns --config crowns/training/$source \
      --train-data-fraction $fraction --train-data-seed 42
  done
done
```

In-domain mAP is measured on the fine-tuning dataset(s) and out-of-domain mAP
is the unweighted mean over the remaining benchmarks. The full numbers are given
in supplementary Table 11; for example, 15 % of the BAMForests labels reach
99.2 % of the full-data real-only in-domain mAP.
