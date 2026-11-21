# EAMF: Full-Fold ITSD Reproducibility

[![Paper](https://img.shields.io/badge/Paper-MVIP%202026-purple.svg)](#scope-and-findings)

Source and recorded experiments for the study "Training-Free Photometric Test-Time Fusion for Traffic-Sign Detection: A Full-Fold ITSD Study" by Hossein Rahimi.

[Portfolio case study](https://hoosseinrahimi.github.io/#research-at-a-glance) · [Versioned evidence release](https://github.com/HoosseinRahimi/eamf-itsd-mvip2026/releases/tag/paper80-camera-ready-20261028) · [Citation](CITATION.cff)

## Abstract

Photometric test-time adaptation can improve traffic-sign detection under illumination shift, but reported comparisons are often difficult to reproduce across folds, detectors and corruption conditions. This study evaluates a training-free EAMF-640 pipeline on all 2,352 official Fold 1 ITSD images using released YOLOv12n weights, with additional camera-ready controls for Zero-DCE, gamma fine-tuning and YOLOv9t. The study reports executed full-fold results, paired bootstrap summaries, input hashes and timing lineage, and explicitly separates a bounded Fold 2 diagnostic from the primary configuration.

## Results at a glance

Official AP from `records/full_gpu640_analysis_v2/all_methods.csv`, 2,352 images per condition. Values are percentages; bold marks the best mAP50:95 among the four shown methods.

| Condition | Baseline | Gamma | EAMF | TTA3 |
|---|---:|---:|---:|---:|
| Clean | 95.91 / 88.67 | 95.87 / 88.55 | 95.98 / 88.75 | **96.29 / 89.16** |
| Severe gamma 3.0 | 92.24 / 83.87 | **96.09 / 88.83** | 95.91 / 88.03 | 91.58 / 83.33 |
| Linear noise | 91.20 / 83.45 | 84.83 / 76.15 | 90.76 / 82.33 | **91.31 / 83.38** |

Cells show `mAP50 / mAP50:95`. Full per-class AP, paired bootstrap draws, GPU timing rows and method variants are in `records/` and in the versioned evidence release.

## Scope and findings

The primary EAMF-640 study uses all 2,352 official Fold 1 images with released YOLOv12n weights. It evaluates seven conditions and photometric/geometric aggregation controls. Camera-ready additions include the original-author Zero-DCE checkpoint, a fixed three-epoch gamma-augmentation fine-tune, and official YOLOv9t weights on the complete clean, severe-gamma, and linear-noise conditions. A separate bounded study evaluates 21 one-factor configurations on 234 disjoint Fold 2 validation images, then reports the selected postprocessing control separately on Fold 1; it does not overwrite the primary configuration or establish an untouched confirmation test.

All tables come from executed code and saved predictions. EAMF is not universally stronger than gamma alone or scale-only TTA. The noisy-input result depends on the detector. There is no verified real-night benchmark or independent historical EAMF parameter-selection record. Exact-image overlap in the official partitions and a filtered diagnostic are documented in the records.

## Limitations

- EAMF does not dominate gamma under severe gamma or TTA3 on clean/noisy inputs in the full-fold table above.
- The linear-noise result is detector-dependent; do not transfer it to other backbones without re-running the protocol.
- The Fold 2 study uses 234 disjoint validation images and a separately reported postprocessing control. It is a bounded diagnostic, not an untouched confirmation test.
- GPU timings start from preloaded 640-pixel inputs and exclude decoding, shared resize, corruption construction, camera capture and batched throughput.
- No publication DOI or proceedings identifier is invented here.

## Reproduce

Use Python 3.12.3 and the CPU or CUDA 12.1 lockfile in `code/`. Follow `code/REPRODUCE.md` for exact commands and fixed protocols. Download official ITSD version 2 data and released weights from https://doi.org/10.6084/m9.figshare.28943957.v2 and verify the documented checksums. Fetch the original-author Zero-DCE files with the pinned `fetch_zero_dce.py` helper; they are not vendored here.

`records/` contains aggregate results, protocols, input manifests, and diagnostics. The accompanying evidence ZIP preserves per-image predictions, matched TP flags, paired bootstrap draws, input hashes, standalone timing rows, and the bounded fine-tune checkpoint. Its top-level `primary/` and `extension/` folders correspond to the two evidence roots in the commands.

The clean-environment check reaggregates the official AP implementation and reproduces 12 fixed sampled images in three conditions for six methods on CPU; it is not a fresh full-dataset inference run. Inspect the stored report for exact tolerances and environment. GPU timings start from preloaded common 640-pixel inputs, exclude native decoding/shared resize/common corruption construction, and are not camera-to-output or batched-throughput measurements.

Historical CPU/native-resolution controls are separate from the new EAMF-640 results. Do not mix old preliminary numbers with revised full-fold values or choose a favorable checkpoint/configuration using test labels.

## Data, software and attribution

See `THIRD_PARTY_NOTICES.md`. Original datasets, official base weights, Python environments, downloaded package wheels, private correspondence, and submission-account information are excluded. No publication DOI or conference-proceedings identifier is invented here.

## Repository and versioned evidence

Repository: https://github.com/HoosseinRahimi/eamf-itsd-mvip2026

Versioned reproducibility release: https://github.com/HoosseinRahimi/eamf-itsd-mvip2026/releases/tag/paper80-camera-ready-20261028
