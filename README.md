# EAMF: Full-Fold ITSD Reproducibility

Source and recorded experiments for the study "Training-Free Photometric Test-Time Fusion for Traffic-Sign Detection: A Full-Fold ITSD Study" by Hossein Rahimi.

## Scope and findings

The primary EAMF-640 study uses all 2,352 official Fold 1 images with released YOLOv12n weights. It evaluates seven conditions and photometric/geometric aggregation controls. Camera-ready additions include the original-author Zero-DCE checkpoint, a fixed three-epoch gamma-augmentation fine-tune, and official YOLOv9t weights on the complete clean, severe-gamma, and linear-noise conditions. A separate bounded study evaluates 21 one-factor configurations on 234 disjoint Fold 2 validation images, then reports the selected postprocessing control separately on Fold 1; it does not overwrite the primary configuration or establish an untouched confirmation test.

All tables come from executed code and saved predictions. EAMF is not universally stronger than gamma alone or scale-only TTA. The noisy-input result depends on the detector. There is no verified real-night benchmark or independent historical EAMF parameter-selection record. Exact-image overlap in the official partitions and a filtered diagnostic are documented in the records.

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
