# Reproduce the reviewer revision

The primary revision evaluates **all 2,352 official Fold 1 images**. EAMF-640 is an explicit preprocessing revision: every method uses a shared aspect-preserving bilinear resize to longest side 640; synthetic corruption and enhancement follow that resize. Do not compare its new numbers as if they were a rerun of the old native-resolution 117-image protocol.

## Official inputs

Figshare dataset DOI: https://doi.org/10.6084/m9.figshare.28943957 (version 2).

- ITSD-dataset.rar, file 54262415, MD5 `0fb4abd6c919cddedda1ffcc7ab0b41a`.
- Main.zip, file 61727068, MD5 `a0d72e2639d9dc90ba17d3664986a20a`.
- `Main/yolov12_result/Fold1/weights/best.pt`, SHA-256 `fc92a7c61edbbd7edc4dea6ec1fd6992825e230e55e9bd22286285810aeb8b54`.

Extract the outer dataset archive and its Fold-1 archive. The commands below use placeholders for the resulting images, labels, and weight paths. Large official data and weights are not redistributed in this package.

## Environment

Use Python 3.12.3 and the supplied CPU or GPU lockfile. The GPU run uses PyTorch 2.2.2+cu121, torchvision 0.17.2+cu121, Ultralytics 8.3.233, NumPy 1.26.4, OpenCV 4.11.0.86, and ensemble-boxes 1.0.9. The full protocol JSON records the actual environment. Use a separate environment rather than modifying an unrelated Python installation.

Run `python test_revision.py` before starting. The tests cover geometry, deterministic corruption, empty predictions, singleton WBF scores, and equality of the bootstrap AP calculation to the official AP calculation on repeated-image samples.

## Collect complete-fold predictions

```text
python collect_revision.py --images <images> --labels <labels> --weights <best.pt> --output <runs> --device 0 --work-longside 640
```

The default conditions are clean, gamma1.8, gamma2.4, gamma3.0, linear_noise, overexposed, and haze. Use `--device cpu` for CPU. Do not use `--limit` for final evidence. Collection stores each complete image record separately and skips matching records when resumed. It refuses a mismatched output protocol.

If the command host interrupts long jobs, use `collect_bounded_batch.py` with the same arguments plus `--max-new 128`, and repeat until all conditions complete. It records the wrapper hash and stops only after a complete image record is written.

The native-resolution control collector is preserved separately as `collect_native_reference.py`. Its source hash matches the recorded native CPU control protocol. Only its complete clean condition was used as a native-resolution control; it is not mislabeled as a seven-condition experiment.

## Analyze

```text
python analyze_revision.py --input <runs> --output <analysis> --conditions clean gamma1.8 gamma2.4 gamma3.0 linear_noise overexposed haze --bootstrap 1000
```

The default analyzer refuses incomplete conditions. It derives all comparisons from the same per-view predictions, using Ultralytics 8.3.233 matching and AP functions for all methods. Pairing is at image level. Bootstrap replicates resample complete images and recompute dataset AP; they do not average per-image AP. The same resampled image IDs and GT-present class set are used for each compared method. Acquisition-session dependence and rare-class instability remain limitations.

Every view uses detector confidence 0.001, NMS IoU 0.70, and maximum 300 predictions per call. WBF uses IoU 0.55, skip_box_thr 0.001, conf_type=avg, allows_overflow=False. Other controls and exact corruption constants are specified in source/protocol JSON. Threshold sensitivity is descriptive; no new optimum is selected.

The protocol's `parameters_frozen` field refers to keeping the earlier manuscript's constants unchanged during this revision. It is not evidence that historical selection used an independent validation set, and "published" in that legacy metadata wording does not establish formal publication of the earlier draft. The manuscript explicitly records the unknown historical selection procedure.

## Class-preserving scale-only control

After primary collection, add only the missing scale-768 view. Existing scale-512 and original-640 predictions are reused without changing their records:

```text
python collect_multiscale.py --cache <runs> --images <images> --weights <best.pt> --output <extra-scales> --device 0
python analyze_multiscale.py --cache <runs> --extra <extra-scales> --output <multiscale-analysis>
```

This evaluates 512/640/768 TTA on all images for clean, gamma3.0, and linear_noise. Additional views are bound to primary record hashes. The analysis includes 1,000 paired bootstrap replicates against EAMF.

## Timing

Wait until accuracy collection is no longer using the device. Then run:

```text
python benchmark_revision.py --images <images> --weights <best.pt> --output <benchmark.json> --device 0 --work-longside 640 --samples 48 --repeats 3 --conditions clean gamma3.0 linear_noise
```

The timer begins from a preloaded, common working-resolution input. Native decoding, shared resize, and common corruption generation are excluded. Router, enhancement/weight computation, detector preprocessing/inference/NMS/box transfer, and fusion are included. Five warm-up calls and synchronization precede measurement; method order is randomized. FPS is reciprocal mean standalone latency, not batched throughput or camera-to-output speed. Resume is supported with the same output/protocol.

## Figures and manuscript

```text
python make_revision_figures.py --input <runs> --images <images> --output <manuscript>
python build_measured_text.py --analysis <analysis> --cache <runs> --benchmark <benchmark.json> --multiscale <multiscale-analysis> --manuscript <manuscript>
python write_review_response.py --analysis <analysis> --benchmark <benchmark.json> --coverage <coverage.json> --multiscale <multiscale-analysis> --output <review-response.md>
```

The manuscript builder refuses missing, partial, mismatched, or insufficient-bootstrap results. Its generated TeX values and tables are linked to input hashes. Compile `paper.tex` with pdfLaTeX, BibTeX, then pdfLaTeX twice. After the last content change, render and inspect every page.

The evidence archive includes raw predictions, matched TP flags, bootstrap draws, all-method/class tables, diagnostics, and lineage. The code repository is https://github.com/HoosseinRahimi/eamf-itsd-mvip2026 and versioned evidence is indexed under its Releases page. Conference upload is performed separately by the author; code publication does not imply that the final manuscript has been submitted.

## Camera-ready additions (28 October 2026)

The additions use the same primary runs; do not replace them with new values silently. Python environments and large original data/weights are not bundled. For CPU, install the lock with `python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-cpu-lock.txt`. The GPU lock contains its CUDA 12.1 extra index. The documented environment uses Python 3.12.3. CPU/GPU floating-point predictions need not be byte-identical.

Fetch the fixed original-author enhancer independently of the conference website:

```text
python fetch_zero_dce.py --output <zero-dce-upstream>
python collect_zero_dce.py --images <Fold1/images> --weights <YOLOv12-Fold1-best.pt> --cache <runs> --upstream <zero-dce-upstream> --output <zero-predictions> --device 0
python analyze_extension.py --input <zero-predictions> --parent <runs> --parent-analysis <analysis> --method ZeroDCE --output <zero-analysis>
```

The audited original dataset archive also contains `ITSD-dataset/yolov9_weights/best_Fold1.pt`, which loads as YOLOv9t with the identical 118-class mapping. This is distinct from YOLOv12 weights in Main.zip:

```text
python collect_detector_extension.py --images <Fold1/images> --weights <YOLOv9-best_Fold1.pt> --cache <runs> --output <yolov9-predictions> --name YOLOv9t-official --device 0
python analyze_extension.py --input <yolov9-predictions> --parent <runs> --parent-analysis <analysis> --method YOLOv9 --output <yolov9-analysis>
python analyze_extra_comparisons.py --analysis <analysis> --output <gamma-comparison>
```

For the bounded training comparator, extract official Fold-2 images/labels and the original class YAML. The script removes all exact filename/file-hash matches to the Fold 1 test manifest, deduplicates remaining Fold 2 images, and creates deterministic train/validation lists before training. The actual split contains 2,110 train and 234 validation images. It uses three fixed epochs and the last checkpoint, not test-set checkpoint selection:

```text
python train_gamma_comparator.py --fold2 <Fold-2> --test-manifest <runs/manifest.json> --classes <data_labels.yaml> --weights <YOLOv12-Fold1-best.pt> --output <gamma-finetune>
python collect_detector_extension.py --images <Fold1/images> --weights <gamma-finetune/run/weights/last.pt> --cache <runs> --output <finetune-predictions> --name GammaFineTune3Epochs --views original --device 0
python analyze_extension.py --input <finetune-predictions> --parent <runs> --parent-analysis <analysis> --method FineTuneGamma --output <finetune-analysis>
```

Store all extension folders under a common extension root with the names used in `build_final_text.py`. The final timing run must start after GPU accuracy collection/training stops:

```text
python benchmark_finalization.py --images <Fold1/images> --weights <YOLOv12-Fold1-best.pt> --finetune-weights <last.pt> --yolov9-weights <YOLOv9-best_Fold1.pt> --upstream <zero-dce-upstream> --output <extension/benchmark_finalization.json>
python audit_fold_overlap.py --archive <ITSD-dataset.rar> --work <extension/fold_overlap> --fold1-manifest <runs/manifest.json> --analysis <analysis>
python build_final_text.py --primary <primary-evidence-root> --extension <extension-evidence-root> --manuscript <manuscript>
```

The overlap audit finds three redundant copies within Fold 1 and 14 Fold 1 images with exact file matches elsewhere; its filtered diagnostic retains 2,335 images. It does not establish which images the original released checkpoints encountered or exclude near-duplicate/session dependence. Keep this caveat.

`verify_final_evidence.py` independently reaggregates official AP from saved matched records and checks the additional collection/training/timing hashes. `verify_reproduction_smoke.py` reproduces 12 fixed sampled images in three conditions for six methods in a separate CPU environment; its tolerances are declared in source. This sampled check is not a fresh full-dataset inference run.

The local clean CPU environment was installed offline from original uv-cached wheel payloads after an upstream DNS failure. `recover_cached_wheels.py` verifies the original wheel RECORD hashes before repacking. The repacked ZIP hashes differ from original downloaded wheel archives; no fresh-upstream-download claim is made. Standard installations can use the published lockfiles and official package indexes directly.

## Separate validation calibration control

Run this stage before `build_final_text.py`. It preserves the primary EAMF configuration. It uses the deterministic 234-image validation list prepared by `train_gamma_comparator.py` (the `--prepare-only` option prepares that list without training). A predeclared one-factor design evaluates the default plus two alternatives for each of ten parameters. Selection uses only the unrounded equally weighted validation mean mAP50:95 over clean, gamma3.0, and linear_noise. The full Fold 1 was previously inspected, so the subsequent selected-control evaluation is explicitly exploratory.

```text
python calibrate_eamf.py --stage validation --images <Fold2/images> --labels <Fold2/labels> --val-list <gamma-finetune/val.txt> --weights <YOLOv12-Fold1-best.pt> --primary <primary-evidence-root> --output <extension/calibration_validation>
python calibrate_eamf.py --stage test --images <Fold1/images> --weights <YOLOv12-Fold1-best.pt> --primary <primary-evidence-root> --selection <extension/calibration_validation/selection.json> --output <extension/calibrated_predictions>
python analyze_calibrated.py --input <extension/calibrated_predictions> --primary <primary-evidence-root> --output <extension/calibrated_analysis>
python test_calibration_compat.py --validation <extension/calibration_validation> --output <extension/calibration_default_compatibility.json>
```

The actual selection changes only WBF IoU from 0.55 to 0.45; cached unchanged photometric predictions are reused with explicitly recomputed postprocessing. The primary tables and runtime measurement continue to refer to the historical default. Training-free refers to no detector/enhancer weight updates for the fusion method; this optional calibration uses labeled validation data. It does not prove the original parameter-selection history.

For a lightweight audit of the distributed evidence, install the CPU lock, extract the evidence ZIP, and run `python verify_final_evidence.py --primary <extracted/primary> --extension <extracted/extension> --output <audit.json>`. This reaggregates official AP and verifies recorded training/timing/raw-output lineage without downloading the original images or base weights. Optional `--manuscript` and `--baseline-delivery` arguments add author-side document and unchanged-baseline checks when those packages are available. Full inference reproduction still requires the official inputs and pinned enhancer artifacts.
