"""Predeclared bounded gamma-augmentation fine-tuning comparator; no Fold 1 tuning."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from provenance_utils import environment_record, build_manifest, sha256_file, write_json


class RandomGamma:
    def __call__(self, labels):
        if random.random() < .5:
            gamma = random.uniform(1.0, 3.0)
            lut = np.array([255 * (value / 255) ** gamma for value in range(256)], np.uint8)
            labels['img'] = cv2.LUT(labels['img'], lut)
        return labels


def attach_gamma(trainer):
    trainer.train_loader.dataset.transforms.insert(0, RandomGamma())
    print('Attached predeclared gamma augmentation before standard dataset transforms', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--fold2', type=Path, required=True)
    p.add_argument('--test-manifest', type=Path, required=True)
    p.add_argument('--classes', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted((args.fold2 / 'images').glob('*.jpg'))
    assert len(paths) > 2000
    source_manifest = build_manifest(paths, args.fold2 / 'labels', args.fold2 / 'images')
    write_json(args.output / 'fold2_source_manifest.json', source_manifest)
    test = json.loads(args.test_manifest.read_text())
    test_hashes = {r['image_sha256'] for r in test}
    test_names = {r['image'] for r in test}
    excluded_overlap = [r for r in source_manifest if r['image_sha256'] in test_hashes or r['image'] in test_names]
    eligible = [r for r in source_manifest if r['image_sha256'] not in test_hashes and r['image'] not in test_names]
    manifest, duplicate_exclusions, seen = [], [], set()
    for row in eligible:
        if row['image_sha256'] in seen:
            duplicate_exclusions.append(row)
        else:
            manifest.append(row)
            seen.add(row['image_sha256'])
    paths = [args.fold2 / 'images' / row['image'] for row in manifest]
    write_json(args.output / 'split_overlap_audit.json', {
        'source_fold2_images': len(source_manifest), 'fold1_images': len(test),
        'excluded_test_overlap': excluded_overlap, 'excluded_internal_duplicate_copies': duplicate_exclusions,
        'eligible_unique_fold2_images': len(manifest),
        'rule': 'Exclude every Fold2 image matching Fold1 by filename or exact encoded-file SHA256; retain lexicographically first remaining encoded-file duplicate only.'})
    assert not ({r['image_sha256'] for r in manifest} & test_hashes)
    assert not ({r['image'] for r in manifest} & test_names)
    shuffled = sorted(paths, key=lambda path: hashlib.sha256(('20261028:' + path.name).encode()).hexdigest())
    n_val = round(.1 * len(paths))
    validation, train = shuffled[:n_val], shuffled[n_val:]
    write_json(args.output / 'fold2_manifest.json', manifest)
    (args.output / 'train.txt').write_text('\n'.join(path.resolve().as_posix() for path in train) + '\n')
    (args.output / 'val.txt').write_text('\n'.join(path.resolve().as_posix() for path in validation) + '\n')
    names = yaml.safe_load(args.classes.read_text())['names']
    data = {'path': args.output.resolve().as_posix(), 'train': 'train.txt', 'val': 'val.txt', 'names': names}
    (args.output / 'dataset.yaml').write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
    protocol = {'name': 'fixed-budget gamma-augmentation fine-tune', 'code_sha256': sha256_file(Path(__file__)),
        'initial_weights_sha256': sha256_file(args.weights), 'fold2_manifest_sha256': sha256_file(args.output / 'fold2_manifest.json'),
        'test_manifest_sha256': sha256_file(args.test_manifest), 'test_overlap_by_filename_and_sha256': 0,
        'train_images': len(train), 'validation_images': len(validation), 'source_partition': 'official Fold-2 only',
        'split': 'Remove exact Fold1 overlap and deduplicate Fold2 before 10% validation by ascending SHA256(20261028:filename); rest training',
        'source_fold2_images': len(source_manifest), 'excluded_test_overlap': len(excluded_overlap),
        'excluded_internal_duplicate_copies': len(duplicate_exclusions),
        'epochs': 3, 'imgsz': 640, 'batch': 4, 'seed': 20261028, 'optimizer': 'SGD',
        'lr0': .001, 'lrf': .1, 'momentum': .9, 'weight_decay': .0005, 'warmup_epochs': 1,
        'augmentation': 'p=.5 gamma U[1,3], uint8 LUT, before standard transforms; geometric/color augmentation disabled',
        'checkpoint_selection': 'last epoch only; no selection using Fold 1 or best validation checkpoint',
        'scope': 'Exploratory short fine-tune, not exhaustive training or independent confirmation of historical EAMF tuning',
        'environment': environment_record()}
    protocol_path = args.output / 'training_protocol.json'
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text())
        for key in ['code_sha256', 'initial_weights_sha256', 'fold2_manifest_sha256', 'train_images', 'validation_images']:
            assert old[key] == protocol[key], 'Changed training protocol'
    else:
        write_json(protocol_path, protocol)
    if args.prepare_only:
        print(f'PREPARED: {len(train)} train, {len(validation)} validation; no Fold 1 overlap', flush=True)
        raise SystemExit(0)
    if (args.output / 'training_complete.json').exists():
        raise RuntimeError('Training already completed; do not repeat or overwrite')
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    # Prevent the optional Albumentations defaults from introducing unrecorded transforms.
    import ultralytics.data.augment as augmentation
    class NoAdditionalAlbumentations:
        def __init__(self, *a, **kw):
            pass
        def __call__(self, labels):
            return labels
    augmentation.Albumentations = NoAdditionalAlbumentations
    model = YOLO(str(args.weights))
    model.add_callback('on_pretrain_routine_end', attach_gamma)
    started = time.perf_counter()
    model.train(data=str(args.output / 'dataset.yaml'), epochs=3, imgsz=640, batch=4, device=0,
        workers=0, amp=False, seed=20261028, deterministic=True, optimizer='SGD', lr0=.001, lrf=.1,
        momentum=.9, weight_decay=.0005, warmup_epochs=1, patience=1000, plots=False, cache=False,
        mosaic=0., mixup=0., copy_paste=0., fliplr=0., flipud=0., hsv_h=0., hsv_s=0., hsv_v=0.,
        degrees=0., translate=0., scale=0., shear=0., perspective=0., close_mosaic=0,
        project=str(args.output), name='run', exist_ok=False, save_period=1, verbose=False)
    final_weight = args.output / 'run' / 'weights' / 'last.pt'
    assert final_weight.is_file()
    write_json(args.output / 'training_complete.json', {'training_seconds': time.perf_counter() - started,
        'last_checkpoint_sha256': sha256_file(final_weight), 'checkpoint': str(final_weight),
        'training_protocol_sha256': sha256_file(protocol_path), 'completed_epochs': 3})
    print('TRAINING COMPLETE; LAST-EPOCH CHECKPOINT FROZEN', flush=True)
