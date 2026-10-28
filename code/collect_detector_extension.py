"""Evaluate an identified extra detector/checkpoint on unchanged full-fold inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from eamf import adaptive_gamma, clahe_luminance
from collect_revision import degradation, working_image, predict
from provenance_utils import environment_record, sha256_file, write_json

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--views', choices=['original', 'photometric'], default='photometric')
    p.add_argument('--device', default='0')
    p.add_argument('--max-new', type=int, default=0)
    args = p.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.manual_seed(20261028)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    parent = json.loads((args.cache / 'protocol.json').read_text())
    assert parent['n_images'] == 2352 and parent['work_longside'] == 640
    model = YOLO(str(args.weights))
    assert len(model.names) == 118
    protocol = {'name': args.name, 'views': args.views, 'conditions': ['clean', 'gamma3.0', 'linear_noise'],
                'n_images': 2352, 'parent_signature': parent['signature'], 'weights_sha256': sha256_file(args.weights),
                'class_names': model.names, 'model_yaml': str(model.model.yaml),
                'code_sha256': sha256_file(Path(__file__)),
                'collect_revision_sha256': sha256_file(Path(__file__).with_name('collect_revision.py')),
                'eamf_sha256': sha256_file(Path(__file__).with_name('eamf.py')),
                'half': False, 'tf32': False, 'device': args.device,
                'detector': {'imgsz': 640, 'conf': .001, 'iou': .70, 'max_det': 300},
                'input': 'same common resized input and corruption as parent; unchanged photometric operators/parameters'}
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())['signature'] != signature:
        raise RuntimeError('Changed protocol; choose new directory')
    if not protocol_path.exists():
        write_json(protocol_path, {**protocol, 'signature': signature, 'environment': environment_record()})
    for _ in range(5):
        predict(model, np.zeros((360, 640, 3), np.uint8), args.device)
    new = 0
    for condition in protocol['conditions']:
        paths = sorted((args.cache / condition).glob('*.json'))
        assert len(paths) == 2352
        for i, path in enumerate(paths):
            source = json.loads(path.read_text())
            destination = args.output / condition / path.name
            parent_sha = sha256_file(path)
            if destination.exists():
                old = json.loads(destination.read_text())
                if old['signature'] != signature or old['parent_sha256'] != parent_sha:
                    raise RuntimeError('Cached record changed')
                continue
            original = cv2.imread(str(args.images / source['image']))
            if original is None:
                raise RuntimeError('Missing image ' + source['image'])
            image = degradation(working_image(original, 640), condition, source['image'])
            images = {'original': image}
            if args.views == 'photometric':
                images.update(gamma=adaptive_gamma(image), clahe=clahe_luminance(image))
            record = {key: source[key] for key in ['image', 'shape', 'gt_cls', 'gt_boxes',
                'stats', 'router_active', 'gamma_duplicates_original']}
            record.update(signature=signature, parent_sha256=parent_sha, condition=condition,
                common_input_pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                views={key: predict(model, view, args.device) for key, view in images.items()})
            write_json(destination, record)
            new += 1
            if (i + 1) % 128 == 0:
                print(f'{condition}: {i+1}/2352', flush=True)
            if args.max_new and new >= args.max_new:
                print(f'BOUNDED CHECKPOINT: {new} records', flush=True)
                raise SystemExit(0)
        print('COMPLETE ' + condition, flush=True)
    print('ALL EXTENSION CONDITIONS COMPLETE', flush=True)
