"""Collect a fixed released learned-enhancer comparator on common full-fold inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collect_revision import degradation, working_image, predict
from provenance_utils import environment_record, sha256_file, write_json
from zero_dce_adapter import ZeroDCE, EXPECTED_MODEL, EXPECTED_WEIGHT

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
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
    assert sha256_file(args.weights) == parent['weights_sha256']
    protocol = {'method': 'ZeroDCE+same-frozen-detector', 'conditions': ['clean', 'gamma3.0', 'linear_noise'],
                'n_images': 2352, 'parent_signature': parent['signature'],
                'weights_sha256': parent['weights_sha256'], 'upstream_source_sha256': EXPECTED_MODEL,
                'enhancer_weights_sha256': EXPECTED_WEIGHT, 'upstream_commit': 'e0f4adc54d0f23348c4a9b84acc08fe8778d5bfd',
                'code_sha256': sha256_file(Path(__file__)),
                'adapter_sha256': sha256_file(Path(__file__).with_name('zero_dce_adapter.py')),
                'collect_revision_sha256': sha256_file(Path(__file__).with_name('collect_revision.py')),
                'device': args.device, 'half': False, 'tf32': False,
                'input': 'same aspect-preserving INTER_LINEAR 640 working resize, then identical fixed corruption',
                'enhancer': 'released Epoch99; unchanged model; RGB float32 /255; second output; uint8 +0.5 clipping; no JPEG encoding',
                'training': 'none on ITSD; fixed external pretrained weights; no test-label tuning',
                'detector': {'imgsz': 640, 'conf': .001, 'iou': .70, 'max_det': 300}}
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())['signature'] != signature:
        raise RuntimeError('Protocol differs; select new output')
    if not protocol_path.exists():
        write_json(protocol_path, {**protocol, 'signature': signature, 'environment': environment_record()})
    enhancer = ZeroDCE(args.upstream, args.device)
    model = YOLO(str(args.weights))
    for _ in range(5):
        predict(model, enhancer.enhance(np.zeros((360, 640, 3), np.uint8)), args.device)
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
                    raise RuntimeError('Cached comparator record changed')
                continue
            image = cv2.imread(str(args.images / source['image']))
            if image is None:
                raise RuntimeError('Image unavailable ' + source['image'])
            image = degradation(working_image(image, 640), condition, source['image'])
            assert list(image.shape[:2]) == source['shape']
            enhanced = enhancer.enhance(image)
            prediction = predict(model, enhanced, args.device)
            write_json(destination, {'image': source['image'], 'condition': condition, 'signature': signature,
                'parent_sha256': parent_sha, 'shape': source['shape'], 'gt_cls': source['gt_cls'],
                'gt_boxes': source['gt_boxes'], 'prediction': prediction,
                'common_input_pixel_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                'enhanced_pixel_sha256': hashlib.sha256(enhanced.tobytes()).hexdigest()})
            new += 1
            if (i + 1) % 128 == 0:
                print(f'{condition}: {i+1}/2352', flush=True)
            if args.max_new and new >= args.max_new:
                print(f'BOUNDED CHECKPOINT: {new} new complete records', flush=True)
                raise SystemExit(0)
        print('COMPLETE ' + condition, flush=True)
    print('ALL ZERO-DCE CONDITIONS COMPLETE', flush=True)
