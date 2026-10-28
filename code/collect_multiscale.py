"""Add a 768-pixel scale to complete caches without altering their records."""
import argparse
import json
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO
from collect_revision import degradation, predict, working_image
from provenance_utils import sha256_file, write_json, environment_record

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='0')
    args = p.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    protocol = json.loads((args.cache / 'protocol.json').read_text())
    manifest = {r['image']: r for r in json.loads((args.cache / 'manifest.json').read_text())}
    if sha256_file(args.weights) != protocol['weights_sha256']:
        raise RuntimeError('Weight identity differs from primary experiment')
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'protocol.json', {
        'parent_signature': protocol['signature'], 'n_images': protocol['n_images'],
        'method': 'equal-weight WBF of 512/640/768, class-preserving scale transforms; 3 calls',
        'code_sha256': sha256_file(Path(__file__)), 'device': args.device,
        'environment': environment_record()})
    model = YOLO(str(args.weights))
    for condition in ['clean', 'gamma3.0', 'linear_noise']:
        folder = args.output / condition
        folder.mkdir(exist_ok=True)
        paths = sorted((args.cache / condition).glob('*.json'))
        if len(paths) != protocol['n_images']:
            raise RuntimeError('Primary condition is incomplete')
        for i, path in enumerate(paths):
            dest = folder / path.name
            if dest.exists():
                continue
            row = json.loads(path.read_text())
            source = args.images / row['image']
            if sha256_file(source) != manifest[row['image']]['image_sha256']:
                raise RuntimeError('Source image changed')
            image = degradation(working_image(cv2.imread(str(source)), protocol['work_longside']), condition, row['image'])
            prediction = predict(model, image, args.device, size=768)
            write_json(dest, {'image': row['image'], 'condition': condition,
                              'parent_record_sha256': sha256_file(path), 'scale768': prediction})
            if (i+1) % 100 == 0 or i+1 == len(paths):
                print(f'{condition} scale768 {i+1}/{len(paths)}', flush=True)
    print('MULTISCALE COLLECTION COMPLETE', flush=True)
