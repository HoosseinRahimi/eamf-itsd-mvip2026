"""Fetch immutable original-author Zero-DCE files with audited SHA-256 checks."""
import argparse
from pathlib import Path
import requests
from provenance_utils import sha256_file, write_json

COMMIT = 'e0f4adc54d0f23348c4a9b84acc08fe8778d5bfd'
ARTIFACTS = {
    'README.md': ('README.md', '095d2f1039d2c111265c88243196644e0310898b1e63157610cfb8569293c006'),
    'model.py': ('Zero-DCE_code/model.py', '8adcc8b73b5bfdb1b6e5a16e672c59a7f8f664ee21aafc3eea44bb316d4b97f8'),
    'lowlight_test.py': ('Zero-DCE_code/lowlight_test.py', '3481f8065395bb43a84c312092fa03f128a1d086f19d3457d5c90bbc864db0f3'),
    'Epoch99.pth': ('Zero-DCE_code/snapshots/Epoch99.pth', 'a4395acb874f320375d9704997cef874eaaaaa26a1777ceb29a92b70f74c3612'),
}

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for local, (upstream, expected) in ARTIFACTS.items():
        url = f'https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE/{COMMIT}/{upstream}'
        path = args.output / local
        if not path.exists():
            response = requests.get(url, timeout=(15, 60))
            response.raise_for_status()
            path.write_bytes(response.content)
        assert sha256_file(path) == expected, 'Upstream checksum mismatch: ' + local
        records.append({'file': local, 'url': url, 'sha256': expected})
    write_json(args.output / 'pinned_source_manifest.json', {'repository': 'https://github.com/Li-Chongyi/Zero-DCE',
        'commit': COMMIT, 'license': 'Original authors: academic research; Attribution-NonCommercial 4.0 International',
        'artifacts': records})
    print('PASS: all pinned original-author files verified')
