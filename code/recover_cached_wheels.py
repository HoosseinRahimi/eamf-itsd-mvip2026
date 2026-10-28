"""Recover original wheel contents from uv's local cache for an offline clean install."""
import argparse
import base64
import csv
import email
import hashlib
import io
from pathlib import Path
import zipfile
from packaging.utils import canonicalize_name
from packaging.tags import sys_tags, parse_tag
from provenance_utils import sha256_file, write_json

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--lock', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    required = [tuple(line.strip().split('==')) for line in args.lock.read_text().splitlines() if '==' in line and not line.startswith('--')]
    cached = {}
    supported = set(sys_tags())
    for folder in sorted(args.cache.iterdir()):
        for meta in folder.glob('*.dist-info/METADATA'):
            fields = email.message_from_bytes(meta.read_bytes())
            wheel_info = meta.with_name('WHEEL')
            wheel_fields = email.message_from_bytes(wheel_info.read_bytes())
            tags = wheel_fields.get_all('Tag', [])
            if any(parse_tag(tag) & supported for tag in tags):
                cached.setdefault((canonicalize_name(fields['Name']), fields['Version']), (folder, meta.parent, tags))
    missing = [name + '==' + version for name, version in required if (canonicalize_name(name), version) not in cached]
    if missing:
        raise RuntimeError('Missing cached distributions: ' + ', '.join(missing))
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for name, version in required:
        folder, info, tags = cached[(canonicalize_name(name), version)]
        parts = [tag.split('-') for tag in tags]
        tag = '-'.join('.'.join(sorted({part[i] for part in parts})) for i in range(3))
        filename = canonicalize_name(name).replace('-', '_') + '-' + version + '-' + tag + '.whl'
        destination = args.output / filename
        rows = list(csv.reader(io.StringIO((info / 'RECORD').read_text(encoding='utf-8'))))
        verified = 0
        for relative, declared_hash, size in rows:
            path = folder / relative
            assert path.is_file() and path.resolve().is_relative_to(folder.resolve())
            if declared_hash:
                algorithm, expected = declared_hash.split('=', 1)
                digest = hashlib.new(algorithm, path.read_bytes()).digest()
                actual = base64.urlsafe_b64encode(digest).decode().rstrip('=')
                assert actual == expected, 'Cached payload hash mismatch: ' + str(path)
                verified += 1
        with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for relative, _, _ in rows:
                archive.write(folder / relative, relative)
        records.append({'distribution': name, 'version': version, 'wheel': filename,
            'wheel_sha256': sha256_file(destination), 'source_cache_entry': folder.name,
            'original_record_sha256': sha256_file(info / 'RECORD'), 'payload_files': len(rows), 'payload_hashes_verified': verified})
        print('VERIFIED CACHED WHEEL ' + name + '==' + version, flush=True)
    write_json(args.output / 'wheel_recovery_manifest.json', {'lock_sha256': sha256_file(args.lock),
        'method': 'Offline repack of unmodified original wheel payloads from uv archive cache; validate original RECORD hashes first; exclude unlisted bytecode/cache files.',
        'limitation': 'Repacked ZIP hashes differ from original upstream wheel archives; this is not a fresh upstream download.',
        'wheels': records})
