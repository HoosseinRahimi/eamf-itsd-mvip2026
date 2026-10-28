from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions(names):
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_record():
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        torch_hardware = {
            "cuda_available": cuda_available,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        }
    except Exception as exc:
        torch_hardware = {"error": repr(exc)}
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_hardware": torch_hardware,
        "packages": package_versions(
            [
                "ultralytics",
                "torch",
                "torchvision",
                "opencv-python",
                "numpy",
                "pandas",
                "PyYAML",
                "tqdm",
                "ensemble-boxes",
            ]
        ),
    }


def build_manifest(paths, label_dir: Path, image_dir: Path):
    rows = []
    for index, image_path in enumerate(paths):
        label_path = label_dir / f"{image_path.stem}.txt"
        rows.append(
            {
                "index": index,
                "image": image_path.relative_to(image_dir).as_posix(),
                "image_bytes": image_path.stat().st_size,
                "image_sha256": sha256_file(image_path),
                "label": label_path.name,
                "label_exists": label_path.exists(),
                "label_bytes": label_path.stat().st_size if label_path.exists() else None,
                "label_sha256": sha256_file(label_path) if label_path.exists() else None,
            }
        )
    return rows


def json_ready_record(record):
    return {
        "image_id": record["image_id"],
        "gt_cls": record["gt_cls"].tolist(),
        "gt_boxes": record["gt_boxes"].tolist(),
        "pred_cls": record["pred_cls"].tolist(),
        "pred_boxes": record["pred_boxes"].tolist(),
        "pred_scores": record["pred_scores"].tolist(),
    }


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
