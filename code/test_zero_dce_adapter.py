"""Validate adapter I/O against the original test path's torchvision conversion."""
import argparse
import io
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.utils import save_image
from zero_dce_adapter import ZeroDCE

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    enhancer = ZeroDCE(args.upstream, 'cpu')
    rng = np.random.default_rng(20261028)
    image = rng.integers(0, 256, (37, 61, 3), dtype=np.uint8)
    actual = enhancer.enhance(image)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    reference_input = torch.from_numpy((np.asarray(Image.fromarray(rgb)) / 255.0)).float().permute(2, 0, 1).unsqueeze(0)
    with torch.inference_mode():
        reference = enhancer.model(reference_input)[1]
    stream = io.BytesIO()
    save_image(reference, stream, format='PNG')
    expected = cv2.cvtColor(np.asarray(Image.open(io.BytesIO(stream.getvalue()))), cv2.COLOR_RGB2BGR)
    np.testing.assert_array_equal(actual, expected)
    for value in [0, 255]:
        endpoint = np.full((19, 23, 3), value, np.uint8)
        np.testing.assert_array_equal(enhancer.enhance(endpoint), endpoint)
    assert actual.shape == image.shape and actual.dtype == np.uint8
    print('PASS: original test RGB normalization and torchvision PNG quantization match exactly; geometry and endpoint invariants pass')
