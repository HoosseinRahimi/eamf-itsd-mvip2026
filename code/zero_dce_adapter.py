"""Audited adapter around the unchanged original-author Zero-DCE network."""
import importlib.util
from pathlib import Path
import cv2
import numpy as np
import torch
from provenance_utils import sha256_file

EXPECTED_MODEL = '8adcc8b73b5bfdb1b6e5a16e672c59a7f8f664ee21aafc3eea44bb316d4b97f8'
EXPECTED_WEIGHT = 'a4395acb874f320375d9704997cef874eaaaaa26a1777ceb29a92b70f74c3612'


class ZeroDCE:
    def __init__(self, upstream: Path, device='0'):
        if sha256_file(upstream / 'model.py') != EXPECTED_MODEL:
            raise RuntimeError('Original-author model source differs')
        if sha256_file(upstream / 'Epoch99.pth') != EXPECTED_WEIGHT:
            raise RuntimeError('Original-author released checkpoint differs')
        spec = importlib.util.spec_from_file_location('upstream_zero_dce_model', upstream / 'model.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.device = torch.device('cpu' if device == 'cpu' else 'cuda:' + str(device))
        self.model = module.enhance_net_nopool().to(self.device).eval()
        self.model.load_state_dict(torch.load(upstream / 'Epoch99.pth', map_location=self.device, weights_only=True))

    @torch.inference_mode()
    def enhance(self, bgr):
        # Official test code uses RGB / 255 and the second network output.
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).to(self.device, torch.float32) / 255.0
        enhanced = self.model(tensor)[1]
        if not torch.isfinite(enhanced).all():
            raise RuntimeError('Non-finite enhancer output')
        # Match torchvision.save_image's uint8 conversion, without lossy JPEG I/O.
        rgb8 = enhanced.mul(255).add(.5).clamp(0, 255).permute(0, 2, 3, 1)[0].to('cpu', torch.uint8).numpy()
        return cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
