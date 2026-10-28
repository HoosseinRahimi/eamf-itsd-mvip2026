import unittest
from types import SimpleNamespace
import numpy as np
import torch
from analyze_revision import aggregate, arrays, bootstrap, bootstrap_ap, fuse, match
from collect_revision import degradation, predict, working_image

class ScientificInvariants(unittest.TestCase):
    def test_working_resize_preserves_aspect_ratio(self):
        image = np.zeros((1080, 1920, 3), np.uint8)
        resized = working_image(image, 640)
        self.assertEqual(resized.shape, (360, 640, 3))
        self.assertIs(working_image(image, 0), image)

    def test_empty_detections_work_for_every_merge_method(self):
        empty = {'boxes': [], 'scores': [], 'classes': []}
        row = {'shape': [100, 100], 'views': {'original': empty, 'gamma': empty, 'clahe': empty}}
        for kind in ['wbf', 'nms', 'softnms']:
            b, s, c = fuse(row, ['original', 'gamma', 'clahe'], [1, 1, 1], kind)
            self.assertEqual(b.shape, (0, 4))
            self.assertEqual(len(s), 0)
            self.assertEqual(len(c), 0)

    def test_a_single_empty_view_does_not_break_nms_or_wbf(self):
        empty = {'boxes': [], 'scores': [], 'classes': []}
        detection = {'boxes': [[10, 10, 30, 30]], 'scores': [.9], 'classes': [0]}
        row = {'shape': [100, 100], 'views': {'original': empty, 'gamma': detection, 'clahe': detection}}
        for kind in ['wbf', 'nms', 'softnms']:
            b, s, c = fuse(row, ['original', 'gamma', 'clahe'], [1, 1, 1], kind)
            self.assertEqual(b.shape[1], 4)
            self.assertGreaterEqual(len(b), 1)
            self.assertEqual(len(b), len(s))
            self.assertEqual(len(b), len(c))

    def test_corruptions_preserve_geometry_and_seed(self):
        image = np.arange(60 * 80 * 3, dtype=np.uint8).reshape(60, 80, 3)
        for name in ['clean', 'gamma3.0', 'linear_noise', 'overexposed', 'haze']:
            result = degradation(image, name, 'sample.jpg')
            self.assertEqual(result.shape, image.shape)
            self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(np.array_equal(image, degradation(image, 'clean', 'sample.jpg')))
        a = degradation(image, 'linear_noise', 'sample.jpg')
        b = degradation(image, 'linear_noise', 'sample.jpg')
        c = degradation(image, 'linear_noise', 'different.jpg')
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, c))

    def test_flip_boxes_map_back_to_original_coordinates(self):
        boxes = SimpleNamespace(xyxy=torch.tensor([[5., 3., 25., 18.]]),
                                conf=torch.tensor([.9]), cls=torch.tensor([2.]))
        model = SimpleNamespace(predict=lambda *a, **k: [SimpleNamespace(boxes=boxes)])
        result = predict(model, np.zeros((40, 100, 3), np.uint8), 'cpu', flip=True)
        np.testing.assert_array_equal(result['boxes'], [[75., 3., 95., 18.]])

    def test_wbf_single_view_evidence_is_score_penalized(self):
        empty = {'boxes': [], 'scores': [], 'classes': []}
        row = {'shape': [100, 100], 'views': {
            'original': {'boxes': [[10, 10, 30, 30]], 'scores': [.9], 'classes': [0]},
            'gamma': empty, 'clahe': empty}}
        boxes, scores, cls = fuse(row, ['original', 'gamma', 'clahe'], [.65, 1, 1])
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(float(scores[0]), .9 * .65 / 2.65, places=6)

    def test_same_raw_predictions_have_zero_paired_difference(self):
        row = {'gt_cls': [0], 'gt_boxes': [[10, 10, 30, 30]]}
        pred = (np.array([[10, 10, 30, 30]], np.float32), np.array([.9], np.float32), np.array([0], np.int32))
        record = match(row, pred)
        self.assertTrue(record[0].all())
        self.assertAlmostEqual(aggregate([record])['mAP50'], .995, places=6)
        ci = bootstrap([record], [record], 20)
        self.assertEqual(ci['delta_mAP50_pp_CI95'], [0., 0.])
        self.assertEqual(ci['delta_mAP50_95_pp_CI95'], [0., 0.])

    def test_bootstrap_ap_equals_official_ap_on_repeated_images(self):
        rng = np.random.default_rng(428)
        items = []
        for i in range(12):
            gt = rng.integers(0, 4, 5)
            classes = rng.integers(0, 4, 7)
            scores = rng.random(7).astype(np.float32)
            tp = np.zeros((7, 10), bool)
            for j in range(7):
                tp[j, :rng.integers(0, 11)] = True
            items.append((tp, scores, classes, gt))
        for _ in range(20):
            indices = rng.integers(0, len(items), len(items))
            official = aggregate(items, indices)
            fast = bootstrap_ap(items, indices)
            np.testing.assert_allclose(fast, [official['mAP50'], official['mAP50_95']], atol=1e-12, rtol=0)

if __name__ == '__main__':
    unittest.main()
