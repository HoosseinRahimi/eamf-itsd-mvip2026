"""Count actual class support and relative box-size coverage of Fold 1."""
import argparse
from collections import Counter
from pathlib import Path
from provenance_utils import write_json

def summarize(paths, labels):
    instances, images, sizes = Counter(), Counter(), Counter()
    for path in paths:
        rows = [line.split() for line in (labels / (path.stem + '.txt')).read_text().splitlines() if line.strip()]
        present = set()
        for row in rows:
            cls = int(row[0])
            instances[cls] += 1
            present.add(cls)
            # Relative image area, not COCO pixel area.
            area = float(row[3]) * float(row[4])
            sizes['below_0.1pct' if area < .001 else '0.1_to_1pct' if area < .01 else 'at_least_1pct'] += 1
        images.update(present)
    return {'n_images': len(paths), 'n_instances': sum(instances.values()),
            'present_classes': len(instances), 'absent_class_ids': [c for c in range(118) if not instances[c]],
            'classes_with_fewer_than_5_instances': [c for c in range(118) if 0 < instances[c] < 5],
            'instances_per_class': dict(sorted(instances.items())),
            'images_per_class': dict(sorted(images.items())), 'relative_area_bins': dict(sizes)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    result = {'old_first117': summarize(paths[:117], args.labels), 'entire_fold1': summarize(paths, args.labels)}
    write_json(args.output, result)
    for name, value in result.items():
        print(name, {k: v for k, v in value.items() if k in ['n_images', 'n_instances', 'present_classes', 'relative_area_bins']})
