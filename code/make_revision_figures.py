"""Create publication figures from saved, complete-fold evidence."""
import argparse
import json
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from analyze_revision import method_prediction, match
from collect_revision import degradation, working_image
from provenance_utils import sha256_file, write_json

def draw(axis, image, prediction, title, correct, n_gt, condition_tag):
    axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    b, s, c = prediction
    placed = []
    for box, score, cls, is_tp in zip(b, s, c, correct):
        if score < .25:
            continue
        x1, y1, x2, y2 = box
        color = '#08765b' if is_tp else '#b32624'
        axis.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color, linewidth=1.8))
        label = f'{"TP" if is_tp else "FP"} {int(cls)}:{score:.2f}'
        label_width = len(label) * 7.5
        label_x = min(x1, max(0, image.shape[1] - label_width))
        label_y = max(y1-9, 0)
        for shift in [0, -27, 27, -54, 54, -81, 81, -108, 108]:
            candidate_y = max(y1-9, 0) + shift
            if not 0 <= candidate_y <= image.shape[0] - 5:
                continue
            if not any(label_x < px+pw and px < label_x+label_width and abs(candidate_y-py) < 26 for px, py, pw in placed):
                label_y = candidate_y
                break
        placed.append((label_x, label_y, label_width))
        axis.text(label_x, label_y, label, fontsize=8, color='white',
                  bbox={'facecolor': color, 'edgecolor': color, 'pad': 1.0})
    visible = s >= .25
    tp = int(correct[visible].sum())
    fp = int(visible.sum()) - tp
    axis.text(.02, .03, f'TP {tp} | FP {fp} | FN {n_gt-tp}', transform=axis.transAxes,
              fontsize=8, color='white', bbox={'facecolor': '#222222', 'alpha': .85, 'pad': 2})
    axis.text(.02, .94, condition_tag, transform=axis.transAxes, fontsize=7.5, color='white',
              bbox={'facecolor': '#222222', 'alpha': .85, 'pad': 2})
    axis.set_title(title, fontsize=9.5)
    axis.axis('off')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((args.input / 'protocol.json').read_text())
    clean_paths = sorted((args.input / 'gamma3.0').glob('*.json'))
    noise_paths = sorted((args.input / 'linear_noise').glob('*.json'))
    if len(clean_paths) != protocol['n_images'] or len(noise_paths) != protocol['n_images']:
        raise RuntimeError('Full-fold gamma3.0 and linear_noise inputs are required')
    rng = np.random.default_rng(20261027)
    random_path = clean_paths[int(rng.integers(len(clean_paths)))]
    worst_path, worst_delta = None, -float('inf')
    for path in noise_paths:
        row = json.loads(path.read_text())
        errors = []
        for method in ['Baseline', 'EAMF']:
            pred = method_prediction(row, method)
            t, s, _, gt = match(row, pred)
            tp = int(t[s >= .25, 0].sum())
            errors.append(int((s >= .25).sum()) - tp + len(gt) - tp)
        delta = errors[1] - errors[0]
        if delta > worst_delta:
            worst_path, worst_delta = path, delta
    selection = [('gamma3.0', random_path, 'seeded random'),
                 ('linear_noise', worst_path, 'largest extra FP+FN')]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.6), dpi=300)
    lineage = []
    for row_index, (condition, path, rule) in enumerate(selection):
        row = json.loads(path.read_text())
        image_path = args.images / row['image']
        image = degradation(working_image(cv2.imread(str(image_path)), protocol.get('work_longside', 0)), condition, row['image'])
        for col, method in enumerate(['Baseline', 'EAMF']):
            prediction = method_prediction(row, method)
            correct = match(row, prediction)[0][:, 0]
            draw(axes[row_index, col], image, prediction,
                 f'{method}: {row["image"]}', correct, len(row['gt_cls']),
                 'g=3; seeded random' if row_index == 0 else 'noise; error-selected')
        lineage.append({'condition': condition, 'image': row['image'], 'selection': rule,
                        'source_image_sha256': sha256_file(image_path), 'raw_record_sha256': sha256_file(path)})
    fig.subplots_adjust(left=.005, right=.995, bottom=.005, top=.94, wspace=.025, hspace=.14)
    fig.savefig(args.output / 'revision_examples.png', dpi=300)
    plt.close(fig)
    write_json(args.output / 'figure_lineage.json', {'seed': 20261027, 'display_threshold': .25,
               'failure_error_definition': 'FP50+FN50 at confidence .25; ties broken by lexicographic image name',
               'worst_extra_errors': worst_delta, 'examples': lineage,
               'output_sha256': sha256_file(args.output / 'revision_examples.png')})

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.7), dpi=300)
    for ax, condition in zip(axes, ['gamma3.0', 'linear_noise']):
        rows = [json.loads(path.read_text()) for path in sorted((args.input / condition).glob('*.json'))]
        for method, color in [('Baseline', '#6b7280'), ('EAMF', '#08765b'), ('TTA3', '#b46d12')]:
            matched = [match(row, method_prediction(row, method)) for row in rows]
            scores = np.concatenate([m[1] for m in matched])
            tp = np.concatenate([m[0][:, 0] for m in matched])
            tp = tp[np.argsort(-scores)]
            cum_tp = tp.cumsum()
            precision = cum_tp / np.maximum(np.arange(len(tp)) + 1, 1)
            recall = cum_tp / sum(len(m[3]) for m in matched)
            ax.plot(recall, precision, label=method, color=color, linewidth=1.7)
        ax.set(xlabel='Micro recall', ylabel='Micro precision', title=condition, xlim=(0, 1), ylim=(0, 1.02))
        ax.grid(alpha=.18)
        ax.legend(fontsize=8, loc='lower left')
    fig.tight_layout()
    fig.savefig(args.output / 'micro_precision_recall.png', dpi=300)
    plt.close(fig)
    print('Figures and source lineage complete')
