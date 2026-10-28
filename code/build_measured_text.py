"""Generate manuscript tables only from complete, validated analysis outputs."""
import argparse
import json
from pathlib import Path
from provenance_utils import sha256_file, write_json
from collect_revision import CONDITIONS

LABELS = {'clean': 'Unmodified', 'gamma1.8': '$g=1.8$', 'gamma2.4': '$g=2.4$',
          'gamma3.0': '$g=3.0$', 'linear_noise': 'Linear-light noise',
          'overexposed': 'Overexposure', 'haze': 'Haze'}
PRETTY = {'Baseline': 'Baseline', 'Gamma': 'Gamma only', 'CLAHE': 'CLAHE only',
          'RoutedGamma': 'Routed gamma', 'Unconditional': 'Unconditional fusion', 'EAMF': 'EAMF',
          'OriginalGamma': 'Original + gamma', 'OriginalCLAHE': 'Original + CLAHE',
          'GammaCLAHE': 'Gamma + CLAHE', 'UniformWBF': 'Uniform WBF',
          'DeduplicatedWBF': 'Deduplicated WBF', 'NoOriginalFloor': 'No original floor',
          'NMS': 'NMS merger', 'SoftNMS': 'Soft-NMS merger', 'TTA3': 'Flip/scale TTA (3 calls)'}

def percent(value):
    return f'{100 * value:.2f}'

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--benchmark', type=Path, required=True)
    p.add_argument('--manuscript', type=Path, required=True)
    p.add_argument('--multiscale', type=Path, required=True)
    args = p.parse_args()
    protocol = json.loads((args.cache / 'protocol.json').read_text())
    data = {}
    hashes = {}
    for condition in CONDITIONS:
        path = args.analysis / (condition + '_results.json')
        result = json.loads(path.read_text())
        if result['n_images'] != 2352 or result['partial'] or result['source_signature'] != protocol['signature']:
            raise RuntimeError(f'Not complete or mismatched: {condition}')
        if result['paired_bootstrap_eamf_vs_baseline']['resamples'] < 1000:
            raise RuntimeError(f'Bootstrap incomplete: {condition}')
        data[condition] = result
        hashes[path.name] = sha256_file(path)
    benchmark = json.loads(args.benchmark.read_text())
    if 'summary_ms' not in benchmark:
        raise RuntimeError('Timing run incomplete')
    hashes[args.benchmark.name] = sha256_file(args.benchmark)
    multiscale = {}
    for condition in ['clean', 'gamma3.0', 'linear_noise']:
        ms_path = args.multiscale / (condition + '_multiscale.json')
        ms = json.loads(ms_path.read_text())
        if ms['n_images'] != 2352 or ms['CI_EAMF_minus_TTAms3']['resamples'] < 1000:
            raise RuntimeError('Three-scale comparison incomplete')
        multiscale[condition] = ms
        hashes[ms_path.name] = sha256_file(ms_path)
    macros = ('% Generated from complete-fold measured outputs.\n'
              '\\newcommand{\\GammaThreeBaseline}{' + percent(data['gamma3.0']['metrics']['Baseline']['mAP50']) + '}\n'
              '\\newcommand{\\GammaThreeEAMF}{' + percent(data['gamma3.0']['metrics']['EAMF']['mAP50']) + '}\n'
              '\\newcommand{\\NoiseBaseline}{' + percent(data['linear_noise']['metrics']['Baseline']['mAP50']) + '}\n'
              '\\newcommand{\\NoiseEAMF}{' + percent(data['linear_noise']['metrics']['EAMF']['mAP50']) + '}\n'
              '\\newcommand{\\GammaThreeDelta}{' + f"{100*(data['gamma3.0']['metrics']['EAMF']['mAP50']-data['gamma3.0']['metrics']['Baseline']['mAP50']):.2f}" + '}\n'
              '\\newcommand{\\GammaThreeCILow}{' + f"{data['gamma3.0']['paired_bootstrap_eamf_vs_baseline']['delta_mAP50_pp_CI95'][0]:.2f}" + '}\n'
              '\\newcommand{\\GammaThreeCIHigh}{' + f"{data['gamma3.0']['paired_bootstrap_eamf_vs_baseline']['delta_mAP50_pp_CI95'][1]:.2f}" + '}\n')
    (args.manuscript / 'measured_macros.tex').write_text(macros, encoding='utf8')

    lines = [r'\begin{table*}[t]', r'\centering\small\setlength{\tabcolsep}{4pt}',
             r'\caption{All 2,352 Fold 1 images. Accuracy columns show mAP@50 / mAP@50:95 in percent. Differences use unrounded measurements; differences and paired bootstrap 95\% intervals are in percentage points.}',
             r'\label{tab:results}', r'\begin{tabular}{lccccc}', r'\toprule',
             r'Condition & Baseline & EAMF & Flip/scale TTA & EAMF $-$ baseline, AP50 & Activation \\', r'\midrule']
    for condition in CONDITIONS:
        row = data[condition]
        m = row['metrics']
        pair = lambda key: percent(m[key]['mAP50']) + ' / ' + percent(m[key]['mAP50_95'])
        delta = 100 * (m['EAMF']['mAP50'] - m['Baseline']['mAP50'])
        low, high = row['paired_bootstrap_eamf_vs_baseline']['delta_mAP50_pp_CI95']
        lines.append(f"{LABELS[condition]} & {pair('Baseline')} & {pair('EAMF')} & {pair('TTA3')} & {delta:+.2f} [{low:+.2f}, {high:+.2f}] & {100*row['activation']:.1f}\\% " + r'\\')
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table*}'])
    lines += [r'\begin{table}[t]', r'\centering\small\setlength{\tabcolsep}{4pt}',
              r'\caption{Ablation mAP@50 (percent); every row uses all 2,352 images. Complete mAP@50:95, class AP, and all stress conditions are supplied in the reproducibility archive.}',
              r'\label{tab:ablation}', r'\begin{tabular}{lccc}', r'\toprule',
              r'Configuration & Unmodified & $g=3.0$ & Linear noise \\', r'\midrule']
    for method, label in PRETTY.items():
        values = [percent(data[c]['metrics'][method]['mAP50']) for c in ['clean', 'gamma3.0', 'linear_noise']]
        lines.append(label + ' & ' + ' & '.join(values) + r' \\')
    lines.append('Scale-only TTA (3 calls) & ' + ' & '.join(percent(multiscale[c]['TTAms3']['mAP50']) for c in ['clean', 'gamma3.0', 'linear_noise']) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    lines += [r'\begin{table*}[t]', r'\centering\small\setlength{\tabcolsep}{4pt}',
              r'\caption{Standalone timing from the common 640-pixel working input; native decode, shared resize, and stress construction are excluded. Stage means are in ms/image. FPS is reciprocal mean latency, not measured batched throughput.}',
              r'\label{tab:latency}', r'\begin{tabular}{llrrrrrr}', r'\toprule',
              r'Condition & Method & Router & Enhance/weights & Detector path & Merge & Total & FPS \\', r'\midrule']
    for condition in ['clean', 'gamma3.0', 'linear_noise']:
        for method in ['Baseline', 'EAMF', 'TTAms3', 'ZeroDCE', 'FineTuneGamma']:
            row = benchmark['summary_ms'][condition][method]
            values = [f'{row[k]:.1f}' for k in ['router', 'enhancement_and_weights', 'inference', 'fusion', 'total']]
            values.append(f'{1000/row["total"]:.1f}')
            timing_label = {'TTAms3': 'Scale-only TTA', 'ZeroDCE': 'Zero-DCE', 'FineTuneGamma': 'Gamma fine-tune'}.get(method, PRETTY.get(method, method))
            lines.append(LABELS[condition] + ' & ' + timing_label + ' & ' + ' & '.join(values) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']

    tables = list(lines)
    lines = [r'Table~\ref{tab:results} compares complete-fold conditions, Table~\ref{tab:ablation} summarizes ablations, and Table~\ref{tab:latency} uses one shared timing run.']

    clean, g3, noise = [data[c] for c in ['clean', 'gamma3.0', 'linear_noise']]
    base = clean['metrics']['Baseline']['mAP50']
    delta = 100 * (clean['metrics']['EAMF']['mAP50'] - base)
    low, high = clean['paired_bootstrap_eamf_vs_baseline']['delta_mAP50_pp_CI95']
    lines.append(f"On unmodified images, EAMF changes mAP@50 by {delta:+.2f} percentage points (paired 95\\% interval [{low:+.2f}, {high:+.2f}]). The ITSD paper reports a six-fold YOLOv12n mean of 96.00\\% with standard deviation 0.70 percentage points \\cite{{zarei2026itsd}}; our full-fold baseline is {percent(base)}\\%. This is contextual comparison, not a reproduction of the six-fold training study.")
    clean_ms = multiscale['clean']
    ci_clean_ms = clean_ms['CI_EAMF_minus_TTAms3']['delta_mAP50_pp_CI95']
    lines.append(f"On the same clean inputs, scale-only TTA reaches {percent(clean_ms['TTAms3']['mAP50'])}\\% mAP@50; the EAMF-minus-scale-TTA interval is [{ci_clean_ms[0]:+.2f}, {ci_clean_ms[1]:+.2f}] percentage points. EAMF is therefore not the strongest clean-image aggregation method in this comparison.")
    lines.append(f"At $g=3.0$, gamma-only preprocessing has a higher mAP@50 point estimate ({percent(data['gamma3.0']['metrics']['Gamma']['mAP50'])}\\%) than full EAMF ({percent(data['gamma3.0']['metrics']['EAMF']['mAP50'])}\\%). The full wrapper therefore has no demonstrated universal advantage over its simpler single-view component.")
    for condition in ['gamma3.0', 'linear_noise']:
        m = data[condition]['metrics']
        ci = data[condition]['paired_bootstrap_eamf_vs_tta']['delta_mAP50_pp_CI95']
        gain = 100 * (m['EAMF']['mAP50'] - m['TTA3']['mAP50'])
        lines.append(f"For {LABELS[condition]}, the EAMF--TTA mAP@50 difference is {gain:+.2f} percentage points with paired 95\\% interval [{ci[0]:+.2f}, {ci[1]:+.2f}]. This comparison shares the same three-call budget but does not equalize transform semantics or FLOPs.")
        ms = multiscale[condition]
        ci_ms = ms['CI_EAMF_minus_TTAms3']['delta_mAP50_pp_CI95']
        delta_ms = 100 * (ms['EAMF']['mAP50'] - ms['TTAms3']['mAP50'])
        lines.append(f"Against class-preserving 512/640/768 scale-only TTA, the difference is {delta_ms:+.2f} percentage points (95\\% interval [{ci_ms[0]:+.2f}, {ci_ms[1]:+.2f}]).")
    noise_b = data['linear_noise']['metrics']['Baseline']
    noise_e = data['linear_noise']['metrics']['EAMF']
    lines.append(f"The sensor-noise stress does not demonstrate a transfer of the gamma-stress gain: mAP@50 changes from {percent(noise_b['mAP50'])}\\% to {percent(noise_e['mAP50'])}\\%, and mAP@50:95 from {percent(noise_b['mAP50_95'])}\\% to {percent(noise_e['mAP50_95'])}\\%. Brightening and view aggregation do not establish recovery of information lost to noise.")
    proxy = clean.get('natural_low_luminance_proxy')
    if proxy:
        lines.append(f"The naturally low-luminance proxy contains {proxy['n_images']} unmodified images. Baseline/EAMF mAP@50 are {percent(proxy['baseline']['mAP50'])}\\% / {percent(proxy['eamf']['mAP50'])}\\%; the subset remains brightness-selected rather than a labeled night benchmark.")
    lines.append(f"Among router-active unmodified images, {clean['gamma_duplicate_active']} have a gamma view identical to the original. The deduplicated and uniform-weight ablations in Table~\\ref{{tab:ablation}} test whether the original scoring rule contributes beyond repeated-view aggregation. The accompanying JSON reports view-weight diagnostics and descriptive threshold sensitivity without selecting new constants.")
    (args.manuscript / 'measured_results.tex').write_text('\n'.join(lines) + '\n', encoding='utf8')
    (args.manuscript / 'measured_tables.tex').write_text('\n'.join(tables) + '\n', encoding='utf8')

    bproto = benchmark['protocol']
    device = bproto['device']
    hardware = benchmark['environment']['torch_hardware']
    hw = hardware.get('cuda_device') if device != 'cpu' else 'AMD Ryzen 7 6800H CPU'
    accuracy_hw = (protocol['environment']['torch_hardware'].get('cuda_device')
                   if protocol['device'] != 'cpu' else 'AMD Ryzen 7 6800H CPU')
    runtime = (f"The accuracy run uses Python 3.12.3, Ultralytics 8.3.233, PyTorch 2.2.2, "
               f"and the {accuracy_hw}. Latency is measured separately on the {hw} using one shared standalone benchmark: "
               f"{len(bproto['selected_images'])} uniformly sampled Fold~1 images, {bproto['repeats']} repetitions, "
               f"five warm-up calls, fixed seed, and randomized method order per image. "
               "The same sampled IDs are used across conditions. Full latency includes the router, enhancement, "
               "quality-weight calculation, detector preprocessing/inference/NMS, CPU transfer of boxes, and fusion; "
               "native-image decoding, the common input resize, and construction of the common stress input are excluded. "
               "Thus these are working-input wrapper timings, not camera-to-output latency. "
               "CUDA measurements synchronize the device where applicable. No source of latency is copied "
               "from the earlier 117-image runs. Table~\\ref{tab:latency} separates stages and does not equate "
               "avoiding extra detector calls with zero preprocessing overhead.\n")
    (args.manuscript / 'runtime_protocol.tex').write_text(runtime, encoding='utf8')
    write_json(args.manuscript / 'manuscript_table_lineage.json', {
        'inputs': hashes, 'cache_signature': protocol['signature'],
        'outputs': {name: sha256_file(args.manuscript / name) for name in [
            'measured_macros.tex', 'measured_results.tex', 'measured_tables.tex', 'runtime_protocol.tex']}})
    print('Measured manuscript text generated; complete-fold gates passed')
