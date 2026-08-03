#!/usr/bin/env python3
"""Aggregate the preprocessing x seed sweep into paper-ready tables + a figure.

Reads every results_<prep>_s<seed>.json in CKPT_DIR, pools seeds per
(model, preprocess), and produces:
  - preprocessing_comparison.csv   : mean +/- std of each metric, per (model, prep)
  - preprocessing_significance.csv : baseline-vs-variant paired tests, per model+metric
  - preprocessing_comparison.tex   : LaTeX table (mean+/-std, best per model bolded)
  - preprocessing_comparison.png   : grouped bars with std error bars (test Top-1)

The comparison is PAIRED across seeds: at each seed all preprocess variants share
the same split, so we pair per-seed differences (baseline vs variant) and run both a
paired t-test and Wilcoxon signed-rank. With few seeds neither test has much power;
report the effect size (mean delta) alongside p, and prefer >=5 seeds for claims.

Usage:
    python aggregate_preprocessing.py --ckpt-dir /scratch/ttran72/checkpoints
"""
import os, re, json, glob, argparse
from collections import defaultdict

import numpy as np

try:
    from scipy import stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

TAG_RE = re.compile(r'^results_(?P<prep>.+)_s(?P<seed>\d+)\.json$')
METRICS = ['top1', 'top5', 'top10', 'macro_f1', 'weighted_f1']
PREP_ORDER = ['baseline', 'clahe', 'sobel', 'clahe_sobel']


def load(ckpt_dir):
    """-> data[model][prep][seed] = {metric: value, ...}; also params/sizes."""
    data = defaultdict(lambda: defaultdict(dict))
    meta = defaultdict(dict)
    for path in glob.glob(os.path.join(ckpt_dir, 'results_*_s*.json')):
        m = TAG_RE.match(os.path.basename(path))
        if not m:
            continue
        prep, seed = m['prep'], int(m['seed'])
        rec = json.load(open(path))
        for model, r in rec.items():
            t = r.get('test', {})
            vals = {k: t.get(k) for k in METRICS}
            if vals.get('top1') is None:
                continue
            data[model][prep][seed] = vals
            meta[model].setdefault('params', r.get('params'))
            meta[model].setdefault('fp32_mb', r.get('tflite_fp32_mb'))
            meta[model].setdefault('fp16_mb', r.get('tflite_fp16_mb'))
    return data, meta


def prep_sort_key(p):
    return (PREP_ORDER.index(p) if p in PREP_ORDER else 99, p)


def summarise(data):
    """-> rows: (model, prep, n_seeds, {metric: (mean, std)})"""
    rows = []
    for model in sorted(data):
        for prep in sorted(data[model], key=prep_sort_key):
            seeds = sorted(data[model][prep])
            stat = {}
            for k in METRICS:
                xs = np.array([data[model][prep][s][k] for s in seeds
                               if data[model][prep][s][k] is not None], float)
                stat[k] = (float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else 0.0)
            rows.append((model, prep, len(seeds), stat))
    return rows


def significance(data, base='baseline'):
    """Paired baseline-vs-variant tests per (model, prep!=base, metric)."""
    out = []
    for model in sorted(data):
        if base not in data[model]:
            continue
        for prep in sorted(data[model], key=prep_sort_key):
            if prep == base:
                continue
            common = sorted(set(data[model][base]) & set(data[model][prep]))
            if not common:
                continue
            for k in METRICS:
                b = np.array([data[model][base][s][k] for s in common], float)
                v = np.array([data[model][prep][s][k] for s in common], float)
                d = v - b
                mean_delta = float(d.mean())
                row = {'model': model, 'prep': prep, 'metric': k,
                       'n_seeds': len(common), 'mean_delta': mean_delta,
                       'baseline_mean': float(b.mean()), 'variant_mean': float(v.mean())}
                if HAVE_SCIPY and len(common) >= 2 and np.any(d != 0):
                    try:
                        row['t_p'] = float(stats.ttest_rel(v, b).pvalue)
                    except Exception:
                        row['t_p'] = None
                    try:
                        row['wilcoxon_p'] = float(stats.wilcoxon(v, b).pvalue) if len(common) >= 3 else None
                    except Exception:
                        row['wilcoxon_p'] = None
                else:
                    row['t_p'] = row['wilcoxon_p'] = None
                out.append(row)
    return out


def write_sig_csv(sig, out_dir):
    sigp = os.path.join(out_dir, 'preprocessing_significance.csv')
    with open(sigp, 'w') as f:
        f.write('model,preprocess_vs_baseline,metric,n_seeds,baseline_mean,'
                'variant_mean,mean_delta,paired_t_p,wilcoxon_p\n')
        for r in sig:
            tp = '' if r['t_p'] is None else f"{r['t_p']:.4g}"
            wp = '' if r['wilcoxon_p'] is None else f"{r['wilcoxon_p']:.4g}"
            f.write(f"{r['model']},{r['prep']},{r['metric']},{r['n_seeds']},"
                    f"{r['baseline_mean']:.4f},{r['variant_mean']:.4f},{r['mean_delta']:+.4f},"
                    f"{tp},{wp}\n")
    return sigp


def write_latex(rows, out_dir):
    # best (highest) test top1 per model gets bolded
    best = {}
    for model, prep, n, stat in rows:
        m = stat['top1'][0]
        if model not in best or m > best[model][1]:
            best[model] = (prep, m)
    texp = os.path.join(out_dir, 'preprocessing_comparison.tex')
    with open(texp, 'w') as f:
        f.write('% auto-generated: mean$\\pm$std over seeds; best Top-1 per model in bold\n')
        f.write('\\begin{tabular}{llrrrr}\n\\toprule\n')
        f.write('Model & Preproc. & Top-1 & Top-5 & Macro-F1 & $n$ \\\\\n\\midrule\n')
        last = None
        for model, prep, n, stat in rows:
            mshow = model if model != last else ''
            last = model
            def cell(k):
                mu, sd = stat[k]
                return f'{mu*100:.2f}\\,$\\pm$\\,{sd*100:.2f}'
            t1 = cell('top1')
            if best[model][0] == prep:
                t1 = f'\\textbf{{{t1}}}'
            f.write(f'{mshow} & {prep} & {t1} & {cell("top5")} & {cell("macro_f1")} & {n} \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')
    return texp


def write_figure(rows, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print('  (matplotlib unavailable, skipping figure:', repr(e)[:80], ')')
        return None
    models = sorted({r[0] for r in rows})
    preps = [p for p in PREP_ORDER if any(r[1] == p for r in rows)]
    x = np.arange(len(models)); w = 0.8 / max(1, len(preps))
    fig, ax = plt.subplots(figsize=(1.6 * len(models) + 2, 4))
    for j, prep in enumerate(preps):
        mus, sds = [], []
        for model in models:
            match = [r for r in rows if r[0] == model and r[1] == prep]
            if match:
                mu, sd = match[0][3]['top1']; mus.append(mu * 100); sds.append(sd * 100)
            else:
                mus.append(0); sds.append(0)
        ax.bar(x + j * w, mus, w, yerr=sds, capsize=3, label=prep)
    ax.set_xticks(x + w * (len(preps) - 1) / 2)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylabel('Test Top-1 (%)'); ax.set_title('Preprocessing comparison (mean $\\pm$ std over seeds)')
    ax.legend(title='preprocess', fontsize=8); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    figp = os.path.join(out_dir, 'preprocessing_comparison.png')
    fig.savefig(figp, dpi=200); plt.close(fig)
    return figp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default='/scratch/ttran72/checkpoints')
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or args.ckpt_dir

    data, meta = load(args.ckpt_dir)
    if not data:
        print('No results_<prep>_s<seed>.json found in', args.ckpt_dir); return

    rows = summarise(data)
    sig = significance(data)

    # console summary
    print(f'{"model":22s} {"prep":12s} {"n":>2s}  {"top1":>16s}  {"macroF1":>16s}')
    for model, prep, n, stat in rows:
        t = stat['top1']; f1 = stat['macro_f1']
        print(f'{model:22s} {prep:12s} {n:2d}  '
              f'{t[0]*100:6.2f} +/- {t[1]*100:4.2f}    {f1[0]*100:6.2f} +/- {f1[1]*100:4.2f}')

    comp = os.path.join(out_dir, 'preprocessing_comparison.csv')
    # write comparison csv
    with open(comp, 'w') as f:
        f.write('model,preprocess,n_seeds,' + ','.join(f'{k}_mean,{k}_std' for k in METRICS) + '\n')
        for model, prep, n, stat in rows:
            f.write(f'{model},{prep},{n},' +
                    ','.join(f'{stat[k][0]:.4f},{stat[k][1]:.4f}' for k in METRICS) + '\n')
    sigp = write_sig_csv(sig, out_dir)
    texp = write_latex(rows, out_dir)
    figp = write_figure(rows, out_dir)

    print('\nwrote:')
    for p in [comp, sigp, texp, figp]:
        if p: print('  ', p)

    if not HAVE_SCIPY:
        print('\n(scipy not installed -> p-values skipped; pip install scipy to enable)')
    n_seeds = max((n for _, _, n, _ in rows), default=0)
    if n_seeds < 5:
        print(f'\nNOTE: only {n_seeds} seed(s) detected. For Q1 significance claims, run >=5 seeds.')


if __name__ == '__main__':
    main()
