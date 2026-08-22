#!/usr/bin/env python3
"""Generate labels.txt in the EXACT training class-index order.

ML Kit returns class index N; the app maps N -> species via this file. Line N
(0-based) MUST equal CLASS_NAMES[N] from the notebook, or every prediction maps
to the wrong species with no error.

Where the order comes from (notebook 02, Cell 12 -> Cell 9 filter_min_images):
    full = datasets.ImageFolder(DATA_DIR)      # full.classes = sorted(dir names)
    keep = sorted(y for y, c in Counter(labels) if c >= MIN_IMAGES_PER_CLASS)
    CLASS_NAMES = [full.classes[y] for y in keep]
i.e. alphabetically-sorted class-folder names, restricted to classes with
>= 25 images, in that same sorted order.

IMPORTANT: this order depends ONLY on the dataset + the >=25 filter. It does NOT
depend on seed or preprocess. => ONE labels.txt is correct for ALL your models,
seeds and preprocess variants.

This CANNOT be reconstructed from the results_*.json / *_history.csv you already
downloaded (they don't store the class list). You need ONE of:
  (A) the split cache pickle  /scratch/ttran72/cache/split_cache_s42.pkl
      (grab it from Carmack: scp ...:/scratch/ttran72/cache/split_cache_s42.pkl .)
      -> this is the object literally used at train/eval time: zero drift. PREFER THIS.
  (B) the dataset itself (ImageFolder root) to re-derive it deterministically.

Usage:
    python make_labels.py --cache split_cache_s42.pkl                # (A) preferred
    python make_labels.py --data-dir /path/to/viet-medi-species-2026 # (B) fallback
    python make_labels.py --cache split_cache_s42.pkl --expect 2721  # assert count
"""
import argparse, os, pickle, sys
from collections import Counter

MIN_IMAGES_PER_CLASS = 25   # MUST match the notebook (Cell 3)


def from_cache(path):
    with open(path, 'rb') as f:
        cache = pickle.load(f)
    names = cache['class_names']
    print(f'  source: split cache  {path}  (seed={cache.get("key", {}).get("seed")})')
    return names


def from_dataset(data_dir):
    # Re-derive bit-identically to the notebook. torchvision only; no timm/torch-cuda.
    from torchvision import datasets
    full = datasets.ImageFolder(data_dir)                    # .classes = sorted dir names
    counts = Counter(y for _, y in full.samples)
    keep = sorted(y for y, c in counts.items() if c >= MIN_IMAGES_PER_CLASS)
    names = [full.classes[y] for y in keep]
    print(f'  source: dataset  {data_dir}  '
          f'({len(full.classes)} raw -> {len(names)} kept at >= {MIN_IMAGES_PER_CLASS})')
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', help='split_cache_s<seed>.pkl (preferred, zero-drift)')
    ap.add_argument('--data-dir', help='ImageFolder root (fallback re-derivation)')
    ap.add_argument('--out', default='labels.txt')
    ap.add_argument('--expect', type=int, default=None,
                    help='assert this many classes (e.g. your model output dim, 2721)')
    args = ap.parse_args()

    if args.cache and os.path.exists(args.cache):
        names = from_cache(args.cache)
    elif args.data_dir:
        names = from_dataset(args.data_dir)
    else:
        sys.exit('give --cache <pkl> (preferred) or --data-dir <ImageFolder root>. '
                 'The class list is NOT in the json/csv you downloaded.')

    if args.expect is not None and len(names) != args.expect:
        sys.exit(f'ERROR: got {len(names)} classes but --expect {args.expect}. '
                 f'Wrong cache/dataset or a filter mismatch -- do NOT ship this labels.txt.')

    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(names) + '\n')

    print(f'wrote {args.out}: {len(names)} labels')
    print('  first 3:', names[:3])
    print('  last  3:', names[-3:])
    print('  -> line N (0-based) = class index N. Same file works for every model/seed/prep.')


if __name__ == '__main__':
    main()
