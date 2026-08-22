#!/usr/bin/env python3
"""Diagnose why a metadata'd tflite returns no results in ML Kit.

Key question this answers: are the model outputs a PROBABILITY distribution (what
ML Kit wants) or RAW LOGITS (the usual cause of "nothing at any confidence")?

This does NOT require tflite_support. If the reader is present it will pull the
embedded mean/std/labels for you; otherwise pass them explicitly. Note: a failure to
READ embedded metadata does NOT mean the metadata is absent -- if your app loads the
model without the normalization error, the metadata is there. We only need the right
normalisation to reproduce ML Kit's preprocessing here.

Run in a venv with ai-edge-litert (or tensorflow) + torch/PIL (your aiedge venv).

Usage (MobileNetV2 / EfficientFormerV2 -> ImageNet stats):
    python diagnose_tflite.py --model ..._meta.tflite --image known_species.jpg \
        --labels labels.txt --norm imagenet

MobileViT (no normalisation, plain /255):
    python diagnose_tflite.py --model ..._meta.tflite --image known_species.jpg \
        --labels labels.txt --norm none

Explicit override (0..255 scale, as ML Kit applies (pixel-mean)/std):
    ... --mean 123.675,116.28,103.53 --std 58.395,57.12,57.375
"""
import argparse, json, sys
import numpy as np

# 0..255-scale presets = 255 * timm(0..1) stats
PRESETS = {
    'imagenet': ([123.675, 116.28, 103.53], [58.395, 57.12, 57.375]),  # mobilenetv2, efficientformerv2
    'none':     ([0.0, 0.0, 0.0],           [255.0, 255.0, 255.0]),      # mobilevit_xxs
}


def load_interp(path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except Exception:
        from tensorflow.lite import Interpreter
    it = Interpreter(model_path=path); it.allocate_tensors()
    return it


def try_read_embedded(path):
    """Best-effort embedded read; returns (mean, std, labels) or (None,None,None).
    Missing reader != missing metadata."""
    try:
        from tflite_support import metadata as _m
    except Exception:
        print('  note: tflite_support not in this venv -> reading stats from CLI instead '
              '(this does NOT mean the model lacks metadata).')
        return None, None, None
    try:
        disp = _m.MetadataDisplayer.with_model_file(path)
        meta = json.loads(disp.get_metadata_json())
        mean = std = None
        for proc in meta['subgraph_metadata'][0]['input_tensor_metadata'][0].get('process_units', []):
            opt = proc.get('options', {})
            if 'mean' in opt and 'std' in opt:
                mean, std = opt['mean'], opt['std']
        labels = None
        for fname in disp.get_packed_associated_file_list():
            labels = disp.get_associated_file_buffer(fname).decode('utf-8').splitlines(); break
        return mean, std, labels
    except Exception as e:
        print('  note: embedded read failed:', repr(e)[:80]); return None, None, None


def parse_vec(s):
    return [float(x) for x in s.split(',')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--image', default=None)
    ap.add_argument('--labels', default=None)
    ap.add_argument('--norm', choices=list(PRESETS), default=None,
                    help='imagenet (mobilenet/efficientformer) or none (mobilevit)')
    ap.add_argument('--mean', default=None, help='comma-sep, 0..255 scale')
    ap.add_argument('--std', default=None, help='comma-sep, 0..255 scale')
    args = ap.parse_args()

    it = load_interp(args.model)
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    ishape, oshape = [int(v) for v in inp['shape']], [int(v) for v in out['shape']]
    N = oshape[-1]
    print(f'input : shape={ishape} dtype={inp["dtype"].__name__}')
    print(f'output: shape={oshape} dtype={out["dtype"].__name__}   -> N classes = {N}')
    nhwc = len(ishape) == 4 and ishape[-1] == 3
    print(f'layout: {"NHWC OK" if nhwc else "NOT NHWC/C=3 -> ML Kit rejects this model"}')

    e_mean, e_std, e_labels = try_read_embedded(args.model)

    # resolve normalisation: explicit > preset > embedded
    if args.mean and args.std:
        mean, std = parse_vec(args.mean), parse_vec(args.std)
    elif args.norm:
        mean, std = PRESETS[args.norm]
    else:
        mean, std = e_mean, e_std
    if mean is None or std is None:
        sys.exit('need normalisation: pass --norm imagenet|none or --mean/--std '
                 '(or run in a venv with tflite_support to read embedded).')
    print(f'using mean={mean}  std={std}')

    labels = None
    if args.labels:
        labels = open(args.labels, encoding='utf-8').read().splitlines()
    elif e_labels:
        labels = e_labels
    if labels is None:
        print('LABELS: none (top-k will show indices only). Pass --labels labels.txt for names.')
    elif len(labels) != N:
        print(f'*** LABEL COUNT MISMATCH: {len(labels)} labels vs {N} classes -> '
              f'mis-map/empty results. Regenerate with --expect {N}. ***')
    else:
        print(f'LABELS: {len(labels)} == N (OK)')

    H, W = (ishape[1], ishape[2]) if nhwc else (ishape[2], ishape[3])
    if args.image:
        from PIL import Image
        px = np.asarray(Image.open(args.image).convert('RGB').resize((W, H)), np.float32)
    else:
        px = np.full((H, W, 3), 127.0, np.float32)
        print('\n(no --image: grey input; scale check only, top-k not meaningful)')

    norm = (px - np.array(mean, np.float32)) / np.array(std, np.float32)
    x = norm[None, ...] if nhwc else norm.transpose(2, 0, 1)[None, ...]
    it.set_tensor(inp['index'], x.astype(inp['dtype'])); it.invoke()
    y = it.get_tensor(out['index']).astype(np.float32).ravel()

    s_min, s_max, s_sum = float(y.min()), float(y.max()), float(y.sum())
    looks_prob = s_min >= -1e-3 and s_max <= 1.0 + 1e-3 and abs(s_sum - 1.0) < 1e-2
    print(f'\noutput scale: min={s_min:.4f}  max={s_max:.4f}  sum={s_sum:.4f}')
    if looks_prob:
        print('  -> PROBABILITY distribution (good). If ML Kit still shows nothing, the')
        print('     issue is label count or the app-side threshold, not the output.')
    else:
        print('  -> RAW LOGITS (not [0,1], sum != 1).  *** usual cause of empty results ***')
        print('     FIX: re-export with softmax -> export_softmax_tflite.py')

    top = y.argsort()[::-1][:5]
    print('\ntop-5:')
    for i in top:
        name = labels[i] if labels and i < len(labels) else f'<idx {i}>'
        print(f'  {i:5d}  {y[i]:+.4f}  {name}')
    if args.image:
        print('\n-> top-1 correct for this image? yes = norm/labels fine, only fix scale/softmax;'
              '  no = normalisation or label order wrong (softmax will NOT fix that).')


if __name__ == '__main__':
    main()
