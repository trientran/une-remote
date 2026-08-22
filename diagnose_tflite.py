#!/usr/bin/env python3
"""Diagnose why a metadata'd tflite returns no results in ML Kit.

Runs the model exactly the way ML Kit's Task Library does (resize to the input
HxW, then (pixel-mean)/std using the EMBEDDED NormalizationOptions), and reports:

  1. output scale  -> logits vs probability distribution
       ML Kit reports the raw output tensor as "confidence". If these are logits
       (unbounded / negative / don't sum to 1), the confidence values are unusable
       and results get filtered -> "nothing at any confidence level".
       FIX: re-export with a softmax head (export_softmax_tflite.py).

  2. label count vs output classes
       If the embedded label file length != N, ML Kit mis-maps/drops results.
       FIX: python make_labels.py ... --expect <N>  then re-run write_metadata.py.

  3. top-5 predictions on your image
       Sanity-check top-1 against the KNOWN species of the image you pass. If it's
       wrong, normalisation or label order is off (not a softmax problem).

Run in the aiedge venv (Python 3.11): ai-edge-litert + tflite-support + torch/PIL.

Usage:
    python diagnose_tflite.py --model mobilenetv2_100_baseline_s42_float16_meta.tflite \
        --image /path/to/a_known_species.jpg
    python diagnose_tflite.py --model ..._meta.tflite            # no image: scale check only
"""
import argparse, json, sys
import numpy as np


def load_interp(path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except Exception:
        from tensorflow.lite import Interpreter
    it = Interpreter(model_path=path); it.allocate_tensors()
    return it


def read_meta(path):
    """Return (mean, std, labels) from embedded metadata, or (None,None,None)."""
    try:
        from tflite_support import metadata as _m
    except Exception as e:
        print('  (tflite-support metadata reader unavailable:', repr(e)[:80], ')')
        return None, None, None
    disp = _m.MetadataDisplayer.with_model_file(path)
    meta = json.loads(disp.get_metadata_json())
    mean = std = None
    try:
        for proc in meta['subgraph_metadata'][0]['input_tensor_metadata'][0]['process_units']:
            opt = proc.get('options', {})
            if 'mean' in opt and 'std' in opt:
                mean, std = opt['mean'], opt['std']
    except Exception:
        pass
    labels = None
    for fname in disp.get_packed_associated_file_list():
        buf = disp.get_associated_file_buffer(fname)
        labels = buf.decode('utf-8').splitlines()
        break
    return mean, std, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--image', default=None, help='a known-species image (best signal)')
    ap.add_argument('--labels', default=None, help='override label file (else read embedded)')
    args = ap.parse_args()

    it = load_interp(args.model)
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    ishape, oshape = list(inp['shape']), list(out['shape'])
    N = oshape[-1]
    print(f'input : shape={ishape} dtype={inp["dtype"].__name__}')
    print(f'output: shape={oshape} dtype={out["dtype"].__name__}   -> N classes = {N}')

    nhwc = len(ishape) == 4 and ishape[-1] == 3
    print(f'layout: {"NHWC OK" if nhwc else "NOT NHWC/C=3 -> ML Kit will reject this model"}')

    mean, std, labels = read_meta(args.model)
    if args.labels:
        labels = open(args.labels, encoding='utf-8').read().splitlines()
    print(f'metadata mean={mean}  std={std}')

    # ---- check 2: label count vs output classes ----
    if labels is None:
        print('LABELS: none found in metadata -> ML Kit returns bare indices, no names.')
    elif len(labels) != N:
        print(f'*** LABEL COUNT MISMATCH: {len(labels)} labels but {N} output classes. ***')
        print('    This alone can produce empty results. Regenerate with --expect', N)
    else:
        print(f'LABELS: {len(labels)} == N  (count OK)')

    if mean is None or std is None:
        print('*** No NormalizationOptions in metadata -> the load error would still occur. ***')
        sys.exit(1)

    H, W = (ishape[1], ishape[2]) if nhwc else (ishape[2], ishape[3])

    # ---- build input the ML Kit way ----
    if args.image:
        from PIL import Image
        img = Image.open(args.image).convert('RGB').resize((W, H))   # plain resize, like ML Kit
        px = np.asarray(img, np.float32)                              # HWC, 0..255
    else:
        px = np.full((H, W, 3), 127.0, np.float32)                   # neutral grey
        print('\n(no --image: using grey; scale check only, top-k not meaningful)')

    norm = (px - np.array(mean, np.float32)) / np.array(std, np.float32)
    x = norm[None, ...] if nhwc else norm.transpose(2, 0, 1)[None, ...]
    it.set_tensor(inp['index'], x.astype(inp['dtype']))
    it.invoke()
    y = it.get_tensor(out['index']).astype(np.float32).ravel()

    # ---- check 1: logits vs probabilities ----
    s_min, s_max, s_sum = float(y.min()), float(y.max()), float(y.sum())
    looks_prob = (s_min >= -1e-3) and (s_max <= 1.0 + 1e-3) and (abs(s_sum - 1.0) < 1e-2)
    print(f'\noutput scale: min={s_min:.4f}  max={s_max:.4f}  sum={s_sum:.4f}')
    if looks_prob:
        print('  -> looks like a PROBABILITY distribution (good for ML Kit).')
    else:
        print('  -> looks like RAW LOGITS (not [0,1], does not sum to 1).')
        print('     *** This is the usual cause of "nothing at any confidence". ***')
        print('     FIX: re-export with a softmax head -> export_softmax_tflite.py')

    # ---- check 3: top-5 ----
    top = y.argsort()[::-1][:5]
    print('\ntop-5:')
    for i in top:
        name = labels[i] if labels and i < len(labels) else f'<idx {i}>'
        print(f'  {i:5d}  {y[i]:+.4f}  {name}')
    if args.image:
        print('\n-> Is top-1 the correct species for this image?')
        print('   yes = normalisation/labels are fine; only the softmax/scale needs fixing.')
        print('   no  = wrong normalisation or label order (softmax will NOT fix that).')


if __name__ == '__main__':
    main()
