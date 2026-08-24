#!/usr/bin/env python3
"""Parity check: does each DEPLOYED tflite agree with the trained PyTorch model?

For one (or more) known test image(s), run:
  (a) the trained PyTorch model (timm + the *_baseline_s42.pt weights), and
  (b) each deployed ML-Kit tflite (softmax head, CNNs NHWC / MobileViT NHWC wrapper),
and compare the top-1 class index. They MUST match, or the model you benchmarked is
not the model you shipped. This is the last gate before latency measurement.

Why this matters per model:
  - MobileNetV2, MobileViT : softmax + layout are exactly equivalent -> expect exact top-1 match.
  - EfficientFormerV2       : uses an APPROXIMATED GELU on-device (-rtpo Erf GeLU),
                              which is NOT bit-identical -> top-1 should still match, but
                              this is the one to watch. A tiny logit drift is fine; a
                              top-1 flip is a problem to report.

Applies the SAME preprocessing ML Kit applies, PER MODEL (from write_metadata.py):
  - mobilenetv2_100 / efficientformerv2_s0 : ImageNet mean/std (on 0..1), i.e. timm stats
  - mobilevit_xxs                          : none  (plain /255)
and the SAME resize. If PyTorch vs tflite disagree, it is almost always preprocessing
or label order, NOT the conversion.

Run in a venv with torch + timm + PIL + (ai_edge_litert OR tensorflow). Your aiedge venv
works. tflite reading uses ai_edge_litert.Interpreter or tf.lite.Interpreter.

Usage:
    python parity_check_deployed.py \
        --weights-dir /path/with/pt/files \
        --tflite-dir  /path/with/deployed/tflite \
        --images img1.jpg img2.jpg ... \
        --num-classes 2721 [--labels labels.txt]

You can point --weights-dir and --tflite-dir at the same folder; the script finds files
by name pattern. Only baseline_s42 is needed (deployment used baseline_s42).
"""
import argparse, glob, os, sys
import numpy as np

# timm (0..1-scale) normalisation per model -> we replicate ML Kit's preprocessing
STATS_01 = {
    'mobilenetv2_100':      ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'efficientformerv2_s0': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'mobilevit_xxs':        ((0.0, 0.0, 0.0),        (1.0, 1.0, 1.0)),
}
MODELS = ['mobilenetv2_100', 'efficientformerv2_s0', 'mobilevit_xxs']


def load_interp(path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except Exception:
        from tensorflow.lite import Interpreter
    it = Interpreter(model_path=path); it.allocate_tensors()
    return it


def find_one(patterns, root):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(root, '**', pat), recursive=True))
        if hits:
            return hits[0]
    return None


def preprocess(img_path, size, mean01, std01):
    from PIL import Image
    im = Image.open(img_path).convert('RGB').resize((size, size), Image.BILINEAR)
    x = np.asarray(im, np.float32) / 255.0                 # 0..1
    x = (x - np.array(mean01, np.float32)) / np.array(std01, np.float32)
    return x                                               # HWC, normalised


def run_torch(name, weights, num_classes, x_hwc):
    import torch, timm
    model = timm.create_model(name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    model.eval()
    x = torch.from_numpy(x_hwc.transpose(2, 0, 1)[None, ...]).float()  # NCHW
    with torch.no_grad():
        logits = model(x).numpy().ravel()
    return logits


def run_tflite(path, x_hwc):
    it = load_interp(path)
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    ishape = [int(v) for v in inp['shape']]
    nhwc = len(ishape) == 4 and ishape[-1] == 3
    x = x_hwc[None, ...] if nhwc else x_hwc.transpose(2, 0, 1)[None, ...]
    it.set_tensor(inp['index'], x.astype(inp['dtype'])); it.invoke()
    return it.get_tensor(out['index']).astype(np.float32).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights-dir', required=True)
    ap.add_argument('--tflite-dir', required=True)
    ap.add_argument('--images', nargs='+', required=True)
    ap.add_argument('--num-classes', type=int, default=2721)
    ap.add_argument('--input-size', type=int, default=224)
    ap.add_argument('--labels', default=None)
    ap.add_argument('--tag', default='baseline_s42')
    args = ap.parse_args()

    labels = None
    if args.labels and os.path.exists(args.labels):
        labels = open(args.labels, encoding='utf-8').read().splitlines()
        if len(labels) != args.num_classes:
            print(f'WARNING: labels count {len(labels)} != num_classes {args.num_classes}')

    def nm(i):
        return labels[i] if labels and 0 <= i < len(labels) else f'<idx {i}>'

    overall_ok = True
    for name in MODELS:
        print('=' * 66); print(name); print('=' * 66)
        weights = find_one([f'{name}_{args.tag}.pt'], args.weights_dir)
        # deployed tflites: CNNs -> *_sm_*_meta ; mobilevit -> *_nhwc_*_meta
        tfl_fp32 = find_one([f'{name}_{args.tag}_sm_float32_meta.tflite',
                             f'{name}_{args.tag}_nhwc_float32_meta.tflite'], args.tflite_dir)
        tfl_fp16 = find_one([f'{name}_{args.tag}_sm_float16_meta.tflite',
                             f'{name}_{args.tag}_nhwc_float16_meta.tflite'], args.tflite_dir)
        if not weights:
            print('  SKIP: no .pt weights found'); continue
        if not (tfl_fp32 or tfl_fp16):
            print('  SKIP: no deployed tflite found'); continue

        mean01, std01 = STATS_01[name]
        for img in args.images:
            if not os.path.exists(img):
                print(f'  image not found: {img}'); continue
            x = preprocess(img, args.input_size, mean01, std01)

            pt = run_torch(name, weights, args.num_classes, x)
            pt_top = int(pt.argmax())

            row = f'  {os.path.basename(img):28s} torch#{pt_top} {nm(pt_top)}'
            print(row)
            for tag, path in [('fp32', tfl_fp32), ('fp16', tfl_fp16)]:
                if not path:
                    continue
                y = run_tflite(path, x)
                t_top = int(y.argmax())
                s = float(y.sum())
                match = (t_top == pt_top)
                overall_ok &= match
                flag = 'MATCH' if match else '*** MISMATCH ***'
                probish = 'prob' if abs(s - 1.0) < 1e-2 else f'sum={s:.3f}(not prob!)'
                print(f'      {tag}: tflite#{t_top} {nm(t_top)}   {flag}   [{probish}]')

    print('=' * 66)
    if overall_ok:
        print('ALL MATCH -> deployed models agree with trained models. Safe to measure latency.')
    else:
        print('*** MISMATCHES above -> deployed model != benchmarked model. Do NOT report')
        print('    latency as if it were the benchmarked model until resolved (check per-model')
        print('    normalisation, resize interpolation, label order, or the GELU approximation).')
        sys.exit(1)


if __name__ == '__main__':
    main()
