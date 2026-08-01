#!/usr/bin/env python3
"""MobileViT -> TFLite (fp32 + true fp16) via litert-torch. Run in the aiedge venv.

Why this exists: onnx2tf mangles MobileViT's attention (NCHW->NHWC transposes one
Mul operand but not the other). litert-torch converts straight from torch.export
and lowers cleanly. fp16 is produced through the tf.lite SavedModel path with real
tf enums (optimizations=DEFAULT + supported_types=[tf.float16]); the raw
numpy.float16 flag silently no-ops and yields fp32-in-disguise.

Working venv (Python 3.11):
    torch==2.6.0  torchvision==0.21.0  litert-torch==0.8.0  torchao==0.11.0  tensorflow-cpu

Usage (usually invoked by the notebook via the aiedge venv python):
    python export_mobilevit_litert.py --run-tag baseline --num-classes 2721
"""
import os, json, argparse

import torch
import timm

try:
    import litert_torch as lrt
except Exception:
    import ai_edge_torch as lrt
assert hasattr(lrt, 'convert'), 'no usable converter (need litert_torch.convert)'

# Real tf enums are required for genuine fp16; without tensorflow, fp16 is skipped.
try:
    import tensorflow as tf
    FP16_FLAGS = {'optimizations': [tf.lite.Optimize.DEFAULT],
                  'target_spec.supported_types': [tf.float16]}
    HAVE_TF = True
except Exception:
    FP16_FLAGS, HAVE_TF = None, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default='/scratch/ttran72/checkpoints')
    ap.add_argument('--run-tag',  default='baseline')
    ap.add_argument('--input-size', type=int, default=224)
    ap.add_argument('--num-classes', type=int, default=2721)
    ap.add_argument('--models', nargs='+', default=['mobilevit_xxs'])
    args = ap.parse_args()

    CKPT, TAG, SZ = args.ckpt_dir, args.run_tag, args.input_size
    print(f'converter: {lrt.__name__} | tensorflow: {"yes" if HAVE_TF else "NO (fp16 skipped)"}')

    results_path = os.path.join(CKPT, f'results_{TAG}.json')
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}

    for name in args.models:
        print('=' * 60); print(f'{name} [{TAG}]'); print('=' * 60)
        weights = os.path.join(CKPT, f'{name}_{TAG}.pt')
        if not os.path.exists(weights):
            print(f'  SKIP: weights not found: {weights}'); continue

        model = timm.create_model(name, pretrained=False, num_classes=args.num_classes)
        model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        model.eval().cpu()
        with torch.no_grad():
            sample = (torch.randn(1, 3, SZ, SZ),)

        rec = results.get(name, {})

        # fp32
        try:
            p32 = os.path.join(CKPT, f'{name}_{TAG}_float32.tflite')
            lrt.convert(model, sample).export(p32)
            rec['tflite_fp32_mb'] = round(os.path.getsize(p32) / 1e6, 2)
            print(f'  fp32 -> {p32}  ({rec["tflite_fp32_mb"]} MB)')
        except Exception as e:
            print(f'  fp32 convert failed: {repr(e)[:300]}')

        # fp16 (true half-precision weights; needs tensorflow for the enums)
        if HAVE_TF:
            try:
                p16 = os.path.join(CKPT, f'{name}_{TAG}_float16.tflite')
                lrt.convert(model, sample, _ai_edge_converter_flags=FP16_FLAGS).export(p16)
                rec['tflite_fp16_mb'] = round(os.path.getsize(p16) / 1e6, 2)
                fp32 = rec.get('tflite_fp32_mb')
                ratio = (rec['tflite_fp16_mb'] / fp32) if fp32 else None
                tag = '' if ratio is None else (
                    '  OK (true fp16)' if ratio < 0.65 else '  SUSPECT: not actually fp16')
                print(f'  fp16 -> {p16}  ({rec["tflite_fp16_mb"]} MB'
                      + (f', ratio {ratio:.2f}{tag}' if ratio else '') + ')')
            except Exception as e:
                print(f'  fp16 convert failed: {repr(e)[:300]}')
        else:
            print('  fp16 skipped: install tensorflow-cpu in this venv for true fp16')

        results[name] = rec

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print('\nupdated', results_path)


if __name__ == '__main__':
    main()
