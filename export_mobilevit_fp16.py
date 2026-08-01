#!/usr/bin/env python3
"""True fp16 TFLite for MobileViT via litert-torch's tf.lite SavedModel path.

The earlier run mislabelled fp32 as fp16 because (a) no tensorflow was importable
so it used numpy.float16, and (b) it never set `optimizations`. The documented,
reliable fp16 path needs BOTH flags with real TF enums:
    optimizations = [tf.lite.Optimize.DEFAULT]
    target_spec.supported_types = [tf.float16]
litert-torch forwards these to the tf.lite.TFLiteConverter it runs on its internal
SavedModel, which is exactly the path onnx2tf used to give clean fp16 for the other
two models.

Requires tensorflow importable (pip install tensorflow-cpu). Same venv as the
working fp32 export (torch 2.6.0 / litert-torch 0.8.0 / torchao 0.11.0 /
torchvision 0.21.0).

Usage:
    python export_mobilevit_fp16.py
"""
import os, json, argparse

import torch
import timm
import tensorflow as tf          # real enums; fail loudly if missing

try:
    import litert_torch as lrt
except Exception:
    import ai_edge_torch as lrt
assert hasattr(lrt, 'convert'), 'no usable converter (need litert_torch.convert)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default='/scratch/ttran72/checkpoints')
    ap.add_argument('--run-tag',  default='baseline')
    ap.add_argument('--input-size', type=int, default=224)
    ap.add_argument('--num-classes', type=int, default=2721)
    ap.add_argument('--models', nargs='+', default=['mobilevit_xxs'])
    args = ap.parse_args()

    CKPT, TAG, SZ = args.ckpt_dir, args.run_tag, args.input_size
    print(f'converter module: {lrt.__name__}   tensorflow: {tf.__version__}')

    results_path = os.path.join(CKPT, f'results_{TAG}.json')
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}

    fp16_flags = {
        'optimizations': [tf.lite.Optimize.DEFAULT],
        'target_spec.supported_types': [tf.float16],
    }

    for name in args.models:
        print('=' * 60); print(name); print('=' * 60)
        weights = os.path.join(CKPT, f'{name}_{TAG}.pt')
        if not os.path.exists(weights):
            print(f'  SKIP: weights not found: {weights}'); continue

        model = timm.create_model(name, pretrained=False, num_classes=args.num_classes)
        model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        model.eval().cpu()
        with torch.no_grad():
            sample = (torch.randn(1, 3, SZ, SZ),)

        rec = results.get(name, {})
        try:
            edge16 = lrt.convert(model, sample, _ai_edge_converter_flags=fp16_flags)
            p16 = os.path.join(CKPT, f'{name}_{TAG}_float16.tflite')
            edge16.export(p16)
            size16 = round(os.path.getsize(p16) / 1e6, 2)
            rec['tflite_fp16_mb'] = size16
            print(f'  fp16 -> {p16}  ({size16} MB)')

            # sanity: real fp16 should be ~half of fp32
            fp32 = rec.get('tflite_fp32_mb')
            if fp32:
                ratio = size16 / fp32
                verdict = 'OK (true fp16)' if ratio < 0.65 else 'SUSPECT: not actually fp16'
                print(f'  fp16/fp32 size ratio = {ratio:.2f}  -> {verdict}')
        except Exception as e:
            print(f'  fp16 convert failed: {repr(e)[:400]}')

        results[name] = rec

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print('\nupdated', results_path)


if __name__ == '__main__':
    main()
