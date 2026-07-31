#!/usr/bin/env python3
"""Standalone ONNX + TFLite export for the trained mobile-transformer models.

Loads the saved *_<run_tag>.pt weights from CKPT_DIR, exports each to ONNX (on CPU,
which is what the notebook run got wrong), then converts to TFLite via onnx2tf.
Writes fp16/fp32 sizes into results_<run_tag>.json (merged, not overwritten).

Usage:
    python export_tflite.py                 # defaults below
    python export_tflite.py --run-tag baseline --input-size 224
"""
import os, sys, json, glob, argparse, subprocess

import torch
import timm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default='/scratch/ttran72/checkpoints')
    ap.add_argument('--run-tag',  default='baseline')
    ap.add_argument('--input-size', type=int, default=224)
    ap.add_argument('--num-classes', type=int, default=2721)   # from results_baseline.json
    ap.add_argument('--models', nargs='+',
                    default=['mobilenetv2_100', 'mobilevit_xxs', 'efficientformerv2_s0'])
    ap.add_argument('--opset', type=int, default=17)
    args = ap.parse_args()

    CKPT, TAG, SZ = args.ckpt_dir, args.run_tag, args.input_size

    # make sure onnx2tf and its helper packages are present
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    'onnxscript', 'onnx2tf', 'onnx', 'onnxruntime',
                    'onnx-graphsurgeon', 'sng4onnx', 'onnxslim', 'ai-edge-litert'])

    results_path = os.path.join(CKPT, f'results_{TAG}.json')
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}

    for name in args.models:
        print('=' * 60); print(name); print('=' * 60)
        weights = os.path.join(CKPT, f'{name}_{TAG}.pt')
        if not os.path.exists(weights):
            print(f'  SKIP: weights not found: {weights}'); continue

        # rebuild architecture and load the trained weights, on CPU
        model = timm.create_model(name, pretrained=False, num_classes=args.num_classes)
        state = torch.load(weights, map_location='cpu')
        model.load_state_dict(state)
        model.eval().cpu()                       # <-- key fix: model on CPU

        onnx_path = os.path.join(CKPT, f'{name}_{TAG}.onnx')
        dummy = torch.randn(1, 3, SZ, SZ)        # <-- CPU dummy, matches the CPU model
        try:
            torch.onnx.export(model, dummy, onnx_path, opset_version=args.opset,
                              input_names=['input'], output_names=['logits'], dynamo=False)
            print(f'  ONNX saved -> {onnx_path}')
        except Exception as e:
            print(f'  ONNX export failed: {repr(e)[:200]}'); continue

        tf_dir = os.path.join(CKPT, f'{name}_{TAG}_tf')
        try:
            subprocess.run(['onnx2tf', '-i', onnx_path, '-o', tf_dir, '-n'], check=True)
        except Exception as e:
            print(f'  onnx2tf failed: {repr(e)[:200]}'); continue

        rec = results.get(name, {})
        for f in os.listdir(tf_dir):
            p = os.path.join(tf_dir, f)
            if f.endswith('_float16.tflite'):
                rec['tflite_fp16_mb'] = round(os.path.getsize(p) / 1e6, 2)
            elif f.endswith('_float32.tflite'):
                rec['tflite_fp32_mb'] = round(os.path.getsize(p) / 1e6, 2)
        results[name] = rec
        print(f"  fp16={rec.get('tflite_fp16_mb','-')} MB  fp32={rec.get('tflite_fp32_mb','-')} MB")

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print('\nupdated', results_path)
    tfl = glob.glob(os.path.join(CKPT, f'*_{TAG}_tf', '*.tflite'))
    print('tflite files:', len(tfl))
    for t in sorted(tfl): print('  ', t)

if __name__ == '__main__':
    main()
