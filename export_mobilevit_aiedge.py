#!/usr/bin/env python3
"""MobileViT -> TFLite via ai-edge-torch (direct PyTorch path, no ONNX detour).

onnx2tf mangles MobileViT's attention block: its NCHW->NHWC pass transposes one
operand of an attention-score Mul but not the other, giving
"Dimensions must be equal, but are 16 and 4". ai-edge-torch converts straight
from torch.export and never inserts those transposes, so the attention survives.

Run in a SEPARATE venv from the training/onnx2tf one (ai-edge-torch pins torch==2.4.*):
    python3.9 -m venv ~/venvs/aiedge
    source ~/venvs/aiedge/bin/activate
    pip install torch==2.4.1 timm ai-edge-torch

Usage:
    python export_mobilevit_aiedge.py
    python export_mobilevit_aiedge.py --run-tag baseline --input-size 224
"""
import os, json, argparse

import torch
import timm
import tensorflow as tf
import ai_edge_torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default='/scratch/ttran72/checkpoints')
    ap.add_argument('--run-tag',  default='baseline')
    ap.add_argument('--input-size', type=int, default=224)
    ap.add_argument('--num-classes', type=int, default=2721)
    ap.add_argument('--models', nargs='+', default=['mobilevit_xxs'])
    args = ap.parse_args()

    CKPT, TAG, SZ = args.ckpt_dir, args.run_tag, args.input_size

    results_path = os.path.join(CKPT, f'results_{TAG}.json')
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}

    for name in args.models:
        print('=' * 60); print(name); print('=' * 60)
        weights = os.path.join(CKPT, f'{name}_{TAG}.pt')
        if not os.path.exists(weights):
            print(f'  SKIP: weights not found: {weights}'); continue

        model = timm.create_model(name, pretrained=False, num_classes=args.num_classes)
        model.load_state_dict(torch.load(weights, map_location='cpu'))
        model.eval().cpu()

        sample = (torch.randn(1, 3, SZ, SZ),)   # NCHW, native PyTorch layout
        rec = results.get(name, {})

        # fp32
        try:
            edge = ai_edge_torch.convert(model, sample)
            p32 = os.path.join(CKPT, f'{name}_{TAG}_float32.tflite')
            edge.export(p32)
            rec['tflite_fp32_mb'] = round(os.path.getsize(p32) / 1e6, 2)
            print(f'  fp32 -> {p32}')
        except Exception as e:
            print(f'  fp32 convert failed: {repr(e)[:300]}')

        # fp16
        try:
            edge16 = ai_edge_torch.convert(
                model, sample,
                _ai_edge_converter_flags={'target_spec.supported_types': [tf.float16]},
            )
            p16 = os.path.join(CKPT, f'{name}_{TAG}_float16.tflite')
            edge16.export(p16)
            rec['tflite_fp16_mb'] = round(os.path.getsize(p16) / 1e6, 2)
            print(f'  fp16 -> {p16}')
        except Exception as e:
            print(f'  fp16 convert failed: {repr(e)[:300]}')

        results[name] = rec
        print(f"  fp16={rec.get('tflite_fp16_mb','-')} MB  fp32={rec.get('tflite_fp32_mb','-')} MB")

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print('\nupdated', results_path)


if __name__ == '__main__':
    main()
