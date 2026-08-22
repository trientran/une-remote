#!/usr/bin/env python3
"""Re-export mobilevit_xxs to NHWC-input TFLite so ML Kit / Task Library accepts it.

Problem: litert-torch preserves PyTorch's NCHW input [1,3,H,W]. ML Kit requires
NHWC [1,H,W,3]. So the existing mobilevit_xxs_*_float*.tflite cannot be loaded by
ImageClassifier and cannot take image metadata the normal way.

Fix: wrap the trained model so its forward takes an NHWC tensor and permutes to NCHW
internally, then export THAT. The exported graph's input is [1,H,W,3] -- exactly what
ML Kit wants -- and the ImageNet-vs-none normalisation is unchanged (MobileViT uses
plain /255 scaling: metadata mean=[0,0,0], std=[255,255,255], handled by write_metadata.py).

Run in the aiedge venv (Python 3.11: torch 2.6, litert-torch, tensorflow-cpu), same as
export_mobilevit_litert.py. After this, run write_metadata.py on the *_nhwc_* outputs.

Usage:
    python reexport_mobilevit_nhwc.py --run-tag baseline_s42 --num-classes 2721
    python reexport_mobilevit_nhwc.py --ckpt-dir /path --run-tag baseline_s42
"""
import argparse, os, json

import torch
import timm

try:
    import litert_torch as lrt
except Exception:
    import ai_edge_torch as lrt
assert hasattr(lrt, 'convert'), 'need litert_torch.convert (aiedge venv)'

try:
    import tensorflow as tf
    FP16_FLAGS = {'optimizations': [tf.lite.Optimize.DEFAULT],
                  'target_spec.supported_types': [tf.float16]}
    HAVE_TF = True
except Exception:
    FP16_FLAGS, HAVE_TF = None, False


class NHWCWrapper(torch.nn.Module):
    """Accept NHWC [B,H,W,3]; permute to NCHW for the real model."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):                      # x: [B,H,W,C]
        return torch.softmax(self.model(x.permute(0, 3, 1, 2).contiguous()), dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default='/scratch/ttran72/checkpoints')
    ap.add_argument('--run-tag',  default='baseline_s42')
    ap.add_argument('--input-size', type=int, default=224)
    ap.add_argument('--num-classes', type=int, default=2721)
    ap.add_argument('--models', nargs='+', default=['mobilevit_xxs'])
    args = ap.parse_args()

    CKPT, TAG, SZ = args.ckpt_dir, args.run_tag, args.input_size
    print(f'converter: {lrt.__name__} | tensorflow: {"yes" if HAVE_TF else "NO (fp16 skipped)"}')

    results_path = os.path.join(CKPT, f'results_{TAG}.json')
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}

    for name in args.models:
        print('=' * 60); print(f'{name} [{TAG}] -> NHWC'); print('=' * 60)
        weights = os.path.join(CKPT, f'{name}_{TAG}.pt')
        if not os.path.exists(weights):
            print(f'  SKIP: weights not found: {weights}'); continue

        model = timm.create_model(name, pretrained=False, num_classes=args.num_classes)
        model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        model.eval().cpu()
        wrapped = NHWCWrapper(model).eval().cpu()

        with torch.no_grad():
            sample = (torch.randn(1, SZ, SZ, 3),)     # NHWC sample

        rec = results.get(name, {})

        # fp32
        try:
            p32 = os.path.join(CKPT, f'{name}_{TAG}_nhwc_float32.tflite')
            lrt.convert(wrapped, sample).export(p32)
            rec['tflite_nhwc_fp32_mb'] = round(os.path.getsize(p32) / 1e6, 2)
            print(f'  fp32 -> {p32}  ({rec["tflite_nhwc_fp32_mb"]} MB)')
        except Exception as e:
            print(f'  fp32 convert failed: {repr(e)[:300]}')

        # fp16
        if HAVE_TF:
            try:
                p16 = os.path.join(CKPT, f'{name}_{TAG}_nhwc_float16.tflite')
                lrt.convert(wrapped, sample, _ai_edge_converter_flags=FP16_FLAGS).export(p16)
                rec['tflite_nhwc_fp16_mb'] = round(os.path.getsize(p16) / 1e6, 2)
                print(f'  fp16 -> {p16}  ({rec["tflite_nhwc_fp16_mb"]} MB)')
            except Exception as e:
                print(f'  fp16 convert failed: {repr(e)[:300]}')
        else:
            print('  fp16 skipped: no tensorflow in this venv')

        results[name] = rec

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print('\nupdated', results_path)
    print('next: python write_metadata.py --labels labels.txt --files '
          f'{os.path.join(CKPT, args.models[0] + "_" + TAG + "_nhwc_float16.tflite")}')


if __name__ == '__main__':
    main()
