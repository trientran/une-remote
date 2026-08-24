#!/usr/bin/env python3
"""Re-export the CNN-path models to TFLite WITH a softmax head (ML Kit wants a
probability distribution, not raw logits).

Identical toolchain to export_tflite.py (onnx2tf, offline calibration shim, CPU
export) so the resulting sizes stay comparable to your reported numbers -- the ONLY
change is a torch.nn.Softmax appended to the model before ONNX export. Softmax is
monotonic, so argmax / Top-k / accuracy are unchanged; only the score semantics
change (now 0..1, summing to 1), which is what ML Kit needs.

Outputs go to a parallel *_sm_tf/ dir so they don't clobber your logit tflite files:
    {name}_{TAG}_sm_tf/*_float16.tflite   etc.
Then attach metadata:
    python write_metadata.py --labels labels.txt --dir {CKPT}/{name}_{TAG}_sm_tf

MobileViT: don't use this script (onnx2tf mangles its attention). Instead add softmax
to reexport_mobilevit_nhwc.py's wrapper -- change NHWCWrapper.forward to:
    return torch.softmax(self.model(x.permute(0,3,1,2).contiguous()), dim=1)
so you get NHWC input + softmax output in one shot.

Run in the training venv (same one export_tflite.py uses).

Usage:
    python export_softmax_tflite.py --run-tag baseline_s42 --num-classes 2721
"""
import os, sys, json, glob, argparse, subprocess

import numpy as np
import torch
import torch.nn as nn
import timm

CALIB_NAME = 'calibration_image_sample_data_20x128x128x3_float32.npy'


def ensure_calibration_file(work_dir):
    path = os.path.join(work_dir, CALIB_NAME)
    if not os.path.exists(path):
        np.save(path, np.zeros((20, 128, 128, 3), dtype=np.float32))
        print(f'  wrote dummy calibration file -> {path}')
    return path


class WithSoftmax(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        return self.sm(self.model(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default='/scratch/ttran72/checkpoints')
    ap.add_argument('--run-tag',  default='baseline_s42')
    ap.add_argument('--input-size', type=int, default=224)
    ap.add_argument('--num-classes', type=int, default=2721)
    ap.add_argument('--models', nargs='+',
                    default=['mobilenetv2_100', 'efficientformerv2_s0'])   # NOT mobilevit
    ap.add_argument('--opset', type=int, default=17)
    args = ap.parse_args()

    CKPT, TAG, SZ = args.ckpt_dir, args.run_tag, args.input_size

    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    'onnxscript', 'onnx2tf', 'onnx', 'onnxruntime',
                    'onnx-graphsurgeon', 'sng4onnx', 'onnxslim', 'ai-edge-litert'])
    ensure_calibration_file(CKPT)

    results_path = os.path.join(CKPT, f'results_{TAG}.json')
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}

    for name in args.models:
        print('=' * 60); print(f'{name} [{TAG}] + softmax'); print('=' * 60)
        weights = os.path.join(CKPT, f'{name}_{TAG}.pt')
        if not os.path.exists(weights):
            print(f'  SKIP: weights not found: {weights}'); continue

        model = timm.create_model(name, pretrained=False, num_classes=args.num_classes)
        model.load_state_dict(torch.load(weights, map_location='cpu'))
        model.eval().cpu()
        wrapped = WithSoftmax(model).eval().cpu()

        onnx_path = os.path.join(CKPT, f'{name}_{TAG}_sm.onnx')
        dummy = torch.randn(1, 3, SZ, SZ)
        try:
            torch.onnx.export(wrapped, dummy, onnx_path, opset_version=args.opset,
                              input_names=['input'], output_names=['probs'], dynamo=False)
            print(f'  ONNX saved -> {onnx_path}')
        except Exception as e:
            print(f'  ONNX export failed: {repr(e)[:200]}'); continue

        tf_dir = os.path.join(CKPT, f'{name}_{TAG}_sm_tf')
        try:
            subprocess.run(['onnx2tf', '-i', onnx_path, '-o', tf_dir, '-n', '-rtpo', 'Erf', 'GeLU'],
               check=True, cwd=CKPT)
        except Exception as e:
            print(f'  onnx2tf failed: {repr(e)[:200]}'); continue

        rec = results.get(name, {})
        for f in os.listdir(tf_dir):
            p = os.path.join(tf_dir, f)
            if f.endswith('_float16.tflite'):
                rec['tflite_sm_fp16_mb'] = round(os.path.getsize(p) / 1e6, 2)
            elif f.endswith('_float32.tflite'):
                rec['tflite_sm_fp32_mb'] = round(os.path.getsize(p) / 1e6, 2)
        results[name] = rec
        print(f"  fp16={rec.get('tflite_sm_fp16_mb','-')} MB  fp32={rec.get('tflite_sm_fp32_mb','-')} MB")

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print('\nupdated', results_path)
    tfl = glob.glob(os.path.join(CKPT, f'*_{TAG}_sm_tf', '*.tflite'))
    print('softmax tflite files:', len(tfl))
    for t in sorted(tfl): print('  ', t)
    print('\nnext: python write_metadata.py --labels labels.txt --dir', os.path.dirname(tfl[0]) if tfl else CKPT)


if __name__ == '__main__':
    main()
