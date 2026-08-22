#!/usr/bin/env python3
"""Attach ML Kit / TFLite Task Library metadata to the exported classifier tflites.

Fixes: "Input tensor has type kTfLiteFloat32: it requires specifying
NormalizationOptions metadata to preprocess input images."

Writes NormalizationOptions (ImageNet mean/std, since training used timm's default
create_transform with no override) + the label map (AssociatedFile, TENSOR_AXIS_LABELS)
+ an optional score threshold, per the ML Kit LiteRT spec.

Labels come from the SAME split cache the models were trained against, so the
label order is guaranteed to match the model's class indices.

Env:  pip install tflite-support
Usage:
    python add_metadata.py                      # all *_float*.tflite under CKPT_DIR
    python add_metadata.py --only mobilenetv2_100_baseline_s42_float16.tflite
"""
import os, glob, argparse, pickle

from tflite_support.metadata_writers import image_classifier, writer_utils

# ImageNet stats on a 0-255 input (timm default create_transform used mean/std on 0-1;
# the metadata normalizes the raw 0-255 image, so scale the stats by 255).
MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]   # [123.675, 116.28, 103.53]
STD  = [0.229 * 255, 0.224 * 255, 0.225 * 255]   # [58.395, 57.12, 57.375]


def write_labels(cache_path, out_path):
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    names = cache['class_names']            # exact training index order
    with open(out_path, 'w') as f:
        f.write('\n'.join(names) + '\n')
    print(f'labels.txt: {len(names)} classes -> {out_path}')
    return out_path


def add_meta(tflite_in, labels_txt, score_threshold):
    out = tflite_in.replace('.tflite', '_meta.tflite')
    writer = image_classifier.MetadataWriter.create_for_inference(
        writer_utils.load_file(tflite_in),
        input_norm_mean=MEAN, input_norm_std=STD,
        label_file_paths=[labels_txt],
        score_calibration_md=None)
    writer_utils.save_file(writer.populate(), out)
    print(f'  {os.path.basename(tflite_in)} -> {os.path.basename(out)}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default='/scratch/ttran72/checkpoints')
    ap.add_argument('--cache', default='/scratch/ttran72/cache/split_cache_s42.pkl')
    ap.add_argument('--score-threshold', type=float, default=0.0)   # ML Kit optional
    ap.add_argument('--only', default=None, help='single tflite filename (in ckpt-dir)')
    args = ap.parse_args()

    labels_txt = write_labels(args.cache, os.path.join(args.ckpt_dir, 'labels.txt'))

    if args.only:
        files = [os.path.join(args.ckpt_dir, args.only)]
    else:
        # every classifier tflite (CNN outputs live in *_tf/ subdirs; mobilevit sits flat)
        files = sorted(set(
            glob.glob(os.path.join(args.ckpt_dir, '*_tf', '*_float*.tflite')) +
            glob.glob(os.path.join(args.ckpt_dir, 'mobilevit_xxs_*_float*.tflite'))))
        files = [f for f in files if '_meta.tflite' not in f]

    print(f'\nattaching metadata to {len(files)} model(s):')
    ok = 0
    for f in files:
        try:
            add_meta(f, labels_txt, args.score_threshold); ok += 1
        except Exception as e:
            print(f'  FAILED {os.path.basename(f)}: {repr(e)[:200]}')
    print(f'\ndone: {ok}/{len(files)} models now have *_meta.tflite')


if __name__ == '__main__':
    main()
