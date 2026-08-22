#!/usr/bin/env python3
"""Embed ML Kit metadata (NormalizationOptions + label map) into the tflite files.

Fixes the runtime error:
    Input tensor has type kTfLiteFloat32: it requires specifying
    NormalizationOptions metadata to preprocess input images.

The TFLite Task Library / ML Kit ImageClassifier does resize+normalise for you,
but for a FLOAT32 input it refuses to guess the normalisation -- it must be in the
model metadata. This writes it, per model, plus the label file.

CRITICAL -- normalisation is PER MODEL (verified against timm's resolved data config
for each backbone; the notebook uses timm's create_transform, so these are the exact
stats each model was trained/evaluated with):

    mobilenetv2_100       ImageNet  mean_01=(.485,.456,.406) std_01=(.229,.224,.225)
    efficientformerv2_s0  ImageNet  mean_01=(.485,.456,.406) std_01=(.229,.224,.225)
    mobilevit_xxs         NONE      mean_01=(0,0,0)          std_01=(1,1,1)   <-- different!

ML Kit's NormalizationOptions operates on RAW pixels in [0,255] as (p - mean)/std,
whereas timm normalises the [0,1]-scaled tensor. So the metadata stats are the
timm stats * 255:
    mean_255 = 255 * mean_01      std_255 = 255 * std_01
=> CNNs:      mean=[123.675,116.28,103.53]  std=[58.395,57.12,57.375]
   mobilevit: mean=[0,0,0]                  std=[255,255,255]   (i.e. plain /255 scaling)
Baking ImageNet stats into MobileViT would silently wreck its predictions.

INPUT LAYOUT: ML Kit requires NHWC, BxHxWxC, C=3. onnx2tf emits NHWC (the CNNs are
fine). litert-torch keeps PyTorch's NCHW [1,3,224,224], so the mobilevit_xxs files
will FAIL this check here and are SKIPPED with a message -- re-export them NHWC first
(see reexport_mobilevit_nhwc.py) before writing metadata.

Requires the REAL tflite-support (0.4.x) -> Python 3.10/3.11 (your aiedge venv):
    pip install "tflite-support==0.4.4"
(The 0.1.0a1 stub that pip may pick on 3.12 has no metadata_writers.)

Usage:
    python write_metadata.py --labels labels.txt --dir /path/with/tflite/files
    python write_metadata.py --labels labels.txt --files a_float16.tflite b_float32.tflite
    python write_metadata.py --labels labels.txt --dir . --suffix _meta   # default suffix
"""
import argparse, glob, os, sys

# model-name prefix -> (mean_01, std_01). Extend here if you add backbones.
STATS_01 = {
    'mobilenetv2_100':      ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'efficientformerv2_s0': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'mobilevit_xxs':        ((0.0, 0.0, 0.0),        (1.0, 1.0, 1.0)),
}


def model_key(fname):
    base = os.path.basename(fname)
    for key in sorted(STATS_01, key=len, reverse=True):   # longest prefix wins
        if base.startswith(key):
            return key
    return None


def input_is_nhwc_c3(model_path):
    """Return (ok, shape). ok=True iff input is [1,H,W,3]. Uses whatever interpreter exists."""
    Interp = None
    try:
        from ai_edge_litert.interpreter import Interpreter as Interp
    except Exception:
        try:
            from tensorflow.lite import Interpreter as Interp
        except Exception:
            try:
                from tflite_runtime.interpreter import Interpreter as Interp
            except Exception:
                Interp = None
    if Interp is None:
        return None, None    # can't check; caller decides
    it = Interp(model_path=model_path); it.allocate_tensors()
    shape = list(it.get_input_details()[0]['shape'])
    ok = len(shape) == 4 and shape[0] == 1 and shape[-1] == 3
    return ok, shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', required=True, help='labels.txt (from make_labels.py)')
    ap.add_argument('--dir', help='directory to scan for *.tflite')
    ap.add_argument('--files', nargs='+', help='explicit tflite files')
    ap.add_argument('--suffix', default='_meta', help='output suffix (default: _meta)')
    ap.add_argument('--out-dir', default=None, help='write outputs here (default: alongside input)')
    args = ap.parse_args()

    # real writer only (the stub lacks this import and will fail loudly here)
    try:
        from tflite_support.metadata_writers import image_classifier, writer_utils
    except Exception as e:
        sys.exit('tflite-support metadata_writers not available. You have the stub or '
                 'wrong Python. Use Python 3.10/3.11 and: pip install "tflite-support==0.4.4"\n'
                 f'  ({e!r})')

    if not os.path.exists(args.labels):
        sys.exit(f'labels file not found: {args.labels} (run make_labels.py first)')
    n_labels = sum(1 for _ in open(args.labels, encoding='utf-8'))

    files = list(args.files or [])
    if args.dir:
        files += sorted(glob.glob(os.path.join(args.dir, '*.tflite')))
    files = [f for f in files if args.suffix not in os.path.basename(f)]   # don't re-process outputs
    if not files:
        sys.exit('no input .tflite files (use --dir or --files)')

    done, skipped, failed = [], [], []
    for path in files:
        name = os.path.basename(path)
        key = model_key(path)
        if key is None:
            print(f'SKIP  {name}: filename does not start with a known model prefix '
                  f'{list(STATS_01)}'); skipped.append(name); continue

        ok, shape = input_is_nhwc_c3(path)
        if ok is False:
            print(f'SKIP  {name}: input shape {shape} is not NHWC/C=3 -- ML Kit needs '
                  f'[1,H,W,3]. Re-export NHWC (reexport_mobilevit_nhwc.py) then re-run.')
            skipped.append(name); continue
        if ok is None:
            print(f'  note {name}: no interpreter to verify input layout; proceeding on '
                  f'filename ({key}). If this is mobilevit_xxs, confirm it is NHWC first.')

        mean_01, std_01 = STATS_01[key]
        mean = [round(m * 255.0, 4) for m in mean_01]
        std  = [round(s * 255.0, 4) for s in std_01]

        try:
            writer = image_classifier.MetadataWriter.create_for_inference(
                writer_utils.load_file(path),
                input_norm_mean=mean, input_norm_std=std,
                label_file_paths=[args.labels])
            out_dir = args.out_dir or os.path.dirname(path) or '.'
            os.makedirs(out_dir, exist_ok=True)
            stem, ext = os.path.splitext(name)
            out_path = os.path.join(out_dir, f'{stem}{args.suffix}{ext}')
            writer_utils.save_file(writer.populate(), out_path)
            print(f'OK    {name}  [{key}]  mean={mean} std={std}  labels={n_labels}  -> {os.path.basename(out_path)}')
            done.append(out_path)
        except Exception as e:
            print(f'FAIL  {name}: {e!r}'); failed.append(name)

    print(f'\n{len(done)} written, {len(skipped)} skipped, {len(failed)} failed.')
    if done:
        print('Verify one with:  python -c "from tflite_support import metadata as m; '
              'print(m.MetadataDisplayer.with_model_file(\'%s\').get_metadata_json())"' % done[0])
    if skipped:
        print('Skipped files still need attention (usually mobilevit_xxs NHWC re-export).')


if __name__ == '__main__':
    main()
