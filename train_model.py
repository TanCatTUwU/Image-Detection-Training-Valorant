from __future__ import annotations
import argparse
import hashlib
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
import yaml
from PIL import Image
from training_metrics import TrainingMetrics, valid_epoch
PROJECT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
RESERVED = {'con', 'prn', 'aux', 'nul', 'clock$', 'conin$', 'conout$'}
RESERVED.update((f'{prefix}{number}' for prefix in ('com', 'lpt') for number in '123456789¹²³'))

@dataclass
class DatasetAudit:
    root: Path
    split_dirs: dict[str, list[Path]]
    images: dict[str, list[Path]]
    class_counts: dict[str, list[int]]
    backgrounds: dict[str, int]

def local_path(value: str, relative_to: Path) -> Path:
    if not isinstance(value, str) or not value.strip() or '://' in value:
        raise ValueError(f'Expected a local path, got {value!r}')
    path = Path(value).expanduser()
    return (relative_to / path).resolve() if not path.is_absolute() else path.resolve()

def label_path(image: Path) -> Path:
    parts = list(image.parts)
    indices = [i for i, part in enumerate(parts[:-1]) if part == 'images']
    if not indices:
        raise ValueError(f"Image directory must contain an 'images' folder: {image}")
    parts[indices[-1]] = 'labels'
    return Path(*parts).with_suffix('.txt')

def audit_dataset(data_path: Path) -> DatasetAudit:
    data_path = Path(data_path).resolve()
    with data_path.open(encoding='utf-8-sig') as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError('Dataset YAML must be a mapping')
    if 'download' in data:
        raise ValueError('Dataset download instructions are disabled; use local directories')
    names = data.get('names')
    if names not in (['Head', 'Body'], {0: 'Head', 1: 'Body'}):
        raise ValueError('Dataset names must be exactly 0: Head and 1: Body')
    if 'nc' in data and data['nc'] != 2:
        raise ValueError('Dataset nc must be 2')
    root = local_path(data.get('path', '.'), data_path.parent)
    report = DatasetAudit(root, {}, {}, {}, {})
    seen_paths: dict[Path, str] = {}
    seen_hashes: dict[str, str] = {}
    for split in ('train', 'val'):
        values = data.get(split)
        values = values if isinstance(values, list) else [values]
        if not values:
            raise ValueError(f'{split} requires at least one local image directory')
        directories = [local_path(value, root) for value in values]
        for directory in directories:
            if not directory.is_dir():
                raise ValueError(f'{split} image directory does not exist: {directory}')
        images = sorted((path.resolve() for directory in directories for path in directory.rglob('*') if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES))
        if not images:
            raise ValueError(f'{split} is empty: no images; add your labeled data first')
        report.split_dirs[split] = directories
        report.images[split] = images
        report.class_counts[split] = [0, 0]
        report.backgrounds[split] = 0
        seen_labels: set[Path] = set()
        for image in images:
            if image in seen_paths:
                raise ValueError(f'Duplicate image/overlap in {seen_paths[image]} and {split}: {image}')
            seen_paths[image] = split
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in seen_hashes and seen_hashes[digest] != split:
                raise ValueError(f'Duplicate image content in both train and val: {image}')
            seen_hashes[digest] = split
            try:
                with Image.open(image) as opened:
                    if min(opened.size) < 10:
                        raise ValueError('image dimensions must be at least 10 pixels')
                    opened.verify()
            except Exception as exc:
                raise ValueError(f'Unreadable image {image}: {exc}') from exc
            label = label_path(image)
            if label in seen_labels:
                raise ValueError(f'Multiple images use the same label: {label}')
            seen_labels.add(label)
            if not label.is_file():
                raise ValueError(f'Missing label: {label}; use an empty .txt for a background image')
            label_text = label.read_text(encoding='utf-8')
            if label_text.startswith('\ufeff'):
                raise ValueError(f'{label}: save the label as UTF-8 without BOM; Ultralytics treats a BOM-prefixed class as corrupt. No file was rewritten.')
            rows = [line.strip() for line in label_text.splitlines() if line.strip()]
            if not rows:
                report.backgrounds[split] += 1
            for number, row in enumerate(rows, 1):
                fields = row.split()
                if len(fields) != 5 or fields[0] not in ('0', '1'):
                    raise ValueError(f'{label}:{number}: expected class(0/1) x_center y_center width height')
                try:
                    x, y, width, height = map(float, fields[1:])
                except ValueError as exc:
                    raise ValueError(f'{label}:{number}: box values must be numbers') from exc
                if not all((math.isfinite(value) for value in (x, y, width, height))):
                    raise ValueError(f'{label}:{number}: box values must be finite')
                if not (0 <= x <= 1 and 0 <= y <= 1 and (0 < width <= 1) and (0 < height <= 1)):
                    raise ValueError(f'{label}:{number}: box values must be normalized, with positive size')
                if min(x - width / 2, y - height / 2) < -1e-06 or max(x + width / 2, y + height / 2) > 1 + 1e-06:
                    raise ValueError(f'{label}:{number}: box extends outside the image')
                report.class_counts[split][int(fields[0])] += 1
    return report

def choose_device(requested: str, torch_module) -> str:
    if not isinstance(requested, str) or not re.fullmatch(r'auto|cpu|cuda(?::\d+)?', requested):
        raise ValueError('Training device must be cpu, auto, cuda, or cuda:N')
    if requested == 'cpu':
        return 'cpu'
    index = int(requested.split(':')[1]) if ':' in requested else 0
    try:
        if not torch_module.cuda.is_available() or index >= torch_module.cuda.device_count():
            raise RuntimeError('The selected NVIDIA CUDA GPU is unavailable')
        if float(torch_module.ones(1, device=f'cuda:{index}').add(1).cpu()[0]) != 2:
            raise RuntimeError('The NVIDIA GPU did not pass its operation check')
    except (OSError, RuntimeError, AssertionError) as exc:
        if requested == 'auto':
            return 'cpu'
        raise ValueError(f'CUDA training is unavailable: {exc}. Select Auto or CPU, or run setup.bat again.') from exc
    return str(index)

def validate_run_name(name: str) -> None:
    if not re.fullmatch('[a-z0-9][a-z0-9_-]{0,63}', name) or name.split('.')[0].lower() in RESERVED:
        raise ValueError('Run name must be 1-64 lowercase ASCII letters/digits/hyphens/underscores, and cannot be a Windows device name')

def reserve_run_directory(parent: Path, name: str) -> Path:
    validate_run_name(name)
    parent = Path(parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    for suffix in range(1, 10000):
        candidate = parent / (name if suffix == 1 else f'{name}-{suffix}')
        if candidate.resolve().parent != parent:
            raise ValueError(f'Run path leaves output directory: {candidate}')
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise ValueError('Too many existing run directories; choose another --name')

def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be greater than zero')
    return number

def validate_augmentation_options(options):
    for key, upper in (('mosaic', 1), ('mixup', 1), ('degrees', 180), ('fliplr', 1)):
        value = options[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= upper or not math.isfinite(value):
            raise ValueError(f'{key} must be a finite number from 0 to {upper}')
    value = options['patience']
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 1000:
        raise ValueError('patience must be an integer from 0 to 1000; 0 disables early stopping')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=str(PROJECT / 'training-data/dataset.yaml'))
    parser.add_argument('--weights', default=str(PROJECT / 'best.pt'), help='Existing local .pt checkpoint')
    parser.add_argument('--check-data', action='store_true', help='Audit data only; no Torch import or output files')
    parser.add_argument('--dry-run', action='store_true', help='Audit data and show a plan; no training or Torch import')
    parser.add_argument('--device', default='auto', help='auto uses a working NVIDIA CUDA GPU, otherwise CPU')
    parser.add_argument('--epochs', type=positive, default=30)
    parser.add_argument('--batch', type=positive, default=4)
    parser.add_argument('--imgsz', type=positive, default=320)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--threads', type=positive, default=2)
    parser.add_argument('--mosaic', type=float, default=0.3)
    parser.add_argument('--mixup', type=float, default=0)
    parser.add_argument('--degrees', type=float, default=0)
    parser.add_argument('--fliplr', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=10, help='Epochs without improvement before stopping; 0 disables early stopping')
    parser.add_argument('--name', default='train-head-body')
    parser.add_argument('--export', action='store_true', help='Export the trained candidate to static 320px ONNX on CPU')
    args = parser.parse_args(argv)
    try:
        validate_augmentation_options(vars(args))
    except ValueError as exc:
        parser.error(str(exc))
    if args.workers < 0 or args.imgsz % 32:
        parser.error('--workers must be nonnegative and --imgsz must be a positive multiple of 32')
    if not re.fullmatch('auto|cpu|cuda(?::\\d+)?', args.device):
        parser.error('--device must be auto, cpu, cuda, or cuda:N; DirectML is inference only')
    return args

def main(argv=None, *, progress_callback=None) -> int:
    args = parse_args(argv)
    try:
        validate_run_name(args.name)
        report = audit_dataset(Path(args.data))
        for split in ('train', 'val'):
            head, body = report.class_counts[split]
            print(f'{split}: {len(report.images[split])} images, {head} Head boxes, {body} Body boxes, {report.backgrounds[split]} backgrounds')
            if not head or not body:
                print(f'Warning: {split} has no examples for one or both classes.')
        if args.check_data:
            print('Dataset check passed. No training started.')
            return 0
        weights = local_path(args.weights, Path.cwd())
        if weights.suffix.lower() != '.pt' or not weights.is_file():
            raise ValueError(f'A local .pt checkpoint is required (no downloads): {weights}')
        print(f'Fine-tune {weights} at {args.imgsz}px: epochs={args.epochs}, batch={args.batch}, workers={args.workers}, CPU threads={args.threads}, requested device={args.device}')
        if args.dry_run:
            print('Dry run only. Device availability has not been checked; no output directory created.')
            return 0
        os.environ['YOLO_CONFIG_DIR'] = str(PROJECT / 'runtime-data')
        os.environ['YOLO_AUTOINSTALL'] = 'false'
        os.environ['YOLO_OFFLINE'] = 'true'
        os.environ['WANDB_MODE'] = 'disabled'
        os.environ['COMET_MODE'] = 'DISABLED'
        os.environ['OMP_NUM_THREADS'] = str(args.threads)
        os.environ['MKL_NUM_THREADS'] = str(args.threads)
        import torch
        device = choose_device(args.device, torch)
        torch.set_num_threads(args.threads)
        from ultralytics import YOLO, settings
        from ultralytics.data import utils as data_utils
        settings.update({key: False for key in ('sync', 'clearml', 'comet', 'dvc', 'mlflow', 'raytune', 'tensorboard', 'wandb')})
        data_utils.check_font = lambda *a, **kw: None
        output_parent = PROJECT / 'training-runs'
        if output_parent.resolve().parent != PROJECT:
            raise ValueError('training-runs must stay inside this project')
        run = reserve_run_directory(output_parent, args.name)
        resolved_yaml = run / 'dataset-resolved.yaml'
        resolved_yaml.write_text(yaml.safe_dump({'path': str(report.root), 'train': [str(path) for path in report.split_dirs['train']], 'val': [str(path) for path in report.split_dirs['val']], 'names': {0: 'Head', 1: 'Body'}}, sort_keys=False), encoding='utf-8')
        device_label = 'CPU' if device == 'cpu' else f'NVIDIA {torch.cuda.get_device_name(int(device))} · CUDA {device}'
        if device == 'cpu' and args.device == 'auto':
            device_label += ' · automatic fallback'
        print(f'Training on {device_label}; new candidate run: {run}')
        if progress_callback:
            progress_callback({'run_path': str(run), 'message': f'Training on {device_label}', 'resolved_device': device})
        model = YOLO(str(weights), task='detect')
        observed_epoch = None
        best_epoch = None
        metrics_reader = TrainingMetrics(PROJECT)
        if progress_callback:

            def report_epoch(trainer):
                nonlocal observed_epoch, best_epoch
                measured = metrics_reader.snapshot(run)
                epoch = measured['latest']['epoch'] if measured['latest'] else valid_epoch(int(trainer.epoch) + 1)
                if epoch is None or epoch > args.epochs:
                    return
                observed_epoch = epoch
                best_epoch = valid_epoch(getattr(getattr(trainer, 'stopper', None), 'best_epoch', None)) or best_epoch
                progress_callback({'epoch': observed_epoch, 'best_epoch': best_epoch, 'message': f'Epoch {observed_epoch}/{args.epochs}'})
            model.add_callback('on_fit_epoch_end', report_epoch)
        model.train(data=str(resolved_yaml), epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, device=device, workers=args.workers, project=str(run.parent), name=run.name, exist_ok=True, resume=False, pretrained=True, save=True, val=True, plots=False, cache=False, amp=False,
                    mosaic=args.mosaic, mixup=args.mixup, degrees=args.degrees, fliplr=args.fliplr, patience=args.patience)
        best = run / 'weights/best.pt'
        if not best.is_file():
            raise ValueError(f'Training did not produce a best checkpoint: {best}')
        print(f'Trained candidate: {best}')
        if progress_callback:
            trainer = getattr(model, 'trainer', None)
            trainer_epoch = getattr(trainer, 'epoch', None)
            measured = metrics_reader.snapshot(run)
            final_epoch = (measured['latest']['epoch'] if measured['latest'] else None) or (valid_epoch(trainer_epoch + 1) if isinstance(trainer_epoch, int) else None) or observed_epoch or args.epochs
            best_epoch = valid_epoch(getattr(getattr(trainer, 'stopper', None), 'best_epoch', None)) or best_epoch or measured['best_epoch']
            progress_callback({'checkpoint': str(best), 'epoch': final_epoch, 'best_epoch': best_epoch, 'stopped_early': final_epoch < args.epochs})
        if args.export:
            destination = best.with_suffix('.onnx')
            if destination.exists():
                raise ValueError(f'Refusing to overwrite an existing candidate: {destination}')
            exported = YOLO(str(best), task='detect').export(format='onnx', imgsz=320, batch=1, device='cpu', dynamic=False, simplify=True, opset=17, quantize=None, nms=False)
            print(f'Candidate ONNX: {exported}')
        return 0
    except (ValueError, OSError, yaml.YAMLError, ImportError) as exc:
        print(f'Training setup failed: {exc}', file=sys.stderr)
        return 2
if __name__ == '__main__':
    raise SystemExit(main())
