from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

REQUIRED = {'settings.json', 'checkpoint.pt', 'requirements.txt', 'train.py', 'run.sh', 'dataset/dataset.yaml'}
RESERVED = {'con', 'prn', 'aux', 'nul', 'clock$', 'conin$', 'conout$'} | {f'{prefix}{n}' for prefix in ('com', 'lpt') for n in '123456789¹²³'}


def read_json(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def redact(root, value):
    text = str(value)
    paths = {str(root), str(Path(root).resolve()), str(Path.home()), Path(root).as_posix(), Path(root).resolve().as_posix()}
    paths.update(json.dumps(path)[1:-1] for path in tuple(paths))
    for path in sorted(paths, key=len, reverse=True):
        text = text.replace(path, '<local>')
    return re.sub(r'(?i)((?:token|api[_-]?key|authorization|password|secret)\s*[=:]\s*)(?:Bearer\s+)?[^\s,;\"\']+', r'\1<redacted>', text)


def telemetry(root, event, **values):
    try:
        path = root / 'outputs/events.jsonl'
        path.parent.mkdir(exist_ok=True)
        if path.exists() and path.stat().st_size > 4 * 1024 * 1024:
            path.replace(path.with_name('events.previous.jsonl'))
        record = {'time': time.time(), 'event': event, 'pid': os.getpid(), **values}
        with path.open('a', encoding='utf-8') as stream:
            stream.write(redact(root, json.dumps(record, allow_nan=False, default=str)) + '\n')
    except Exception as exc:
        print(f'Diagnostics event unavailable: {type(exc).__name__}', file=sys.stderr, flush=True)


def collect_environment(root):
    values = {'python': sys.version, 'platform': platform.system(), 'kernel': platform.release(), 'architecture': platform.machine(), 'cpu_count': os.cpu_count(), 'disk_free_bytes': shutil.disk_usage(root).free}
    values['packages'] = {}
    for name in ('torch', 'torchvision', 'ultralytics', 'onnx', 'onnxruntime', 'numpy'):
        try:
            values['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, check=True)
        values['gpu'] = result.stdout[:8192].strip()
    except Exception as exc:
        values['gpu_query_error'] = type(exc).__name__
    write_json(root / 'outputs/environment.json', values)


def snapshot_environment(root):
    try:
        collect_environment(root)
    except Exception as exc:
        telemetry(root, 'environment_unavailable', error_type=type(exc).__name__)


def attach_telemetry(root, model):
    if not callable(getattr(model, 'add_callback', None)):
        return
    def epoch_end(trainer):
        metrics = {}
        for key, value in getattr(trainer, 'metrics', {}).items():
            try:
                number = float(value)
                if math.isfinite(number):
                    metrics[str(key)] = number
            except (TypeError, ValueError):
                pass
        telemetry(root, 'epoch_completed', epoch=int(trainer.epoch) + 1, epochs=int(trainer.epochs), metrics=metrics)
    model.add_callback('on_fit_epoch_end', epoch_end)


def record_failure(root, exc, stage):
    try:
        value = {'time': time.time(), 'stage': stage, 'type': type(exc).__name__, 'message': redact(root, exc), 'traceback': redact(root, ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)))}
        write_json(root / 'outputs/error.json', value)
        telemetry(root, 'error', stage=stage, error_type=value['type'], message=value['message'])
        package_diagnostics(root)
    except Exception as diagnostic_error:
        print(f'Diagnostics unavailable: {type(diagnostic_error).__name__}', file=sys.stderr, flush=True)


def package_diagnostics(root):
    outputs = root / 'outputs'
    outputs.mkdir(exist_ok=True)
    names = ('environment.json', 'error.json', 'events.jsonl', 'events.previous.jsonl', 'worker.log', 'bootstrap.log', 'status.json', 'settings.json', 'versions.json', 'run.json', 'run/results.csv', 'run/args.yaml')
    temporary = outputs / 'diagnostics.zip.tmp'
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name in names:
            path = outputs / name
            if not path.is_file() or path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != root):
                continue
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size - 2 * 1024 * 1024))
                content = stream.read(2 * 1024 * 1024).decode('utf-8', errors='replace')
            archive.writestr(name, redact(root, content))
    temporary.replace(outputs / 'diagnostics.zip')
    return outputs / 'diagnostics.zip'


def save_diagnostics(root):
    try:
        return package_diagnostics(root)
    except Exception as exc:
        telemetry(root, 'diagnostics_unavailable', error_type=type(exc).__name__)
        return None


def safe_name(name):
    return isinstance(name, str) and re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', name) and name not in RESERVED


def archive_name(name):
    if not isinstance(name, str) or not name or '\\' in name or ':' in name:
        return False
    parts = name.split('/')
    if any(part in ('', '.', '..') or part.endswith((' ', '.')) or part.split('.')[0].lower() in RESERVED or any(ord(c) < 32 or c in '<>"|?*' for c in part) for part in parts):
        return False
    if name in REQUIRED:
        return True
    return len(parts) == 4 and parts[:2] in (['dataset', 'images'], ['dataset', 'labels']) and parts[2] in ('train', 'val') and Path(parts[3]).suffix.lower() in ({'.png', '.jpg', '.jpeg', '.webp', '.bmp'} if parts[1] == 'images' else {'.txt'})


def validate_bundle(root):
    root = Path(root).resolve()
    manifest_path = root / 'bundle.json'
    if manifest_path.is_symlink():
        raise ValueError('Bundle manifest must be a regular file')
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get('schema_version') != 1 or not isinstance(manifest.get('files'), dict):
        raise ValueError('Unsupported bundle manifest')
    files = manifest['files']
    if not REQUIRED.issubset(files):
        raise ValueError('Bundle is missing required inputs')
    for name, expected in files.items():
        if not archive_name(name) or not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
            raise ValueError(f'Invalid manifest file: {name}')
        path = root / name
        if not path.is_file() or path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != root) or digest(path) != expected:
            raise ValueError(f'Bundle file is missing or has changed: {name}')
    for path in root.rglob('*'):
        relative = path.relative_to(root)
        if relative.parts[0] in ('outputs', '.venv'):
            if (root / relative.parts[0]).is_symlink():
                raise ValueError('Runtime directories cannot be symbolic links')
            if relative.parts[0] == 'outputs' and path.is_symlink():
                raise ValueError('Output files and directories cannot be symbolic links')
            continue
        if path.is_symlink():
            raise ValueError(f'Bundle cannot contain symbolic links: {relative}')
        if path.is_file() and relative.as_posix() not in files and relative.as_posix() != 'bundle.json':
            raise ValueError(f'Unlisted bundle input: {relative}')
    settings = read_json(root / 'settings.json')
    validate_settings(settings)
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return settings, fingerprint


def validate_settings(settings):
    if not isinstance(settings, dict):
        raise ValueError('Settings must be a mapping')
    ranges = {'epochs': (1, 10000), 'workers': (0, 16), 'threads': (1, 128), 'model_width': (32, 4096), 'model_height': (32, 4096), 'patience': (0, 1000)}
    required = set(ranges) | {'batch', 'device', 'amp', 'name', 'mosaic', 'mixup', 'degrees', 'fliplr'}
    if not required.issubset(settings) or set(settings) - required - {'imgsz'}:
        raise ValueError('Missing or unsupported training settings')
    for key, (lower, upper) in ranges.items():
        value = settings[key]
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f'{key} must be an integer from {lower} to {upper}')
    if any(settings[key] % 32 for key in ('model_width', 'model_height')):
        raise ValueError('Model width and height must be multiples of 32')
    if 'imgsz' in settings and (type(settings['imgsz']) is not int or settings['imgsz'] != max(settings['model_width'], settings['model_height'])):
        raise ValueError('imgsz must match the larger selected dimension')
    if type(settings['batch']) is not int or settings['batch'] not in {-1, *range(1, 129)}:
        raise ValueError('batch must be -1 for auto or an integer from 1 to 128')
    if settings['device'] != 'cuda:0' or type(settings['amp']) is not bool or not safe_name(settings['name']):
        raise ValueError('CUDA device, AMP flag, or run name is invalid')
    for key, upper in (('mosaic', 1), ('mixup', 1), ('degrees', 180), ('fliplr', 1)):
        value = settings[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= upper:
            raise ValueError(f'{key} must be a finite number from 0 to {upper}')


def check_cuda(torch):
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError('No NVIDIA CUDA GPU is available')
        if float(torch.ones(1, device='cuda:0').add(1).cpu()[0]) != 2:
            raise RuntimeError('CUDA operation produced an incorrect result')
        torch.cuda.synchronize()
    except Exception as exc:
        raise RuntimeError(f'CUDA preflight failed: {exc}. Select a Linux CUDA PyTorch template.') from exc


def check_local_amp(model, torch):
    training = model.training
    try:
        model.eval()
        image = torch.zeros((1, 3, 256, 256), device='cuda:0')
        with torch.inference_mode():
            reference = model(image)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                mixed = model(image)
        while isinstance(reference, (tuple, list)):
            reference = reference[0]
        while isinstance(mixed, (tuple, list)):
            mixed = mixed[0]
        valid = reference.shape == mixed.shape and bool(torch.isfinite(reference).all()) and bool(torch.isfinite(mixed).all()) and torch.allclose(reference, mixed.float(), rtol=0.05, atol=0.5)
        print('AMP selected-checkpoint check passed.' if valid else 'AMP check found numerical differences; using FP32.', flush=True)
        return bool(valid)
    except Exception as exc:
        print(f'AMP check unavailable; using FP32: {exc}', flush=True)
        return False
    finally:
        model.train(training)


def environment(root, settings):
    values = {'YOLO_CONFIG_DIR': str(root / 'outputs/yolo-settings'), 'YOLO_AUTOINSTALL': 'false', 'YOLO_OFFLINE': 'true', 'WANDB_MODE': 'disabled', 'COMET_MODE': 'DISABLED', 'WANDB_DISABLED': 'true', 'OMP_NUM_THREADS': str(settings['threads']), 'MKL_NUM_THREADS': str(settings['threads']), 'PYTHONDONTWRITEBYTECODE': '1', 'PIP_DISABLE_PIP_VERSION_CHECK': '1', 'MPLCONFIGDIR': str(root / 'outputs/matplotlib')}
    os.environ.update(values)
    sys.dont_write_bytecode = True


def runtime_versions(torch):
    return {'python': f'{sys.version_info.major}.{sys.version_info.minor}', 'torch': str(torch.__version__), 'cuda': str(torch.version.cuda), **{name: importlib.metadata.version(name) for name in ('ultralytics', 'onnx', 'PyYAML')}}


def status(root, state, training_complete=False, **values):
    write_json(root / 'outputs/status.json', {'state': state, 'training_complete': training_complete, 'pid': os.getpid(), 'updated_at': time.time(), **values})
    telemetry(root, 'phase', state=state, training_complete=training_complete)


def validate_recovery(root, fingerprint, versions, resume, export_only):
    outputs = root / 'outputs'
    state_path = outputs / 'status.json'
    run_path = outputs / 'run.json'
    previous = read_json(state_path) if state_path.exists() else {}
    if previous.get('state') == 'completed':
        raise ValueError('This run is completed; extract a fresh bundle for another run')
    if not run_path.exists():
        if (outputs / 'run').exists():
            raise ValueError('Unrecognized run outputs; extract a fresh bundle')
        if export_only:
            raise ValueError('Export retry requires a completed training checkpoint')
        return 'new'
    run = read_json(run_path)
    if not resume and not export_only:
        raise ValueError('Existing run requires explicit --resume')
    if run.get('fingerprint') != fingerprint or run.get('versions') != versions:
        raise ValueError('Resume requires the same bundle, settings, and compatible runtime versions')
    if previous.get('training_complete'):
        if not (outputs / 'run/weights/best.pt').is_file():
            raise ValueError('Export retry requires best.pt')
        return 'export'
    if export_only:
        raise ValueError('Training is incomplete; use --resume')
    if not run.get('training_started'):
        return 'new'
    if not (outputs / 'run/weights/last.pt').is_file():
        raise ValueError('Interrupted run has no last.pt; extract a fresh bundle')
    return 'resume'


def prepare_dataset(root):
    import yaml
    source = root / 'dataset/dataset.yaml'
    data = yaml.safe_load(source.read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict) or set(data) - {'path', 'train', 'val', 'names', 'nc'} or data.get('names') not in (['Head', 'Body'], {0: 'Head', 1: 'Body'}) or data.get('nc', 2) != 2 or data.get('path', '.') != '.':
        raise ValueError('Dataset YAML must contain the local Head/Body train and val split')
    resolved = {'path': str(root / 'dataset'), 'names': ['Head', 'Body'], 'nc': 2}
    for split in ('train', 'val'):
        entries = data.get(split)
        entries = entries if isinstance(entries, list) else [entries]
        if not entries or any(entry != f'images/{split}' for entry in entries) or len(entries) != 1:
            raise ValueError(f'Dataset {split} must use images/{split}')
        directory = root / 'dataset' / entries[0]
        if not directory.is_dir() or not any(path.is_file() for path in directory.iterdir()):
            raise ValueError(f'Dataset {split} is empty')
        resolved[split] = str(directory)
    destination = root / 'outputs/dataset-resolved.yaml'
    destination.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding='utf-8')
    return destination


def verify_onnx(path, width, height):
    import onnx
    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    initializers = {value.name for value in model.graph.initializer}
    inputs = [value for value in model.graph.input if value.name not in initializers]
    if len(inputs) != 1:
        raise ValueError('ONNX must have exactly one image input')
    shape = [dimension.dim_value if not dimension.dim_param else None for dimension in inputs[0].type.tensor_type.shape.dim]
    if shape != [1, 3, height, width]:
        raise ValueError(f'ONNX input shape {shape} does not match [1, 3, {height}, {width}]')
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError('ONNX must be a self-contained model')


def package_artifacts(root):
    outputs = root / 'outputs'
    sources = {'best.pt': outputs / 'run/weights/best.pt', 'last.pt': outputs / 'run/weights/last.pt', 'best.onnx': outputs / 'best.onnx', 'results.csv': outputs / 'run/results.csv', 'settings.json': outputs / 'settings.json', 'versions.json': outputs / 'versions.json', 'run.json': outputs / 'run.json', 'status.json': outputs / 'status.json'}
    for name in ('worker.log', 'bootstrap.log', 'run/args.yaml', 'diagnostics.zip'):
        if (outputs / name).is_file():
            sources[name] = outputs / name
    missing = [name for name, path in sources.items() if not path.is_file() or path.is_symlink()]
    if missing:
        raise ValueError(f'Training artifacts are missing: {", ".join(missing)}')
    temporary = outputs / 'artifacts.zip.tmp'
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for name, path in sources.items():
            if name == 'status.json':
                archive.writestr(name, json.dumps({**read_json(path), 'state': 'completed', 'artifacts': 'artifacts.zip'}, indent=2) + '\n')
            else:
                archive.write(path, name)
    temporary.replace(outputs / 'artifacts.zip')


def execute(root, settings, fingerprint, versions, model_factory, resume=False, export_only=False):
    mode = validate_recovery(root, fingerprint, versions, resume, export_only)
    outputs = root / 'outputs'
    outputs.mkdir(exist_ok=True)
    training_complete = mode == 'export'
    run_record = {'fingerprint': fingerprint, 'versions': versions, 'training_started': mode != 'new'}
    write_json(outputs / 'run.json', run_record)
    write_json(outputs / 'settings.json', settings)
    write_json(outputs / 'versions.json', versions)
    try:
        if not training_complete:
            data_path = prepare_dataset(root)
            run_record['training_started'] = True
            write_json(outputs / 'run.json', run_record)
            status(root, 'training')
            model = model_factory(str(outputs / 'run/weights/last.pt' if mode == 'resume' else root / 'checkpoint.pt'), task='detect')
            attach_telemetry(root, model)
            if mode == 'resume':
                checkpoint = getattr(model, 'ckpt', None)
                if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('epoch'), int) or checkpoint['epoch'] < 0 or checkpoint.get('optimizer') is None:
                    raise ValueError('last.pt is not a resumable checkpoint; refusing to start new training')
            model.train(data=str(data_path), epochs=settings['epochs'], batch=settings['batch'], imgsz=max(settings['model_width'], settings['model_height']), device='0', workers=settings['workers'], project=str(outputs), name=settings['name'], save_dir=str(outputs / 'run'), exist_ok=True, resume=mode == 'resume', pretrained=True, save=True, val=True, plots=False, cache=False, amp=settings['amp'], mosaic=settings['mosaic'], mixup=settings['mixup'], degrees=settings['degrees'], fliplr=settings['fliplr'], patience=settings['patience'])
            if not all((outputs / name).is_file() for name in ('run/weights/best.pt', 'run/weights/last.pt', 'run/results.csv')):
                raise ValueError('Training did not produce best.pt, last.pt, and results.csv')
            training_complete = True
        status(root, 'exporting', training_complete=True)
        best = outputs / 'run/weights/best.pt'
        exported = Path(model_factory(str(best), task='detect').export(format='onnx', imgsz=[settings['model_height'], settings['model_width']], batch=1, device='cpu', dynamic=False, simplify=False, opset=17, nms=False, half=False))
        if exported.resolve().parent != best.parent.resolve() or exported.suffix != '.onnx' or exported.is_symlink():
            raise ValueError('ONNX exporter returned an unexpected path')
        verify_onnx(exported, settings['model_width'], settings['model_height'])
        shutil.copy2(exported, outputs / 'best.onnx')
        status(root, 'packaging', training_complete=True)
        save_diagnostics(root)
        package_artifacts(root)
        status(root, 'completed', training_complete=True, artifacts='artifacts.zip')
        print(f'Completed: {outputs / "artifacts.zip"}', flush=True)
    except BaseException as exc:
        status(root, 'failed', training_complete=training_complete, error=str(exc))
        record_failure(root, exc, 'export' if training_complete else 'training')
        raise


def acquire_lock(root, inherited=None):
    if sys.platform != 'linux':
        raise RuntimeError('Run this bundle on Linux with an NVIDIA CUDA PyTorch template')
    import fcntl
    outputs = root / 'outputs'
    outputs.mkdir(exist_ok=True)
    if outputs.is_symlink():
        raise ValueError('outputs must be a local directory')
    lock = outputs / 'runner.lock'
    if lock.is_symlink():
        raise ValueError('Runner lock cannot be a symbolic link')
    descriptor = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600) if inherited is None else inherited
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise RuntimeError('A bundle worker is already running; see outputs/worker.log') from exc
    os.set_inheritable(descriptor, True)
    return descriptor


def bootstrap(root, settings, descriptor, resume, export_only):
    import torch
    check_cuda(torch)
    before = str(torch.__version__)
    constraints = [f'torch=={importlib.metadata.version("torch")}']
    try:
        constraints.append('torchvision==' + importlib.metadata.version('torchvision'))
    except importlib.metadata.PackageNotFoundError:
        pass
    (root / 'outputs/torch-constraints.txt').write_text('\n'.join(constraints) + '\n', encoding='utf-8')
    previous = read_json(root / 'outputs/status.json') if (root / 'outputs/status.json').exists() else {}
    status(root, 'bootstrapping', training_complete=previous.get('training_complete', False))
    subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', str(root / '.venv')], check=True)
    python = root / '.venv/bin/python'
    subprocess.run([str(python), '-m', 'pip', 'install', '--constraint', str(root / 'outputs/torch-constraints.txt'), '-r', str(root / 'requirements.txt')], check=True)
    arguments = [str(python), '-u', str(root / 'train.py'), '--worker', '--lock-fd', str(descriptor), '--expected-torch', before]
    if resume:
        arguments.append('--resume')
    if export_only:
        arguments.append('--export-only')
    os.execv(str(python), arguments)


def launch(root, settings, descriptor, resume, export_only):
    previous = read_json(root / 'outputs/status.json') if (root / 'outputs/status.json').exists() else {}
    if previous.get('state') == 'completed':
        raise ValueError('This run is completed; extract a fresh bundle for another run')
    if (root / 'outputs/run.json').exists() and not resume and not export_only:
        raise ValueError('Existing run requires explicit --resume')
    import torch
    check_cuda(torch)
    arguments = [sys.executable, '-u', str(root / 'train.py'), '--bootstrap', '--lock-fd', str(descriptor)]
    if resume:
        arguments.append('--resume')
    if export_only:
        arguments.append('--export-only')
    with (root / 'outputs/worker.log').open('a', encoding='utf-8') as stream:
        child = subprocess.Popen(arguments, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(descriptor,), cwd=root)
    write_json(root / 'outputs/worker.json', {'pid': child.pid, 'started_at': time.time()})
    print(f'Worker PID: {child.pid}\nLog: {root / "outputs/worker.log"}\nStatus: {root / "outputs/status.json"}\nMonitor: tail -f "{root / "outputs/worker.log"}"', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--launch', action='store_true')
    modes.add_argument('--bootstrap', action='store_true')
    modes.add_argument('--worker', action='store_true')
    modes.add_argument('--check', action='store_true')
    parser.add_argument('--diagnostics', action='store_true')
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument('--resume', action='store_true')
    recovery.add_argument('--export-only', action='store_true')
    parser.add_argument('--lock-fd', type=int)
    parser.add_argument('--expected-torch')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    descriptor = None
    try:
        settings, fingerprint = validate_bundle(root)
        environment(root, settings)
        if args.check:
            print('Bundle integrity and settings passed; CUDA has not been checked.')
            return 0
        descriptor = acquire_lock(root, args.lock_fd)
        if args.diagnostics:
            snapshot_environment(root)
            print(f'Diagnostics: {package_diagnostics(root)}', flush=True)
            return 0
        snapshot_environment(root)
        telemetry(root, 'runner_started', mode='bootstrap' if args.bootstrap else 'worker' if args.worker else 'launch', resume=args.resume, export_only=args.export_only)
        if args.launch or not (args.bootstrap or args.worker):
            launch(root, settings, descriptor, args.resume, args.export_only)
        elif args.bootstrap:
            bootstrap(root, settings, descriptor, args.resume, args.export_only)
        else:
            import torch
            check_cuda(torch)
            if args.expected_torch and str(torch.__version__) != args.expected_torch:
                raise RuntimeError('Dependency installation changed template PyTorch; refusing to train')
            torch.set_num_threads(settings['threads'])
            from ultralytics import YOLO, settings as yolo_settings
            from ultralytics.data import dataset, utils
            from ultralytics.engine import trainer
            yolo_settings.update({key: False for key in ('sync', 'clearml', 'comet', 'dvc', 'mlflow', 'raytune', 'tensorboard', 'wandb')})
            utils.check_font = lambda *a, **kw: None
            dataset.save_dataset_cache_file = lambda prefix, path, values, version: values.update(version=version)
            trainer.check_amp = lambda model: check_local_amp(model, torch)
            execute(root, settings, fingerprint, runtime_versions(torch), YOLO, args.resume, args.export_only)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        if descriptor is not None:
            previous = read_json(root / 'outputs/status.json') if (root / 'outputs/status.json').exists() else {}
            if previous.get('state') != 'completed':
                status(root, 'failed', training_complete=previous.get('training_complete', False), error=str(exc))
            record_failure(root, exc, 'bootstrap' if args.bootstrap else 'worker' if args.worker else 'preflight')
        print(f'Cloud runner failed: {exc}', file=sys.stderr, flush=True)
        print(redact(root, ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))), file=sys.stderr, flush=True)
        return 2
    finally:
        if descriptor is not None:
            os.close(descriptor)


if __name__ == '__main__':
    raise SystemExit(main())
