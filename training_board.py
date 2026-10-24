from __future__ import annotations
import argparse
import contextlib
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from model_registry import ModelRegistry, atomic_json, file_lock, process_alive, read_json, utc_now
from training_metrics import TrainingMetrics, valid_epoch
PROJECT = Path(__file__).resolve().parent
ACTIVE_STATES = {'queued', 'running', 'exporting'}
DEFAULTS = dict(epochs=30, batch=4, imgsz=320, model_width=320, model_height=320,
                workers=0, threads=2, device='cpu', name='train-head-body', auto_activate=True,
                mosaic=0.3, mixup=0, degrees=0, fliplr=0.5, patience=10, background_ratio=0.25)
RESERVED = {'con', 'prn', 'aux', 'nul', 'clock$', 'conin$', 'conout$'}
RESERVED.update((f'{prefix}{number}' for prefix in ('com', 'lpt') for number in '123456789¹²³'))

def _safe_name(name):
    if not isinstance(name, str) or not re.fullmatch('[a-z0-9][a-z0-9_-]{0,47}', name) or name.casefold() in RESERVED:
        raise ValueError('Run name must use 1–48 lowercase ASCII letters, digits, hyphens or underscores')

def validate_options(payload, registry):
    if not isinstance(payload, dict):
        raise ValueError('Training options must be a JSON object')
    if set(payload) - (set(DEFAULTS) | {'checkpoint_id'}):
        raise ValueError('Unknown training option')
    options = {**DEFAULTS, **payload}
    for key, lower, upper in (('epochs', 1, 1000), ('batch', 1, 128), ('imgsz', 64, 1280),
                              ('model_width', 64, 1280), ('model_height', 64, 1280), ('workers', 0, 8), ('threads', 1, 32)):
        value = options[key]
        if isinstance(value, bool) or not isinstance(value, int) or (not lower <= value <= upper):
            raise ValueError(f'{key} must be an integer from {lower} to {upper}')
    from train_model import validate_augmentation_options
    validate_augmentation_options(options)
    background_ratio = options['background_ratio']
    if type(background_ratio) not in (int, float) or not 0 <= background_ratio <= 1:
        raise ValueError('background_ratio must be a finite number from 0 to 1')
    if any(options[key] % 32 for key in ('imgsz', 'model_width', 'model_height')):
        raise ValueError('Image size, model width and model height must be multiples of 32')
    if options['device'] != 'cpu':
        from model_inference import hardware
        available = hardware().get('device')
        if not available or not available.startswith('cuda:') or options['device'] != available:
            raise ValueError('The selected training GPU is unavailable in this runtime')
    if not isinstance(options['auto_activate'], bool):
        raise ValueError('auto_activate must be true or false')
    _safe_name(options['name'])
    checkpoints = registry.checkpoints()
    checkpoint_id = options.setdefault('checkpoint_id', checkpoints[0]['id'] if checkpoints else None)
    registry.checkpoint_path(checkpoint_id)
    return options

def _ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"

def prepared_environment(project, threads=2):
    env = dict(os.environ)
    env.update(VIRTUAL_ENV=str(project / 'runtime-env'), YOLO_CONFIG_DIR=str(project / 'runtime-data'), YOLO_AUTOINSTALL='false', YOLO_OFFLINE='true', WANDB_MODE='disabled', COMET_MODE='DISABLED', OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), PYTHONUNBUFFERED='1')
    env['PATH'] = str(project / 'runtime-env/Scripts') + os.pathsep + env.get('PATH', '')
    return env

def _public_job(job, project=None, metrics=None):
    keys = ('id', 'state', 'epoch', 'epochs', 'progress', 'message', 'model_id', 'run_path')
    result = {key: job.get(key) for key in keys}
    project = Path(project or job['project'])
    metrics = metrics or TrainingMetrics(project)
    result['metrics'] = metrics.snapshot(job.get('run_path'), stopped_early=job.get('stopped_early', False), best_epoch=job.get('best_epoch'))
    result['log_tail'] = ''
    if job.get('state') not in {'completed', 'failed', 'interrupted'}:
        return result
    log = Path(job['log_path'])
    try:
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size - 12000))
            result['log_tail'] = stream.read().decode('utf-8', errors='replace')
    except OSError:
        result['log_tail'] = ''
    return result

def _queued_timeout(job):
    return job.get('state') == 'queued' and (not job.get('worker_pid')) and ((datetime.now(timezone.utc) - datetime.fromisoformat(job['created_at'])).total_seconds() > 45)

class TrainingBoard:

    def __init__(self, project, *, registry=None, launcher=None, alive=None, runtime_python=None):
        self.project = Path(project).resolve()
        self.registry = registry or ModelRegistry(self.project)
        self.root = self.project / 'training-jobs'
        self.launcher = launcher or subprocess.Popen
        self.alive = alive or process_alive
        self.python = Path(runtime_python) if runtime_python else self.project / 'runtime-env/Scripts/python.exe'
        from local_model_import import LocalModelImport
        from model_inference import HardwareMonitor
        self.model_imports = LocalModelImport(self.project, self.registry, self.python)
        self.hardware = HardwareMonitor()
        self.metrics = TrainingMetrics(self.project)

    def _prepare(self):
        if self.root.is_symlink() or self.root.resolve().parent != self.project:
            raise ValueError('training-jobs must stay inside the project')
        self.root.mkdir(exist_ok=True)

    def _latest(self):
        pointer = read_json(self.root / 'latest.json', {})
        job_id = pointer.get('id', '') if isinstance(pointer, dict) else ''
        if not re.fullmatch('job-[a-f0-9]{16}', job_id):
            return None
        path = self.root / job_id / 'job.json'
        job = read_json(path)
        if not isinstance(job, dict):
            return None
        if job.get('state') in ACTIVE_STATES:
            pid = job.get('worker_pid') or job.get('launch_pid')
            queued_timeout = _queued_timeout(job)
            if queued_timeout or (pid and (not self.alive(pid))):
                with file_lock(path.parent / 'status.lock'):
                    job = read_json(path, job)
                    pid = job.get('worker_pid') or job.get('launch_pid')
                    queued_timeout = _queued_timeout(job)
                    if job.get('state') in ACTIVE_STATES and (queued_timeout or (pid and (not self.alive(pid)))):
                        job.update(state='failed' if queued_timeout else 'interrupted', message='The PowerShell console did not start its training worker' if queued_timeout else 'Training console or worker closed before completion', updated_at=utc_now())
                        atomic_json(path, job)
        return job

    def snapshot(self):
        job = self._latest()
        models = self.registry.list_models()
        from model_inference import choose_inference
        capabilities = self.hardware.snapshot()
        devices = [{'id': 'cpu', 'label': 'CPU · local fine-tuning'}]
        if capabilities and str(capabilities.get('device', '')).startswith('cuda:'):
            devices.append({'id': capabilities['device'], 'label': capabilities['label']})
        for model in models:
            model['inference'] = choose_inference(model['path'], capabilities) if capabilities is not None else {'backend': 'checking', 'label': 'Checking hardware…', 'device': None}
        return dict(options=dict(defaults=dict(DEFAULTS), devices=devices, checkpoints=self.registry.checkpoints()), models=models, active_model_id=next((entry['id'] for entry in models if entry['is_active']), None), training=_public_job(job, self.project, self.metrics) if job else None, model_import=self.model_imports.snapshot())

    def _console(self, script, mode, manifest, *, threads=2):
        if not self.python.is_file():
            raise ValueError('The prepared runtime is missing; run setup.bat first')
        env = prepared_environment(self.project, threads)
        lines = ["$ErrorActionPreference = 'Stop'", 'Set-Location -LiteralPath ' + _ps_quote(self.project)]
        for key in ('VIRTUAL_ENV', 'YOLO_CONFIG_DIR', 'YOLO_AUTOINSTALL', 'YOLO_OFFLINE', 'WANDB_MODE', 'COMET_MODE', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'PYTHONUNBUFFERED', 'PATH'):
            lines.append(f'$env:{key} = {_ps_quote(env[key])}')
        arguments = ['-u', str(PROJECT / 'training_board.py'), mode, str(manifest)]
        lines += ["Write-Host 'Prepared local runtime: " + str(self.python).replace("'", "''") + "'", '& ' + _ps_quote(self.python) + ' ' + ' '.join(map(_ps_quote, arguments)), '$jobExit = $LASTEXITCODE', "Write-Host ('Process exit code: ' + $jobExit)"]
        script.write_text('\n'.join(lines) + '\n', encoding='utf-8-sig')
        shell = str(Path(os.environ.get('SystemRoot', 'C:\\Windows')) / 'System32/WindowsPowerShell/v1.0/powershell.exe')
        command = [shell, '-NoLogo', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-NoExit', '-File', str(script)]
        flags = getattr(subprocess, 'CREATE_NEW_CONSOLE', 0) | getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
        return self.launcher(command, cwd=str(self.project), env=env, creationflags=flags)

    def start_training(self, payload, export_callback):
        options = validate_options(payload, self.registry)
        if not self.python.is_file():
            raise ValueError('The prepared runtime is missing; run setup.bat first')
        self._prepare()
        with file_lock(self.root / 'start.lock'):
            previous = self._latest()
            if previous and previous['state'] in ACTIVE_STATES:
                raise ValueError('A training job is already running; wait for it to finish')
            exported = export_callback()
            if not isinstance(exported, dict) or not isinstance(exported.get('path'), str) or any((type(exported.get(split)) is not int or exported[split] <= 0 for split in ('train', 'val'))):
                raise ValueError('Reviewed export must contain approved training and validation images')
            data = Path(exported['path']).resolve() / 'dataset.yaml'
            from train_model import audit_dataset
            report = audit_dataset(data)
            if any((len(report.images[split]) != exported[split] for split in ('train', 'val'))):
                raise ValueError('Reviewed export image counts do not match its dataset')
            for _ in range(100):
                job_id = 'job-' + secrets.token_hex(8)
                folder = self.root / job_id
                try:
                    folder.mkdir(exist_ok=False)
                    break
                except FileExistsError:
                    continue
            else:
                raise ValueError('Could not reserve a unique training job')
            job = dict(id=job_id, project=str(self.project), options=options, data=str(data), dataset_export=exported, checkpoint=str(self.registry.checkpoint_path(options['checkpoint_id'])), python=str(self.python), state='queued', epoch=0, epochs=options['epochs'], progress=0, message='Starting the prepared PowerShell training console', model_id=None, run_path=None, log_path=str(folder / 'training.log'), created_at=utc_now(), updated_at=utc_now())
            manifest = folder / 'job.json'
            atomic_json(manifest, job)
            atomic_json(self.root / 'latest.json', {'id': job_id})
            try:
                process = self._console(folder / 'start.ps1', '--worker', manifest, threads=options['threads'])
                with file_lock(folder / 'status.lock'):
                    job = read_json(manifest, job)
                    job['launch_pid'] = process.pid
                    atomic_json(manifest, job)
            except Exception as exc:
                job.update(state='failed', message=f'Could not launch training console: {exc}', failure_stage='launch')
                atomic_json(manifest, job)
                raise ValueError(job['message']) from exc
        return _public_job(job, self.project, self.metrics)

    def select_model(self, model_id):
        return self.registry.select(model_id)

class _Tee:

    def __init__(self, console, log):
        self.console, self.log = (console, log)
        self.lock = threading.Lock()

    def write(self, value):
        with self.lock:
            self.console.write(value)
            self.log.write(value)
            self.log.flush()
        return len(value)

    def flush(self):
        self.console.flush()
        self.log.flush()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return 'utf-8'

def _train(job, update):
    from train_model import main
    options = {**DEFAULTS, **job['options']}
    arguments = ['--data', job['data'], '--weights', job['checkpoint'], '--device', options['device'], '--name', job['id']]
    for key in ('epochs', 'batch', 'imgsz', 'workers', 'threads', 'mosaic', 'mixup', 'degrees', 'fliplr', 'patience'):
        value = max(options[key], options.get('model_width', 320), options.get('model_height', 320)) if key == 'imgsz' else options[key]
        arguments += ['--' + key, str(value)]
    result = {}

    def progress(event):
        if 'run_path' in event:
            result['run_path'] = event['run_path']
        if 'checkpoint' in event:
            result['checkpoint'] = event['checkpoint']
        fields = {key: event[key] for key in ('epoch', 'run_path', 'message', 'stopped_early', 'best_epoch') if key in event}
        if 'epoch' in fields:
            fields['progress'] = min(90, 90 * fields['epoch'] / options['epochs'])
        update(**fields)
    if main(arguments, progress_callback=progress) != 0:
        raise ValueError('Trainer failed; see the training log')
    return Path(result['checkpoint'])

def _export(checkpoint, destination, *, width=320, height=320):
    from ultralytics import YOLO
    exported = Path(YOLO(str(checkpoint), task='detect').export(format='onnx', imgsz=[height, width], batch=1, device='cpu', dynamic=False, simplify=True, opset=17, quantize=None, nms=False)).resolve()
    if exported != destination.resolve():
        import shutil
        shutil.copyfile(exported, destination)
    return destination

def _variants(source):
    from build_focus_models import PRESETS, build_model
    paths = []
    for view, (width, height, threshold) in PRESETS.items():
        target = source.with_name(f'trained-{view}.onnx')
        paths.append((view, build_model(source, target, width, height, threshold)))
    return paths

def run_training_job(manifest, *, registry=None, trainer=None, exporter=None, variant_builder=None):
    manifest = Path(manifest).resolve()
    job = read_json(manifest)
    if not isinstance(job, dict):
        raise ValueError('Training job metadata is missing')
    project = Path(job['project']).resolve()
    if manifest != project / 'training-jobs' / job['id'] / 'job.json':
        raise ValueError("Training job must be inside its project's managed job folder")
    registry = registry or ModelRegistry(project)
    lock = manifest.parent / 'status.lock'

    def update(**fields):
        nonlocal job
        with file_lock(lock):
            job = read_json(manifest, job)
            job.update(fields, updated_at=utc_now())
            atomic_json(manifest, job)
    try:
        with file_lock(manifest.parent / 'worker.lock', timeout=0.1):
            if read_json(manifest, {}).get('state') not in {'queued', 'running'}:
                raise ValueError('This training job has already finished')
            log_path = Path(job['log_path'])
            if log_path != manifest.parent / 'training.log':
                raise ValueError('Training log must stay inside its job folder')
            with log_path.open('a', encoding='utf-8', buffering=1) as log:
                with contextlib.redirect_stdout(_Tee(sys.stdout, log)), contextlib.redirect_stderr(_Tee(sys.stderr, log)):
                    stage = 'training'
                    try:
                        options = validate_options(job['options'], registry)
                        update(options=options)
                        os.environ.update({key: value for key, value in prepared_environment(project, job['options']['threads']).items() if key != 'PATH'})
                        update(state='running', worker_pid=os.getpid(), message='Fine-tuning the reviewed dataset')
                        print(f"Job {job['id']}: CPU training, {job['epochs']} epochs; log: {log_path}", flush=True)
                        checkpoint = Path((trainer or _train)(job, update)).resolve()
                        if not checkpoint.is_file() or checkpoint.suffix != '.pt':
                            raise ValueError('Training did not produce a best checkpoint')
                        stage = 'export'
                        width, height = options['model_width'], options['model_height']
                        run_path = str(checkpoint.parent.parent)
                        measured = TrainingMetrics(project).snapshot(run_path)
                        final_epoch = (measured['latest']['epoch'] if measured['latest'] else None) or valid_epoch(job.get('epoch')) or job['epochs']
                        stopped_early = bool(job.get('stopped_early')) or final_epoch < job['epochs']
                        update(state='exporting', epoch=final_epoch, stopped_early=stopped_early,
                               best_epoch=valid_epoch(job.get('best_epoch')) or measured['best_epoch'],
                               progress=92, run_path=run_path, message=f'PT checkpoint saved; exporting {width} × {height} ONNX')
                        if exporter is None:
                            exported = Path(_export(checkpoint, checkpoint.with_suffix('.onnx'), width=width, height=height)).resolve()
                        else:
                            exported = Path(exporter(checkpoint, checkpoint.with_suffix('.onnx'))).resolve()
                        if list(registry.validator(exported)['input_shape']) != [1, 3, height, width]:
                            raise ValueError('Exported model input size does not match the requested resolution')
                        candidates = [('focus', exported)]
                        if (width, height) == (320, 320):
                            try:
                                candidates += list((variant_builder or _variants)(exported))
                            except Exception as exc:
                                print(f'Smaller view variants unavailable; retaining the validated focus model: {exc}', flush=True)
                        stage = 'registration'
                        update(progress=96, message='Validating and storing trained models')
                        entries = []
                        for view, path in candidates:
                            entry = registry.register(path, label=f"{job['options']['name']} · {view}", view=view, checkpoint=checkpoint)
                            entries.append(entry)
                        selected = entries[0]
                        if job['options']['auto_activate']:
                            registry.select(selected['id'])
                        update(state='completed', progress=100, model_id=selected['id'],
                               message=f'PT and ONNX saved · {width} × {height}' + (' · this model is now the default' if options['auto_activate'] else ''))
                        print(job['message'], flush=True)
                        return 0
                    except KeyboardInterrupt:
                        update(state='interrupted', message=f'Cancelled during {stage}; the previous default was preserved', failure_stage=stage)
                        print(job['message'], flush=True)
                        return 130
                    except Exception as exc:
                        update(state='failed', message=f'{stage.capitalize()} failed: {exc}', failure_stage=stage)
                        traceback.print_exc()
                        return 1
    except ValueError:
        raise

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--worker', type=Path)
    args = parser.parse_args(argv)
    return run_training_job(args.worker)
if __name__ == '__main__':
    raise SystemExit(main())
