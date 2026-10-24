from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import stat
import subprocess
import sys
import threading
import time


ACTIVE_STATES = {'queued', 'running', 'exporting'}


def _dimensions(width, height):
    if any(type(value) is not int or not 64 <= value <= 1280 or value % 32 for value in (width, height)):
        raise ValueError('Model width and height must be multiples of 32 from 64 to 1280')
    return width, height


def _signature(path):
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _source(path):
    path = Path(path)
    if not path.is_absolute() or str(path).startswith(('\\\\', '//')) or '://' in str(path) or path.is_symlink():
        raise ValueError('Select an ordinary local ONNX or PT model file')
    try:
        resolved = path.resolve(strict=True)
        signature = _signature(resolved)
    except OSError:
        raise ValueError('The selected model is missing; browse for it again') from None
    if not resolved.is_file() or resolved.suffix.lower() not in ('.onnx', '.pt') or not 0 < signature[2] <= 2 * 1024 ** 3:
        raise ValueError('Select an ONNX or PT model file no larger than 2 GB')
    return resolved, signature


def _digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def _environment(python, folder):
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    env.update(VIRTUAL_ENV=str(python.parent.parent), YOLO_CONFIG_DIR=str(folder / 'yolo-config'),
               YOLO_AUTOINSTALL='false', YOLO_OFFLINE='true', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               WANDB_MODE='disabled', COMET_MODE='DISABLED', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='',
               PYTHONUTF8='1', PYTHONUNBUFFERED='1')
    env['PATH'] = str(python.parent) + os.pathsep + env.get('PATH', '')
    return env


def _validate_runtime(path):
    from model_inference import validate_runtime
    return validate_runtime(path)


class LocalModelImport:
    def __init__(self, project, registry, runtime_python, *, picker=None, runner=None, clock=None, runtime_probe=None):
        self.project = Path(project).resolve()
        self.registry = registry
        self.python = Path(runtime_python)
        self._runner = runner or subprocess.run
        self._picker = picker or self._pick
        self._clock = clock or time.monotonic
        self._runtime_probe = runtime_probe or _validate_runtime
        self._lock = threading.RLock()
        self._browsing = False
        self._selections = {}
        self._job = None

    def _available(self):
        if self._browsing or self._job and self._job['state'] in ACTIVE_STATES:
            raise ValueError('Another model import is busy; wait for it to finish')

    def _pick(self):
        if not self.python.is_file():
            raise ValueError('The prepared runtime is missing; run setup.bat first')
        env = dict(os.environ, PYTHONUTF8='1')
        env.pop('QT_QPA_PLATFORM', None)
        try:
            result = self._runner([str(self.python), str(Path(__file__).resolve()), '--pick'],
                                  stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding='utf-8',
                                  errors='replace', env=env, timeout=600, shell=False,
                                  creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except subprocess.TimeoutExpired:
            raise ValueError('The model picker timed out; open it again') from None
        except OSError:
            raise ValueError('Could not open the native model picker') from None
        try:
            payload = json.loads(result.stdout)
        except (ValueError, TypeError):
            raise ValueError('The model picker did not return a valid result') from None
        if result.returncode or not isinstance(payload, dict) or 'error' in payload:
            raise ValueError('Could not open the native model picker')
        return payload

    def browse(self):
        with self._lock:
            self._available()
            self._browsing = True
        try:
            result = self._picker()
            if result == {'cancelled': True}:
                return {'cancelled': True}
            if not isinstance(result, dict) or set(result) != {'path'} or not isinstance(result['path'], str):
                raise ValueError('The model picker must return one local model file')
            path, signature = _source(result['path'])
            with self._lock:
                now = self._clock()
                self._selections = {key: value for key, value in self._selections.items() if value['expires'] > now}
                if len(self._selections) >= 32:
                    del self._selections[next(iter(self._selections))]
                selection_id = 'model-selection-' + secrets.token_urlsafe(24)
                self._selections[selection_id] = dict(path=path, signature=signature, expires=now + 600)
            return dict(selection_id=selection_id, name=path.name, path=str(path), format=path.suffix.lower()[1:])
        finally:
            with self._lock:
                self._browsing = False

    def _options(self, payload, required):
        if not isinstance(payload, dict) or required not in payload or set(payload) - {required, 'auto_activate', 'model_width', 'model_height'}:
            raise ValueError(f'Expected {required} and optional model import settings')
        if not isinstance(payload[required], str) or not payload[required] or len(payload[required]) > 128:
            raise ValueError(f'A valid {required} is required')
        activate = payload.get('auto_activate', True)
        if type(activate) is not bool:
            raise ValueError('auto_activate must be true or false')
        width, height = _dimensions(payload.get('model_width', 640), payload.get('model_height', 448))
        return activate, width, height

    def start(self, payload):
        activate, width, height = self._options(payload, 'selection_id')
        with self._lock:
            self._available()
            selected = self._selections.get(payload['selection_id'])
            if not selected or selected['expires'] <= self._clock():
                raise ValueError('The model selection expired; browse for the file again')
            path, signature = _source(selected['path'])
            if signature != selected['signature']:
                raise ValueError('The selected model changed; browse for it again')
            job = self._begin(path, signature, activate, width, height)
            del self._selections[payload['selection_id']]
            return job

    def convert(self, payload):
        activate, width, height = self._options(payload, 'model_id')
        with self._lock:
            self._available()
            checkpoint = self.registry.artifact_path(payload['model_id'], 'pt')
            path, signature = _source(checkpoint)
            return self._begin(path, signature, activate, width, height)

    def _begin(self, path, signature, activate, width, height):
        if path.suffix.lower() == '.pt' and not self.python.is_file():
            raise ValueError('The prepared runtime is missing; run setup.bat first')
        self._job = dict(id='import-' + secrets.token_hex(8), state='queued', message='Preparing model import',
                         model_id=None, auto_activate=activate)
        try:
            threading.Thread(target=self._work, args=(dict(self._job), path, signature, width, height),
                             name='local-model-import', daemon=True).start()
        except RuntimeError:
            self._job.update(state='failed', message='Could not start the model import worker')
        return dict(self._job)

    def snapshot(self):
        with self._lock:
            return dict(self._job) if self._job else None

    def _update(self, job_id, **fields):
        with self._lock:
            if self._job and self._job['id'] == job_id:
                self._job.update(fields)

    def _safe_directory(self, path):
        if path.is_symlink() or not path.resolve().is_relative_to(self.project):
            raise ValueError('Model import folders must stay inside the application')
        if path.exists() and getattr(path.lstat(), 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 1024):
            raise ValueError('Model import folders must be ordinary local directories')
        path.mkdir(exist_ok=True)
        if not path.is_dir() or path.resolve() != path.absolute():
            raise ValueError('Model import folders must be ordinary local directories')

    def _folder(self, job_id):
        parent = self.project
        for name in ('runtime-data', 'model-imports'):
            parent = parent / name
            self._safe_directory(parent)
        destination = parent / job_id
        number = 1
        while destination.exists():
            destination = parent / f'{job_id}-{number}'
            number += 1
        if destination.is_symlink() or destination.resolve().parent != parent:
            raise ValueError('Model import destination leaves its managed folder')
        destination.mkdir(exist_ok=False)
        return destination

    def _copy(self, source, signature, destination):
        if _source(source)[1] != signature:
            raise ValueError('The selected model changed before it could be copied; browse again')
        with source.open('rb') as incoming, destination.open('xb') as outgoing:
            before = incoming.seek(0, 2)
            incoming.seek(0)
            if before != signature[2]:
                raise ValueError('The selected model changed before it could be copied; browse again')
            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        if _source(source)[1] != signature or destination.stat().st_size != signature[2]:
            raise ValueError('The selected model changed while copying; browse again')

    def _export(self, checkpoint, width, height):
        expected = checkpoint.with_suffix('.onnx')
        result = self._runner([str(self.python), str(Path(__file__).resolve()), '--convert', str(checkpoint),
                               '--width', str(width), '--height', str(height)], cwd=str(checkpoint.parent),
                              env=_environment(self.python, checkpoint.parent), stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=300,
                              shell=False, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if len(result.stdout) > 2 * 1024 * 1024:
            raise ValueError('The checkpoint converter returned an oversized response')
        try:
            payload = json.loads(result.stdout)
        except (ValueError, TypeError):
            raise ValueError('Checkpoint conversion failed; the previous default was preserved') from None
        if result.returncode or not isinstance(payload, dict) or 'error' in payload:
            error = payload.get('error') if isinstance(payload, dict) else None
            raise ValueError(str(error)[:400] if isinstance(error, str) and error else 'Checkpoint conversion failed')
        if payload.get('path') != str(expected.resolve()) or expected.is_symlink() or expected.resolve().parent != checkpoint.parent or not expected.is_file():
            raise ValueError('The converter did not return the expected private ONNX file')
        if list(self.registry.validator(expected)['input_shape']) != [1, 3, height, width]:
            raise ValueError('The exported ONNX dimensions do not match the requested dimensions')
        return expected

    def _work(self, job, source, signature, width, height):
        try:
            self._update(job['id'], state='running', message='Copying the selected model into the application')
            folder = self._folder(job['id'])
            copied = folder / ('source' + source.suffix.lower())
            self._copy(source, signature, copied)
            checkpoint = None
            candidate = copied
            if copied.suffix == '.pt':
                self._update(job['id'], state='exporting', message=f'Validating the Head/Body checkpoint and exporting {width} × {height} ONNX')
                digest = _digest(copied)
                candidate = self._export(copied, width, height)
                if _digest(copied) != digest:
                    raise ValueError('The checkpoint changed during conversion')
                checkpoint = copied
            candidate = candidate.rename(folder / 'model.onnx')
            if checkpoint is not None:
                checkpoint = checkpoint.rename(folder / 'checkpoint.pt')
            self._update(job['id'], state='running', message='Checking the imported model with the local inference runtime')
            self.registry.validator(candidate)
            self._runtime_probe(candidate)
            self._update(job['id'], state='running', message='Registering the validated model')
            model = self.registry.register(candidate, label=source.stem.strip()[:150] or 'Imported model', checkpoint=checkpoint)
            if job['auto_activate']:
                self.registry.select(model['id'])
            self._update(job['id'], state='completed', model_id=model['id'],
                         message='Model imported' + (' and applied as the default' if job['auto_activate'] else '; the current default was kept'))
        except subprocess.TimeoutExpired:
            self._update(job['id'], state='failed', message='Checkpoint conversion timed out; the previous default was preserved')
        except Exception as exc:
            self._update(job['id'], state='failed', message=f'Model import failed: {str(exc)[:400]}')


def choose_model():
    from PySide6.QtWidgets import QApplication, QFileDialog
    app = QApplication.instance() or QApplication([])
    app.setApplicationName('Local model picker')
    selected, _ = QFileDialog.getOpenFileName(None, 'Choose a local Head/Body model', '',
                                             'Head/Body models (*.onnx *.pt);;ONNX models (*.onnx);;Ultralytics checkpoints (*.pt)',
                                             options=QFileDialog.Option(0))
    return {'path': selected} if selected else {'cancelled': True}


def convert_checkpoint(checkpoint, width, height, *, model_factory=None):
    width, height = _dimensions(width, height)
    checkpoint, _ = _source(checkpoint)
    if checkpoint.suffix.lower() != '.pt':
        raise ValueError('A local PT checkpoint is required')
    import torch
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass
    if model_factory is None:
        from ultralytics import YOLO
        model_factory = YOLO
    model = model_factory(str(checkpoint))
    if model.task != 'detect' or model.names != {0: 'Head', 1: 'Body'}:
        raise ValueError('The checkpoint must be a detection model with exactly 0: Head and 1: Body')
    exported = Path(model.export(format='onnx', imgsz=[height, width], batch=1, device='cpu', dynamic=False,
                                 simplify=False, opset=17, nms=False)).resolve()
    expected = checkpoint.with_suffix('.onnx')
    if exported != expected or exported.is_symlink() or not exported.is_file():
        raise ValueError('The checkpoint export must stay beside its private copy')
    return {'path': str(exported)}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Import local Head/Body models')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--pick', action='store_true')
    mode.add_argument('--convert', type=Path)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=448)
    arguments = parser.parse_args(argv)
    try:
        if arguments.pick:
            result = choose_model()
        else:
            with contextlib.redirect_stdout(sys.stderr):
                result = convert_checkpoint(arguments.convert, arguments.width, arguments.height)
        print(json.dumps(result, ensure_ascii=True), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({'error': str(exc)[:500]}, ensure_ascii=True), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
