from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import threading
import time
import zipfile
from model_registry import atomic_json, file_lock, process_alive, read_json, replace_file, utc_now
from training_board import RESERVED, validate_options
from train_model import audit_dataset, label_path

SOURCE = Path(__file__).resolve().parent
ACTIVE = {'preparing', 'exporting'}
JOB_ID = re.compile(r'job-[a-f0-9]{16}\Z')
REQUIREMENTS = '\n'.join(('ultralytics==8.4.182', 'onnx==1.23.2', 'onnxruntime==1.24.4', 'PyYAML==6.0.3',
                           'Pillow==12.3.0', 'numpy==2.5.4', 'opencv-python==5.0.0.93')) + '\n'
SETTINGS = ('epochs', 'batch', 'imgsz', 'model_width', 'model_height', 'workers', 'threads', 'device', 'name',
            'mosaic', 'mixup', 'degrees', 'fliplr', 'patience', 'amp')
PUBLIC = ('id', 'state', 'message', 'progress', 'completed_bytes', 'total_bytes', 'archive_name', 'source_recording_ids', 'checkpoint_id')


def _validate_checkpoint(path):
    from ultralytics import YOLO
    model = YOLO(str(path), task='detect')
    if getattr(model, 'task', None) != 'detect':
        raise ValueError('The checkpoint must be an Ultralytics detection model')


def _signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


def _json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')


class BundleManager:

    def __init__(self, project, registry):
        self.project = Path(project).resolve()
        self.registry = registry
        self.root = self.project / 'bundle-jobs'
        self._thread = None

    def _managed(self, path, root=None):
        root = root or self.project
        path = Path(path)
        if not path.is_absolute():
            path = root / path
        if not path.is_relative_to(root) or path.resolve() != path:
            raise ValueError('Bundle paths must stay inside their managed directory')
        current = path
        while current != root:
            if current.is_symlink():
                raise ValueError('Bundle paths cannot use symbolic links')
            current = current.parent
        if root.is_symlink():
            raise ValueError('Bundle directories cannot use symbolic links')
        return path

    def _prepare(self):
        self._managed(self.root)
        self.root.mkdir(exist_ok=True)

    def _folder(self, job_id):
        if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
            raise ValueError('Choose a known training bundle job')
        self._managed(self.root)
        return self._managed(self.root / job_id, self.root)

    def _latest(self):
        self._managed(self.root)
        pointer = read_json(self.root / 'latest.json', {})
        job_id = pointer.get('id') if isinstance(pointer, dict) else None
        if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
            return None
        folder = self._folder(job_id)
        job = read_json(folder / 'job.json')
        if not isinstance(job, dict) or job.get('id') != job_id:
            return None
        if job.get('state') in ACTIVE:
            pid = job.get('worker_pid') or job.get('preparation_pid')
            if type(pid) is not int or not process_alive(pid):
                with file_lock(folder / 'status.lock'):
                    job = read_json(folder / 'job.json', job)
                    pid = job.get('worker_pid') or job.get('preparation_pid')
                    if job.get('state') in ACTIVE and (type(pid) is not int or not process_alive(pid)):
                        job.update(state='interrupted', message='Training ZIP preparation stopped before completion', updated_at=utc_now())
                        atomic_json(folder / 'job.json', job)
                temporary = self._managed(folder / 'archive.tmp', folder)
                temporary.unlink(missing_ok=True)
        return job

    def snapshot(self):
        job = self._latest()
        return {key: job.get(key) for key in PUBLIC} if job else None

    def _update(self, job_id, **changes):
        folder = self._folder(job_id)
        with file_lock(folder / 'status.lock'):
            job = read_json(folder / 'job.json')
            if not isinstance(job, dict) or job.get('state') not in ACTIVE:
                raise ValueError('The training ZIP job is no longer active')
            job.update(changes, updated_at=utc_now())
            atomic_json(folder / 'job.json', job)
        return job

    def _space(self, required):
        if shutil.disk_usage(self.project).free < required + 64 * 1024 * 1024:
            raise ValueError('Not enough free disk space to prepare the training ZIP')

    def start(self, payload, export_callback, source_recording_ids=()):
        if not isinstance(payload, dict):
            raise ValueError('Training options must be a JSON object')
        if not callable(export_callback):
            raise ValueError('A reviewed dataset export is required')
        remote = {key: payload.get(key, default) for key, default in (('cloud_batch', 0), ('cloud_workers', 4), ('cloud_amp', True))}
        for key, maximum in (('cloud_batch', 128), ('cloud_workers', 16)):
            if type(remote[key]) is not int or not 0 <= remote[key] <= maximum:
                raise ValueError(f'{key} must be an integer from 0 to {maximum}')
        if type(remote['cloud_amp']) is not bool:
            raise ValueError('cloud_amp must be true or false')
        local = {key: value for key, value in payload.items() if key not in remote}
        local['device'] = 'cpu'
        options = validate_options(local, self.registry)
        options.update(device='cuda:0', batch=remote['cloud_batch'] or -1, workers=remote['cloud_workers'], amp=remote['cloud_amp'])
        options['imgsz'] = max(options['model_width'], options['model_height'])
        if source_recording_ids:
            from training_dataset import validate_recording_selection
            validate_recording_selection(source_recording_ids)
        elif source_recording_ids not in ((), []):
            raise ValueError('Training source recording IDs must be a list')
        checkpoint = self._managed(Path(self.registry.checkpoint_path(options['checkpoint_id'])))
        if checkpoint.suffix.lower() != '.pt' or not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise ValueError('A nonempty local PT checkpoint is required')
        self._space(checkpoint.stat().st_size * 2)
        self._prepare()
        with file_lock(self.root / 'start.lock'):
            previous = self._latest()
            if previous and previous.get('state') in ACTIVE:
                raise ValueError('A training ZIP job is already running; wait for it to finish')
            for _ in range(100):
                job_id = 'job-' + secrets.token_hex(8)
                folder = self._folder(job_id)
                try:
                    folder.mkdir(exist_ok=False)
                    break
                except FileExistsError:
                    continue
            else:
                raise ValueError('Could not reserve a unique training ZIP job')
            job = dict(id=job_id, state='preparing', message='Preparing reviewed training images', progress=0,
                       completed_bytes=0, total_bytes=0, archive_name=f'training-{job_id}.zip', options=options,
                       checkpoint_id=options['checkpoint_id'], checkpoint=str(checkpoint),
                       checkpoint_signature=list(_signature(checkpoint)), source_recording_ids=list(source_recording_ids),
                       preparation_pid=os.getpid(), worker_pid=os.getpid(), created_at=utc_now(), updated_at=utc_now())
            atomic_json(folder / 'job.json', job)
            atomic_json(self.root / 'latest.json', {'id': job_id})
        self._thread = threading.Thread(target=self._build, args=(job, export_callback), name='training-bundle', daemon=True)
        try:
            self._thread.start()
        except BaseException as exc:
            self._update(job_id, state='failed', message=f'Training ZIP preparation could not start: {exc}')
            raise
        return {key: job.get(key) for key in PUBLIC}

    def _files(self, exported):
        if not isinstance(exported, dict) or not isinstance(exported.get('path'), str):
            raise ValueError('Reviewed dataset export did not return a dataset directory')
        root = self._managed(Path(exported['path']))
        if not root.is_dir():
            raise ValueError('Reviewed dataset export is missing')
        yaml = self._managed(root / 'dataset.yaml', root)
        signatures = {}
        for kind in ('images', 'labels'):
            directory = self._managed(root / kind, root)
            for path in directory.rglob('*'):
                self._managed(path, root)
                if path.is_file():
                    signatures[path] = _signature(path)
        yaml_signature = _signature(yaml)
        report = audit_dataset(yaml)
        if report.root != root or _signature(yaml) != yaml_signature:
            raise ValueError('Reviewed dataset paths changed or leave the export directory')
        files = {}
        for split in ('train', 'val'):
            if report.split_dirs[split] != [root / 'images' / split]:
                raise ValueError('Reviewed dataset must use separate images/train and images/val directories')
            for image in report.images[split]:
                for path in (image, label_path(image)):
                    self._managed(path, root)
                    relative = path.relative_to(root).as_posix()
                    parts = PurePosixPath(relative).parts
                    suffixes = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'} if parts[0] == 'images' else {'.txt'}
                    if len(parts) != 3 or path.suffix.lower() not in suffixes or any(part in ('', '.', '..') or part.endswith((' ', '.')) or part.split('.')[0].casefold() in RESERVED or re.search(r'[\\:<>"|?*\x00-\x1f]', part) for part in parts):
                        raise ValueError('Reviewed dataset contains a filename unsupported by the portable runner')
                    signature = signatures.get(path)
                    if signature is None or _signature(path) != signature:
                        raise ValueError('Reviewed dataset changed during its integrity audit; export again')
                    files['dataset/' + relative] = (path, signature)
        return files

    def _build(self, job, export_callback):
        job_id = job['id']
        folder = self._folder(job_id)
        temporary = folder / 'archive.tmp'
        final = folder / job['archive_name']
        try:
            latest_progress = 0
            next_report = 0
            def progress(event):
                nonlocal latest_progress, next_report
                if not isinstance(event, dict):
                    return
                completed, total = event.get('completed'), event.get('total')
                if type(completed) is int and type(total) is int and total > 0:
                    now = time.monotonic()
                    if now < next_report and completed != total:
                        return
                    next_report = now + .2
                    amount = max(latest_progress, min(55, max(0, 55 * completed / total)))
                    latest_progress = amount
                    self._update(job_id, progress=amount, message='Preparing reviewed training images')
            exported = export_callback(progress=progress)
            self._update(job_id, progress=56, message='Checking dataset and checkpoint')
            files = self._files(exported)
            checkpoint = self._managed(Path(job['checkpoint']))
            signature = tuple(job['checkpoint_signature'])
            if _signature(checkpoint) != signature:
                raise ValueError('The selected checkpoint changed; export again')
            _validate_checkpoint(checkpoint)
            if _signature(checkpoint) != signature:
                raise ValueError('The selected checkpoint changed during validation')
            files['checkpoint.pt'] = (checkpoint, signature)
            for source, target in (('cloud_runner.py', 'train.py'), ('cloud_run.sh', 'run.sh')):
                path = SOURCE / source
                if not path.is_file() or path.is_symlink():
                    raise ValueError('The portable training runner is missing; update the application')
                files[target] = (path, _signature(path))
            files['dataset/dataset.yaml'] = b'path: .\ntrain: images/train\nval: images/val\nnames:\n  0: Head\n  1: Body\n'
            files['settings.json'] = _json_bytes({key: job['options'][key] for key in SETTINGS})
            files['requirements.txt'] = REQUIREMENTS.encode('utf-8')
            total = sum(len(value) if isinstance(value, bytes) else value[1][0] for value in files.values())
            self._space(total + total // 50 + len(files) * 1024)
            self._update(job_id, state='exporting', progress=60, completed_bytes=0, total_bytes=total, message='Writing training ZIP')
            completed, hashes = 0, {}
            next_report = 0
            def report_bytes():
                nonlocal next_report
                now = time.monotonic()
                if now >= next_report or completed == total:
                    self._update(job_id, completed_bytes=completed, progress=60 + 39 * completed / total)
                    next_report = now + .2
            self._managed(temporary, folder)
            with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
                for name, source in sorted(files.items()):
                    digest = hashlib.sha256()
                    with archive.open(name, 'w', force_zip64=True) as outgoing:
                        if isinstance(source, bytes):
                            outgoing.write(source)
                            digest.update(source)
                            completed += len(source)
                        else:
                            path, expected = source
                            if _signature(path) != expected:
                                raise ValueError('A training input changed while writing the ZIP; export again')
                            with path.open('rb') as incoming:
                                if _signature(path) != expected:
                                    raise ValueError('A training input changed while opening the ZIP; export again')
                                for block in iter(lambda: incoming.read(1024 * 1024), b''):
                                    outgoing.write(block)
                                    digest.update(block)
                                    completed += len(block)
                                    report_bytes()
                            if _signature(path) != expected:
                                raise ValueError('A training input changed while writing the ZIP; export again')
                    hashes[name] = digest.hexdigest()
                    report_bytes()
                manifest = dict(schema_version=1, classes=['Head', 'Body'], files=hashes)
                archive.writestr('bundle.json', _json_bytes(manifest))
            self._managed(temporary, folder)
            self._managed(final, folder)
            with temporary.open('r+b') as stream:
                os.fsync(stream.fileno())
            replace_file(temporary, final)
            self._update(job_id, state='completed', progress=100, completed_bytes=total, message='Training ZIP is ready')
        except BaseException as exc:
            for path in (temporary, final):
                try:
                    self._managed(path, folder).unlink(missing_ok=True)
                except (ValueError, OSError):
                    pass
            try:
                self._update(job_id, state='failed', message=f'Training ZIP preparation failed: {exc}')
            except (ValueError, OSError):
                pass

    def archive_path(self, job_id):
        folder = self._folder(job_id)
        job = read_json(folder / 'job.json')
        if not isinstance(job, dict) or job.get('id') != job_id or job.get('state') != 'completed':
            raise ValueError('The training ZIP is not complete')
        expected = f'training-{job_id}.zip'
        if job.get('archive_name') != expected:
            raise ValueError('The training ZIP archive name is invalid')
        path = self._managed(folder / expected, folder)
        if not path.is_file():
            raise ValueError('The completed training ZIP is missing')
        return path

    def reveal(self, job_id):
        archive = self.archive_path(job_id)
        if os.name != 'nt':
            raise ValueError('Show in folder is available on Windows')
        os.startfile(str(archive.parent))
        return {'id': job_id, 'revealed': True}
