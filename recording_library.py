from __future__ import annotations
import argparse
import contextlib
import copy
import hashlib
import inspect
import json
import math
import os
import re
import secrets
import subprocess
import sys
import time
import traceback
import threading
from datetime import datetime, timezone
from pathlib import Path
from model_registry import ModelRegistry, atomic_json, file_lock, process_alive, read_json, utc_now
from local_trash import LocalTrash, checked_managed_path, _ordinary
PROJECT = Path(__file__).resolve().parent
DEFAULTS = dict(sample_fps=1, max_images=600, confidence=0.25, tiles=True)
RUN_ID = re.compile('run-[a-f0-9]{24}\\Z')
SELECTION_ID = re.compile('selection-[a-f0-9]{24}\\Z')
JOB_ID = re.compile('video-[a-f0-9]{16}\\Z')
FRAME_ID = re.compile('frame-[0-9]{6,12}\\Z')

def recording_id(run):
    canonical = os.path.normcase(str(Path(run).resolve()))
    return 'run-' + hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:24]

def _recording_entry(run):
    run = Path(run).resolve(strict=True)
    if not run.is_dir() or (run / 'manifest.json').is_symlink():
        raise ValueError('Recording must contain an ordinary manifest file')
    data = read_json(run / 'manifest.json')
    if not isinstance(data, dict) or data.get('schema_version') != 1 or data.get('status') not in ('running', 'complete', 'failed', 'interrupted') or (not isinstance(data.get('frames'), list)):
        raise ValueError('Recording has an invalid preparation manifest')
    for directory in ('images', 'draft-labels'):
        if not (run / directory).resolve().is_relative_to(run):
            raise ValueError('Recording frame directory leaves the run')
    seen = set()
    counts = {'approved': 0, 'rejected': 0}
    approved_splits = {'train': 0, 'val': 0}
    backgrounds = 0
    for frame in data['frames']:
        if not isinstance(frame, dict) or not isinstance(frame.get('id'), str) or (not FRAME_ID.fullmatch(frame['id'])) or (frame['id'] in seen) or (frame.get('status') not in ('pending', 'approved', 'rejected', 'auto_removed')) or (frame.get('split') not in ('train', 'val')) or (type(frame.get('revision')) is not int) or (frame['revision'] < 0) or (type(frame.get('empty_confirmed')) is not bool) or (frame.get('width') != 640) or (frame.get('height') != 448) or (not isinstance(frame.get('boxes'), list)):
            raise ValueError('Recording contains an invalid review frame')
        seen.add(frame['id'])
        for key, expected in (('image', f"images/{frame['id']}.png"), ('label', f"draft-labels/{frame['id']}.txt")):
            if frame.get(key) != expected:
                raise ValueError('Recording contains an unsafe frame path')
        if len(frame['boxes']) > 4096:
            raise ValueError('Recording frame has too many boxes')
        for box in frame['boxes']:
            if not isinstance(box, dict) or type(box.get('class_id')) is not int or box['class_id'] not in (0, 1):
                raise ValueError('Recording contains an invalid class')
            values = [box.get(key) for key in ('x1', 'y1', 'x2', 'y2')]
            if any((type(value) not in (int, float) or not math.isfinite(value) or (not 0 <= value <= 1) for value in values)) or values[2] <= values[0] or values[3] <= values[1]:
                raise ValueError('Recording contains invalid box coordinates')
            if 'score' in box and (type(box['score']) not in (int, float) or not math.isfinite(box['score']) or (not 0 <= box['score'] <= 1)):
                raise ValueError('Recording contains an invalid box score')
        if frame['status'] == 'approved' and (not frame['boxes']) and (not frame['empty_confirmed']):
            raise ValueError('Recording contains an unconfirmed approved empty frame')
        if frame['status'] == 'auto_removed' and (frame['boxes'] or frame['empty_confirmed']):
            raise ValueError('Automatically removed frames must remain empty and unconfirmed')
        if frame['status'] in counts:
            counts[frame['status']] += 1
        if frame['status'] == 'approved':
            approved_splits[frame['split']] += 1
            backgrounds += int(not frame['boxes'] and frame['empty_confirmed'])
    source = data.get('source', {})
    if not isinstance(source, dict):
        raise ValueError('Recording source metadata must be an object')
    source_name = Path(str(source.get('path', ''))).name
    return dict(id=recording_id(run), label=f'{source_name} · {run.name}' if source_name else run.name, path=str(run), status=data['status'], frames=len(data['frames']), approved=counts['approved'], rejected=counts['rejected'], approved_train=approved_splits['train'], approved_val=approved_splits['val'], approved_backgrounds=backgrounds, source_name=source_name)

def _validate_metadata(metadata):
    if not isinstance(metadata, dict) or (metadata.get('width'), metadata.get('height')) not in ((1280, 882), (1280, 720)):
        raise ValueError('Choose a native 1280x882 or 1280x720 recording')
    for key in ('fps', 'duration_seconds'):
        value = metadata.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('Video has no reliable frame rate/duration metadata')
    if type(metadata.get('frame_count')) is not int or metadata['frame_count'] <= 0:
        raise ValueError('Video has no reliable frame count')
    return {key: metadata[key] for key in ('width', 'height', 'fps', 'frame_count', 'duration_seconds')}

def _options(payload):
    if not isinstance(payload, dict) or set(payload) - (set(DEFAULTS) | {'selection_id', 'model_id'}):
        raise ValueError('Expected known video preparation options')
    values = {**DEFAULTS, **payload}
    for key, minimum, maximum in (('sample_fps', 0.1, 10), ('confidence', 0.05, 0.95)):
        value = values[key]
        if type(value) not in (int, float) or not math.isfinite(value) or (not minimum <= value <= maximum):
            raise ValueError(f'{key} must be a finite number from {minimum} to {maximum}')
    if type(values['max_images']) is not int or not 2 <= values['max_images'] <= 128000:
        raise ValueError('max_images must be an integer from 2 to 128000')
    if type(values['tiles']) is not bool:
        raise ValueError('tiles must be true or false')
    if not isinstance(values.get('selection_id'), str) or not SELECTION_ID.fullmatch(values['selection_id']):
        raise ValueError('Choose a video using Browse video first')
    if not isinstance(values.get('model_id'), str) or not values['model_id'] or len(values['model_id']) > 128:
        raise ValueError('Choose a known annotation model')
    return values

def _bootstrap_expired(job):
    return job.get('state') == 'queued' and (not job.get('worker_pid')) and ((datetime.now(timezone.utc) - datetime.fromisoformat(job['created_at'])).total_seconds() > 45)

def _tail(path, limit=12000):
    try:
        with Path(path).open('rb') as stream:
            stream.seek(max(0, Path(path).stat().st_size - limit))
            return stream.read().decode('utf-8', errors='replace')
    except OSError:
        return ''

def _public_job(job, folder):
    keys = ('id', 'state', 'processed', 'total', 'saved', 'progress', 'message', 'recording_id', 'run_path', 'cancel_requested', 'resume_from_job_id')
    result = {key: job.get(key) for key in keys}
    result['cancel_requested'] = bool(job.get('cancel_requested'))
    result['can_cancel'] = job.get('state') in ('queued', 'running') and not result['cancel_requested']
    result['can_resume'] = job.get('state') in ('failed', 'interrupted', 'cancelled')
    result['log_tail'] = (_tail(folder / 'preparation.log') + _tail(folder / 'bootstrap.log', 2000))[-12000:]
    return result

class RecordingLibrary:

    def __init__(self, project, initial_run, *, registry=None, picker=None, launcher=None, alive=None, runtime_python=None, inspector=None):
        self.project = Path(project).resolve()
        self.initial_run = Path(initial_run).resolve()
        if self.initial_run.exists():
            _recording_entry(self.initial_run)
        self.root = self.project / 'video-jobs'
        self.registry = registry or ModelRegistry(self.project)
        self.picker = picker
        self.inspector = inspector
        self.launcher = launcher or subprocess.Popen
        self.alive = alive or process_alive
        self.python = Path(runtime_python) if runtime_python else self.project / 'runtime-env/Scripts/python.exe'
        self._catalog_cache = {}
        self._catalog_lock = threading.RLock()

    def _prepare(self):
        _ordinary(self.root)
        self.root.mkdir(exist_ok=True)

    def _runs(self):
        runs = [self.initial_run]
        parent = self.project / 'dataset-runs'
        if parent.is_dir() and (not parent.is_symlink()) and (parent.resolve().parent == self.project):
            runs += [path for path in sorted(parent.iterdir()) if path.is_dir() and (not path.is_symlink()) and (path.resolve().parent == parent)]
        seen = set()
        for run in runs:
            if run in seen:
                continue
            seen.add(run)
            try:
                manifest = run / 'manifest.json'
                if manifest.is_symlink():
                    continue
                with self._catalog_lock:
                    stat = manifest.stat()
                    signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
                    cached = self._catalog_cache.get(run)
                    if cached and cached[0] == signature:
                        entry = copy.deepcopy(cached[1])
                    else:
                        entry = _recording_entry(run)
                        after = manifest.stat()
                        if signature == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                            self._catalog_cache[run] = (signature, copy.deepcopy(entry))
                            if len(self._catalog_cache) > 128:
                                self._catalog_cache.pop(next(iter(self._catalog_cache)))
            except (ValueError, OSError):
                continue
            yield (run, entry)

    def resolve(self, run_id):
        if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
            raise ValueError('Choose a known recording ID')
        for path, entry in self._runs():
            if entry['id'] == run_id:
                return path
        raise ValueError('Unknown or invalid recording; refresh the recording library')

    def _job(self, job_id):
        if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
            raise ValueError('Choose a known video preparation job ID')
        _ordinary(self.root)
        folder = _ordinary(self.root / job_id)
        path = _ordinary(folder / 'job.json')
        job = read_json(path)
        if not isinstance(job, dict) or job.get('id') != job_id or Path(job.get('project', '')).resolve() != self.project:
            raise ValueError('Unknown or invalid video preparation job')
        if job.get('state') in ('queued', 'running'):
            pid = job.get('worker_pid') or job.get('launch_pid')
            if _bootstrap_expired(job) or (pid and (not self.alive(pid))):
                with file_lock(folder / 'status.lock'):
                    job = read_json(path, job)
                    pid = job.get('worker_pid') or job.get('launch_pid')
                    expired = _bootstrap_expired(job)
                    if job.get('state') in ('queued', 'running') and (expired or (pid and (not self.alive(pid)))):
                        startup = not job.get('worker_pid')
                        job.update(state='failed' if startup else 'interrupted', updated_at=utc_now(), message='The preparation worker did not start' if startup else 'The preparation worker stopped before completion')
                        atomic_json(path, job)
        return (job, folder)

    def _latest(self):
        pointer = read_json(self.root / 'latest.json', {})
        job_id = pointer.get('id') if isinstance(pointer, dict) else None
        if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
            return (None, None)
        try:
            return self._job(job_id)
        except ValueError:
            return (None, None)

    def _jobs(self):
        if not self.root.is_dir():
            return []
        _ordinary(self.root)
        jobs = []
        for folder in self.root.iterdir():
            if not JOB_ID.fullmatch(folder.name):
                continue
            try:
                jobs.append(self._job(folder.name))
            except (ValueError, OSError):
                continue
        return sorted(jobs, key=lambda item: item[0].get('created_at', ''), reverse=True)

    def _resume_path(self, job):
        if job.get('state') not in ('failed', 'interrupted', 'cancelled'):
            raise ValueError('Only interrupted, failed or cancelled preparation jobs can resume')
        run = checked_managed_path(self.project, 'recording', job['run_path'], exists=False) if job.get('run_path') else None
        if run is None or not run.is_dir():
            run = self._find_resume_checkpoint(job)
            if run is None:
                if job.get('run_path'):
                    raise ValueError('The unfinished checkpoint is missing; no matching managed recording is available')
                return None
        raw = read_json(_ordinary(run / 'manifest.json'))
        if not isinstance(raw, dict) or raw.get('status') not in ('running', 'failed', 'interrupted'):
            raise ValueError('Only unfinished recordings can resume; complete recordings are preserved')
        if not isinstance(raw.get('frames'), list) or any(not isinstance(frame, dict) or frame.get('status') != 'pending' or frame.get('revision') != 0 or frame.get('empty_confirmed') for frame in raw['frames']):
            raise ValueError('Resume will not replace reviewed annotations')
        return run

    def _find_resume_checkpoint(self, job):
        parent = _ordinary(self.project / 'dataset-runs')
        if not parent.is_dir():
            return None
        matches = []
        for candidate in parent.iterdir():
            try:
                run = checked_managed_path(self.project, 'recording', candidate)
                raw = read_json(_ordinary(run / 'manifest.json'))
                if not isinstance(raw, dict) or raw.get('schema_version') != 1 or raw.get('status') not in ('running', 'failed', 'interrupted'):
                    continue
                source, settings = raw.get('source'), raw.get('settings')
                if not isinstance(source, dict) or not isinstance(settings, dict):
                    continue
                if Path(source.get('path', '')).resolve() != Path(job['video_path']).resolve() or source.get('size_bytes') != job['size_bytes'] or source.get('mtime_ns') != job['mtime_ns']:
                    continue
                options = job['options']
                expected = dict(sample_fps=options['sample_fps'], max_images=options['max_images'], view='focus', start_seconds=0,
                    end_seconds=source.get('duration_seconds'), val_fraction=0.2, gap_seconds=5,
                    model=str(Path(job['model_path']).resolve()), model_sha256=job['model_sha256'], confidence=options['confidence'],
                    head_detail_tiles=options['tiles'], decoder_threads=1)
                if settings != expected:
                    continue
                matches.append(run)
            except (ValueError, OSError, KeyError, TypeError):
                continue
        if len(matches) > 1:
            raise ValueError('Multiple unfinished checkpoints match this job; resume is ambiguous and all recordings were preserved')
        return matches[0] if matches else None

    def _public(self, job, folder):
        result = _public_job(job, folder)
        if result['can_resume']:
            try:
                run = self._resume_path(job)
                if run is not None:
                    result.update(run_path=str(run), recording_id=recording_id(run))
            except (ValueError, OSError):
                result['can_resume'] = False
        return result

    def snapshot(self):
        recordings = [entry for _, entry in self._runs()]
        initial_id = recording_id(self.initial_run)
        jobs = self._jobs()
        active = [item for item, _ in jobs if item.get('state') in ('queued', 'running')]
        resume_by_run = {}
        for candidate, _ in jobs:
            try:
                run = self._resume_path(candidate)
                if run is not None:
                    resume_by_run.setdefault(recording_id(run), candidate['id'])
            except (ValueError, OSError):
                continue
        for entry in recordings:
            reason = None
            try:
                checked_managed_path(self.project, 'recording', entry['path'])
            except (ValueError, OSError):
                reason = 'Only managed dataset-runs folders can be deleted'
            if any(item.get('recording_id') == entry['id'] or item.get('run_path') == entry['path'] for item in active):
                reason = 'This recording is being prepared; cancel and wait for preparation to stop'
            entry.update(can_delete=reason is None, delete_reason=reason, resume_job_id=None if active else resume_by_run.get(entry['id']))
        job, folder = self._latest()
        preparation = self._public(job, folder) if job else None
        if preparation and active:
            preparation['can_resume'] = False
        if preparation and preparation['state'] == 'completed' and preparation.get('run_path') and (not Path(preparation['run_path']).is_dir()):
            preparation.update(recording_id=None, run_path=None, message='The prepared recording directory is no longer available. Prepare the video again to recreate it.')
        default_id = next((entry['id'] for entry in recordings if entry['id'] == initial_id), recordings[0]['id'] if recordings else None)
        return dict(recordings=recordings, default_recording_id=default_id, preparation=preparation, options={'defaults': dict(DEFAULTS)})

    def delete(self, run_id):
        self._prepare()
        with file_lock(self.root / 'start.lock'):
            run = self.resolve(run_id)
            checked_managed_path(self.project, 'recording', run)
            if any(job.get('state') in ('queued', 'running') and (job.get('run_path') == str(run) or job.get('recording_id') == run_id) for job, _ in self._jobs()):
                raise ValueError('This recording is being prepared; cancel and wait for preparation to stop')
            with file_lock(run / 'prepare.lock', timeout=0.1):
                entry = _recording_entry(run)
                raw = read_json(run / 'manifest.json', {})
                source = raw.get('source', {}).get('path')
                if source and Path(source).resolve().is_relative_to(run):
                    raise ValueError('The original source video is inside this folder; move the original video before deleting the prepared recording')
            record = LocalTrash(self.project).put('recording', run, entry['label'], recording_id=run_id)
            with self._catalog_lock:
                self._catalog_cache.pop(run, None)
            return dict(deleted=True, trash_id=record['id'], recording_id=run_id)

    def restore(self, trash_id):
        self._prepare()
        with file_lock(self.root / 'start.lock'):
            result = LocalTrash(self.project).restore(trash_id, kind='recording', commit=lambda record, path: _recording_entry(path))
            with self._catalog_lock:
                self._catalog_cache.pop(Path(result['path']), None)
            return result

    def cancel_preparation(self, payload):
        if not isinstance(payload, dict) or set(payload) != {'job_id'}:
            raise ValueError('Expected an exact video preparation job ID')
        job, folder = self._job(payload['job_id'])
        with file_lock(folder / 'status.lock'):
            job = read_json(folder / 'job.json', job)
            if job.get('state') not in ('queued', 'running'):
                return self._public(job, folder)
            flag = _ordinary(folder / 'cancel.flag')
            if not flag.exists():
                with flag.open('xb') as stream:
                    stream.write(b'cancel\n')
                    stream.flush()
                    os.fsync(stream.fileno())
            fields = dict(cancel_requested=True, updated_at=utc_now(), message='Cancellation requested; finishing the current preparation work unit')
            if job['state'] == 'queued':
                fields.update(state='cancelled', message='Video preparation was cancelled before it started')
            job.update(fields)
            atomic_json(folder / 'job.json', job)
        return self._public(job, folder)

    def _ensure_idle(self):
        if any(job.get('state') in ('queued', 'running') for job, _ in self._jobs()):
            raise ValueError('A video preparation job is already running')

    def resume_preparation(self, payload):
        if not isinstance(payload, dict) or set(payload) != {'job_id'}:
            raise ValueError('Expected an exact video preparation job ID')
        self._prepare()
        with file_lock(self.root / 'start.lock'):
            self._ensure_idle()
            job, _ = self._job(payload['job_id'])
            run = self._resume_path(job)
            _options({**job['options'], 'selection_id': job['selection_id'], 'model_id': job['model_id']})
            video = Path(job['video_path']).resolve(strict=True)
            stat = video.stat()
            if not video.is_file() or (stat.st_size, stat.st_mtime_ns) != (job.get('size_bytes'), job.get('mtime_ns')):
                raise ValueError('The original video source changed; resume requires the exact original video')
            model = self.registry.validate(job['model_id'])
            if Path(model['path']).resolve() != Path(job['model_path']).resolve() or hashlib.sha256(Path(model['path']).read_bytes()).hexdigest() != job.get('model_sha256'):
                raise ValueError('The original annotation model changed; resume requires the exact original model')
            if not self.python.is_file():
                raise ValueError('The prepared runtime is missing; run setup.bat first')
            if run is not None:
                with file_lock(run / 'prepare.lock', timeout=0.1):
                    pass
            fields = {key: job[key] for key in ('options', 'selection_id', 'video_path', 'size_bytes', 'mtime_ns', 'model_id', 'model_path', 'model_sha256')}
            fields.update(resume_from_job_id=job['id'], resume_path=str(run) if run is not None else None, run_path=str(run) if run is not None else None, recording_id=recording_id(run) if run is not None else None)
            return self._launch_job(fields)

    def browse(self):
        self._prepare()
        with file_lock(self.root / 'browse.lock', timeout=0.1):
            if self.picker is None:
                from local_video_picker import pick_local_video
                selected = pick_local_video(self.python)
            else:
                selected = self.picker()
            if selected is None:
                return {'cancelled': True}
            path = Path(selected).expanduser().resolve(strict=True)
            if not path.is_file():
                raise ValueError('Choose an existing local video file')
            if self.inspector is None:
                from prepare_dataset import inspect_video
                metadata = inspect_video(path)
            else:
                metadata = self.inspector(path)
            metadata = _validate_metadata(metadata)
            stat = path.stat()
            if stat.st_size <= 0:
                raise ValueError('The selected video is empty')
            selections = self.root / 'selections'
            if selections.is_symlink() or selections.resolve().parent != self.root:
                raise ValueError('Video selections must stay inside their managed directory')
            selections.mkdir(exist_ok=True)
            while True:
                selection_id = 'selection-' + secrets.token_hex(12)
                manifest = selections / f'{selection_id}.json'
                try:
                    with manifest.open('x', encoding='utf-8'):
                        pass
                    break
                except FileExistsError:
                    continue
            selection = dict(selection_id=selection_id, name=path.name, path=str(path), metadata=metadata, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, created_at=utc_now())
            atomic_json(manifest, selection)
            return {key: selection[key] for key in ('selection_id', 'name', 'path', 'metadata')}

    def _selection(self, selection_id):
        if not isinstance(selection_id, str) or not SELECTION_ID.fullmatch(selection_id):
            raise ValueError('Choose a video using Browse video first')
        manifest = self.root / 'selections' / f'{selection_id}.json'
        if manifest.is_symlink() or manifest.resolve().parent != self.root / 'selections':
            raise ValueError('Video selection leaves its managed directory')
        selection = read_json(manifest)
        if not isinstance(selection, dict) or selection.get('selection_id') != selection_id:
            raise ValueError('Unknown video selection; use Browse video again')
        try:
            path = Path(selection['path']).resolve(strict=True)
            stat = path.stat()
            metadata = _validate_metadata(selection['metadata'])
        except (OSError, KeyError, TypeError) as exc:
            raise ValueError('The selected video is no longer available; browse again') from exc
        if not path.is_file() or (stat.st_size, stat.st_mtime_ns) != (selection.get('size_bytes'), selection.get('mtime_ns')):
            raise ValueError('The selected video changed; use Browse video again')
        return {**selection, 'path': str(path), 'metadata': metadata}

    def start_preparation(self, payload):
        values = _options(payload)
        selection = self._selection(values['selection_id'])
        selected = self.registry.resolve(values['model_id'])
        focus = self.registry.for_view(selected['id'], 'focus')
        model = self.registry.validate((focus or selected)['id'])
        model_sha256 = hashlib.sha256(Path(model['path']).read_bytes()).hexdigest()
        if not self.python.is_file():
            raise ValueError('The prepared runtime is missing; run setup.bat first')
        parent = self.project / 'dataset-runs'
        if parent.is_symlink() or parent.resolve().parent != self.project:
            raise ValueError('dataset-runs must stay inside the project')
        self._prepare()
        with file_lock(self.root / 'start.lock'):
            self._ensure_idle()
            options = {key: values[key] for key in DEFAULTS}
            return self._launch_job(dict(options=options, selection_id=selection['selection_id'], video_path=selection['path'], size_bytes=selection['size_bytes'], mtime_ns=selection['mtime_ns'], model_id=model['id'], model_path=model['path'], model_sha256=model_sha256))

    def _launch_job(self, fields):
        for _ in range(100):
            job_id = 'video-' + secrets.token_hex(8)
            folder = self.root / job_id
            try:
                folder.mkdir(exist_ok=False)
                break
            except FileExistsError:
                continue
        else:
            raise ValueError('Could not reserve a unique video preparation job')
        job = {**dict(id=job_id, state='queued', project=str(self.project), processed=0, total=0, saved=0, progress=0, recording_id=None, run_path=None, cancel_requested=False, message='Starting local video preparation', created_at=utc_now(), updated_at=utc_now()), **fields}
        manifest = folder / 'job.json'
        atomic_json(manifest, job)
        atomic_json(self.root / 'latest.json', {'id': job_id})
        try:
            env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONUTF8='1', YOLO_OFFLINE='true', YOLO_AUTOINSTALL='false', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            command = [str(self.python), '-u', str(PROJECT / 'recording_library.py'), '--worker', str(manifest)]
            flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0) | getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
            with (folder / 'bootstrap.log').open('ab') as bootstrap:
                process = self.launcher(command, cwd=str(self.project), env=env, stdin=subprocess.DEVNULL, stdout=bootstrap, stderr=subprocess.STDOUT, close_fds=True, creationflags=flags)
            with file_lock(folder / 'status.lock'):
                job = read_json(manifest, job)
                job['launch_pid'] = process.pid
                atomic_json(manifest, job)
        except Exception as exc:
            job.update(state='failed', message=f'Could not start video preparation: {exc}')
            atomic_json(manifest, job)
            raise ValueError(job['message']) from exc
        return self._public(job, folder)

def run_preparation_job(manifest, *, registry=None, preparer=None):
    from prepare_dataset import PreparationCancelled
    manifest = _ordinary(Path(manifest).absolute())
    job = read_json(manifest)
    if not isinstance(job, dict) or not isinstance(job.get('id'), str) or (not JOB_ID.fullmatch(job['id'])):
        raise ValueError('Missing or invalid preparation job metadata')
    project = Path(job['project']).resolve()
    folder = project / 'video-jobs' / job['id']
    if manifest != folder / 'job.json' or folder.is_symlink():
        raise ValueError('Preparation job must stay inside its managed folder')
    registry = registry or ModelRegistry(project)

    def update(**fields):
        nonlocal job
        with file_lock(folder / 'status.lock'):
            job = read_json(manifest, job)
            job.update(fields, updated_at=utc_now())
            atomic_json(manifest, job)
    _ordinary(folder.parent)
    _ordinary(folder)
    with file_lock(folder / 'worker.lock', timeout=0.1), file_lock(folder.parent / 'process.lock', timeout=0.1):
        with file_lock(folder / 'status.lock'):
            job = read_json(manifest, {})
            if job.get('state') != 'queued':
                raise ValueError('This preparation job has already started or finished')
            if job.get('cancel_requested') or _ordinary(folder / 'cancel.flag').exists():
                job.update(state='cancelled', updated_at=utc_now(), message='Video preparation was cancelled before it started')
                atomic_json(manifest, job)
                return 130
            if _bootstrap_expired(job):
                job.update(state='failed', updated_at=utc_now(), message='The preparation worker did not start')
                atomic_json(manifest, job)
                raise ValueError('This preparation job expired before its worker started')
            job.update(state='running', worker_pid=os.getpid(), updated_at=utc_now(), message='Inspecting the selected local recording')
            atomic_json(manifest, job)
        with (folder / 'preparation.log').open('a', encoding='utf-8', buffering=1) as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                try:
                    _options({**job['options'], 'selection_id': job['selection_id'], 'model_id': job['model_id']})
                    model = registry.validate(job['model_id'])
                    if Path(model['path']).resolve() != Path(job['model_path']).resolve():
                        raise ValueError('The selected model changed before preparation started')
                    if hashlib.sha256(Path(model['path']).read_bytes()).hexdigest() != job.get('model_sha256'):
                        raise ValueError('The selected model content changed before preparation started')
                    video = Path(job['video_path']).resolve(strict=True)
                    stat = video.stat()
                    if not video.is_file() or (stat.st_size, stat.st_mtime_ns) != (job['size_bytes'], job['mtime_ns']):
                        raise ValueError('The selected video changed before preparation started')
                    parent = project / 'dataset-runs'
                    if parent.is_symlink() or parent.resolve().parent != project:
                        raise ValueError('Dataset output must stay inside the project')
                    last_update = 0

                    def progress(event):
                        nonlocal last_update
                        fields = {}
                        if 'run_path' in event:
                            run = Path(event['run_path']).resolve()
                            if run.parent != parent or run.is_symlink():
                                raise ValueError('Preparation output leaves the dataset library')
                            if str(run) != job.get('run_path'):
                                fields.update(run_path=str(run), recording_id=recording_id(run))
                        for key in ('processed', 'total', 'saved'):
                            if key in event:
                                if type(event[key]) is not int or event[key] < 0:
                                    raise ValueError('Invalid preparation progress')
                                fields[key] = event[key]
                        processed, total = (fields.get('processed', job['processed']), fields.get('total', job['total']))
                        fields['progress'] = min(99, 100 * processed / max(1, total))
                        fields['message'] = f"Processed {processed}/{total} samples; saved {fields.get('saved', job['saved'])} draft frames"
                        if isinstance(event.get('message'), str) and len(event['message']) <= 400:
                            fields['message'] = event['message']
                        now = time.monotonic()
                        if now - last_update >= 0.2 or 'run_path' in fields or processed >= total:
                            update(**fields)
                            last_update = now
                    if preparer is None:
                        from prepare_dataset import prepare_video
                        preparer = prepare_video
                    def cancelled():
                        return _ordinary(folder / 'cancel.flag').exists()
                    if cancelled():
                        raise PreparationCancelled('Video preparation cancellation requested')
                    keywords = dict(view='focus', **job['options'], progress_callback=progress)
                    parameters = inspect.signature(preparer).parameters
                    variadic = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
                    if variadic or 'cancel_callback' in parameters:
                        keywords['cancel_callback'] = cancelled
                    if job.get('resume_path'):
                        resumed = checked_managed_path(project, 'recording', job['resume_path'])
                        if variadic or 'resume' in parameters:
                            keywords['resume'] = resumed
                        else:
                            raise ValueError('The preparation worker does not support resuming an unfinished run')
                    print(f"Preparing local video: {video}\nAnnotation model: {model['path']}\nAll labels remain unreviewed drafts.", flush=True)
                    run = Path(preparer(video, parent, Path(model['path']), **keywords)).resolve(strict=True)
                    if run.parent != parent or run.is_symlink():
                        raise ValueError('Preparation returned an output outside the dataset library')
                    entry = _recording_entry(run)
                    if entry['status'] != 'complete' or not entry['frames']:
                        raise ValueError('Preparation did not produce a completed draft recording')
                    raw = read_json(run / 'manifest.json', {})
                    total = raw.get('statistics', {}).get('scheduled', job['total'])
                    update(state='completed', cancel_requested=False, processed=total, total=total, saved=entry['frames'], progress=100, recording_id=entry['id'], run_path=str(run), message=f"Prepared {entry['frames']} draft frames. Open the recording to review them.")
                    print(job['message'], flush=True)
                    return 0
                except PreparationCancelled:
                    update(state='cancelled', cancel_requested=False, message='Video preparation was cancelled; complete draft frames were saved and can be resumed')
                    return 130
                except KeyboardInterrupt:
                    update(state='interrupted', cancel_requested=False, message='Video preparation stopped; complete draft frames were saved and can be resumed')
                    return 130
                except Exception as exc:
                    update(state='failed', message=f'Video preparation failed: {exc}')
                    traceback.print_exc()
                    return 1

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, required=True)
    return run_preparation_job(parser.parse_args(argv).worker)
if __name__ == '__main__':
    raise SystemExit(main())
