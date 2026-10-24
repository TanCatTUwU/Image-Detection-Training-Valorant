from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import secrets
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from model_registry import ModelRegistry, atomic_json, file_lock, process_alive, read_json, utc_now
PROJECT = Path(__file__).resolve().parent
JOB_ID = re.compile('empty-[a-f0-9]{16}\\Z')
FRAME_ID = re.compile('frame-[0-9]{6,12}\\Z')
ACTIVE = {'queued', 'running'}
CONFIDENCE = 0.08

def _run_key(root):
    return hashlib.sha256(os.path.normcase(str(Path(root).resolve())).encode('utf-8')).hexdigest()

def _candidate(frame):
    return frame.get('status') == 'pending' and frame.get('boxes') == [] and (type(frame.get('revision')) is int) and (frame['revision'] == 0)

def _load_candidates(job, folder):
    if 'candidate_file' in job:
        path = folder / 'candidates.json'
        if job['candidate_file'] != 'candidates.json' or path.is_symlink() or path.resolve().parent != folder:
            raise ValueError('Empty scan candidates must use the managed candidates.json file')
        candidates = read_json(path)
    else:
        candidates = job.get('candidates')
    if type(job.get('total')) is not int or job['total'] < 1 or not isinstance(candidates, list) or len(candidates) != job['total']:
        raise ValueError('Invalid empty scan candidate count')
    seen = set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get('id'), str) or not FRAME_ID.fullmatch(candidate['id']) or type(candidate.get('revision')) is not int or candidate['revision'] != 0 or candidate['id'] in seen:
            raise ValueError('Empty scan candidates must be unique untouched frame IDs with revision zero')
        seen.add(candidate['id'])
    return candidates

def _expired(job):
    return job.get('state') == 'queued' and (not job.get('worker_pid')) and ((datetime.now(timezone.utc) - datetime.fromisoformat(job['created_at'])).total_seconds() > 45)

def _public(job, folder):
    result = {key: job.get(key) for key in ('id', 'state', 'model_id', 'model_label', 'inference_backend', 'inference_device', 'runtime_label', 'confidence', 'reason_counts', 'total', 'processed', 'removed', 'kept', 'skipped', 'message')}
    result['confidence'] = job.get('confidence', CONFIDENCE)
    result['log_tail'] = ''
    for name in ('scan.log', 'bootstrap.log'):
        try:
            with (folder / name).open('rb') as stream:
                stream.seek(max(0, (folder / name).stat().st_size - 6000))
                result['log_tail'] += stream.read().decode('utf-8', errors='replace')
        except OSError:
            pass
    return result

class EmptyScanManager:

    def __init__(self, project, *, registry=None, launcher=None, alive=None, runtime_python=None):
        self.project = Path(project).resolve()
        self.root = self.project / 'empty-scan-jobs'
        self.registry = registry or ModelRegistry(self.project)
        self.launcher = launcher or subprocess.Popen
        self.alive = alive or process_alive
        self.python = Path(runtime_python) if runtime_python else self.project / 'runtime-env/Scripts/python.exe'

    def _prepare(self):
        if self.root.is_symlink() or self.root.resolve().parent != self.project:
            raise ValueError('Empty scan jobs must stay inside the project')
        self.root.mkdir(exist_ok=True)

    def _job(self, job_id):
        if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
            return (None, None)
        folder = self.root / job_id
        if folder.is_symlink() or folder.resolve().parent != self.root:
            raise ValueError('Empty scan job leaves its managed folder')
        path = folder / 'job.json'
        job = read_json(path)
        if not isinstance(job, dict):
            return (None, None)
        if job.get('state') in ACTIVE:
            pid = job.get('worker_pid') or job.get('launch_pid')
            if _expired(job) or (pid and (not self.alive(pid))):
                with file_lock(folder / 'status.lock'):
                    job = read_json(path, job)
                    pid = job.get('worker_pid') or job.get('launch_pid')
                    if job.get('state') in ACTIVE and (_expired(job) or (pid and (not self.alive(pid)))):
                        job.update(state='interrupted' if job.get('worker_pid') else 'failed', message='The empty scan worker stopped before completion' if job.get('worker_pid') else 'The empty scan worker did not start', updated_at=utc_now())
                        atomic_json(path, job)
        return (job, folder)

    def snapshot(self, store):
        runs = read_json(self.root / 'runs.json', {})
        job_id = runs.get(_run_key(store.root)) if isinstance(runs, dict) else None
        job, folder = self._job(job_id)
        if job is None:
            return None
        if Path(job.get('run_path', '')).resolve() != Path(store.root).resolve():
            raise ValueError('Empty scan job belongs to another recording')
        return _public(job, folder)

    def start(self, payload, store):
        if not isinstance(payload, dict) or set(payload) - {'model_id', 'confidence'} or not isinstance(payload.get('model_id'), str):
            raise ValueError('Choose one known model for the empty scan')
        confidence = payload.get('confidence', CONFIDENCE)
        if type(confidence) not in (int, float) or not .05 <= confidence <= .95 or not math.isfinite(confidence):
            raise ValueError('Scan confidence must be a finite number from 0.05 to 0.95')
        chosen = self.registry.resolve(payload['model_id'])
        model = self.registry.for_view(chosen['id'], 'focus') or self.registry.for_view(chosen['id'], 'full') or chosen
        model = self.registry.validate(model['id'])
        shape = model['input_shape']
        if len(shape) != 4 or min(shape[2:]) < 320:
            raise ValueError('Empty screening requires a detailed focus/full model with both input dimensions at least 320')
        data = store.manifest()
        if data.get('status') != 'complete':
            raise ValueError('Wait until recording preparation is complete before screening')
        candidates = [dict(id=frame['id'], revision=0) for frame in data['frames'] if _candidate(frame)]
        if not candidates:
            raise ValueError('No untouched pending empty frames are available to screen')
        if not Path(store.root).is_dir() or not self.python.is_file():
            raise ValueError('The recording or prepared Python runtime is unavailable')
        fingerprint = hashlib.sha256(Path(model['path']).read_bytes()).hexdigest()
        self._prepare()
        with file_lock(self.root / 'start.lock'):
            pointer = read_json(self.root / 'active.json', {})
            previous, _ = self._job(pointer.get('id') if isinstance(pointer, dict) else None)
            if previous and previous['state'] in ACTIVE:
                raise ValueError('An empty scan is already running; wait or cancel it first')
            for _ in range(100):
                job_id = 'empty-' + secrets.token_hex(8)
                folder = self.root / job_id
                try:
                    folder.mkdir(exist_ok=False)
                    break
                except FileExistsError:
                    continue
            else:
                raise ValueError('Could not reserve an empty scan job')
            job = dict(id=job_id, project=str(self.project), run_path=str(Path(store.root).resolve()), state='queued', model_id=model['id'], model_label=model['label'], model_path=model['path'], model_sha256=fingerprint, confidence=confidence, reason_counts={}, candidate_file='candidates.json', total=len(candidates), processed=0, removed=0, kept=0, skipped=0, message='Starting model screening', created_at=utc_now(), updated_at=utc_now())
            manifest = folder / 'job.json'
            atomic_json(folder / 'candidates.json', candidates)
            atomic_json(manifest, job)
            runs = read_json(self.root / 'runs.json', {})
            if not isinstance(runs, dict):
                raise ValueError('Empty scan index is damaged')
            runs[_run_key(store.root)] = job_id
            atomic_json(self.root / 'runs.json', runs)
            atomic_json(self.root / 'active.json', {'id': job_id})
            try:
                flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0) | getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0) | getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0)
                env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONUTF8='1', YOLO_OFFLINE='true', YOLO_AUTOINSTALL='false', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
                with (folder / 'bootstrap.log').open('ab') as log:
                    process = self.launcher([str(self.python), '-u', str(PROJECT / 'empty_scan.py'), '--worker', str(manifest)], cwd=str(self.project), env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, close_fds=True, creationflags=flags)
                with file_lock(folder / 'status.lock'):
                    job = read_json(manifest, job)
                    job['launch_pid'] = process.pid
                    atomic_json(manifest, job)
            except Exception as exc:
                job.update(state='failed', message=f'Could not start empty screening: {exc}')
                atomic_json(manifest, job)
                raise ValueError(job['message']) from exc
        return _public(job, folder)

    def cancel(self, store):
        runs = read_json(self.root / 'runs.json', {})
        job, folder = self._job(runs.get(_run_key(store.root)) if isinstance(runs, dict) else None)
        if job is None:
            raise ValueError('There is no empty scan for this recording')
        if Path(job['run_path']).resolve() != Path(store.root).resolve():
            raise ValueError('Empty scan job belongs to another recording')
        with file_lock(folder / 'status.lock'):
            job = read_json(folder / 'job.json', job)
            if job['state'] in ACTIVE:
                (folder / 'cancel.flag').touch(exist_ok=True)
                if job['state'] == 'queued':
                    job['state'] = 'cancelled'
                job.update(message='Empty screening cancelled' if job['state'] == 'cancelled' else 'Cancellation requested', updated_at=utc_now())
                atomic_json(folder / 'job.json', job)
        return _public(job, folder)

class ScanCancelled(Exception):
    pass

def _quality_reason(pixels):
    import cv2
    import numpy as np
    if pixels is None or pixels.shape != (448, 640, 3):
        return 'The 640x448 crop could not be read'
    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    if float(gray.mean()) < 20 or float((gray < 16).mean()) > 0.8:
        return 'Crop is too dark for strict empty screening'
    if float(gray.mean()) > 235 or float((pixels.min(axis=2) > 220).mean()) > 0.45:
        return 'Crop is washed out or too bright'
    if float(cv2.Laplacian(gray, cv2.CV_32F).var()) < 15:
        return 'Crop has insufficient sharpness'
    return None

def _positions(size, tile, step):
    return sorted(set([0, max(0, size - tile), *range(0, max(1, size - tile + 1), step)]))

def _views(pixels, *, dense, flipped):
    import numpy as np
    height, width = pixels.shape[:2]
    for flip in [False, True] if flipped else [False]:
        source = np.ascontiguousarray(pixels[:, ::-1]) if flip else pixels
        yield (source, (0, 0, width, height), flip)
        tw, th = (min(384, width), min(320, height))
        for y in _positions(height, th, th):
            for x in _positions(width, tw, tw):
                yield (source[y:y + th, x:x + tw], (x, y, tw, th), flip)
        if dense:
            for y in _positions(height, 320, 160):
                for x in _positions(width, 320, 160):
                    yield (source[y:y + 320, x:x + 320], (x, y, 320, 320), flip)

def _mapped(box, region, flipped):
    x, y, width, height = region
    result = {**box, 'x1': (x + box['x1'] * width) / 640, 'x2': (x + box['x2'] * width) / 640, 'y1': (y + box['y1'] * height) / 448, 'y2': (y + box['y2'] * height) / 448}
    if flipped:
        result['x1'], result['x2'] = (1 - result['x2'], 1 - result['x1'])
    return result

def _scan_pixels(proposer, pixels, *, dense, flipped, cancelled):
    passes = 0
    contains_targets = getattr(proposer, 'contains_targets', None)
    for crop, region, flip in _views(pixels, dense=dense, flipped=flipped):
        if cancelled():
            raise ScanCancelled()
        if callable(contains_targets):
            found = contains_targets(crop)
            passes += 1
            if type(found) is not bool:
                raise ValueError('The detector returned invalid empty-screening evidence')
            if found:
                return (True, passes)
            continue
        detections = proposer._pass(crop)
        passes += 1
        if not isinstance(detections, list):
            raise ValueError('The detector returned invalid empty-screening evidence')
        for box in detections:
            if not isinstance(box, dict) or type(box.get('class_id')) is not int or box['class_id'] not in (0, 1):
                raise ValueError('The detector returned an unsupported class')
            _mapped(box, region, flip)
        if detections:
            return (True, passes)
    return (False, passes)

def verify_empty(proposer, pixels, data, frame, *, neighbor_loader=None, cancelled=lambda: False):
    reason = _quality_reason(pixels)
    if reason:
        return ('skipped', 0, reason)
    found, passes = _scan_pixels(proposer, pixels, dense=True, flipped=True, cancelled=cancelled)
    if found:
        return ('kept', passes, 'A Head or Body was detected')
    if cancelled():
        raise ScanCancelled()
    if neighbor_loader is None:
        from annotation_assist import load_neighbors
        neighbor_loader = lambda data, frame, pixels: load_neighbors(data, frame, pixels, strict=True)
    neighbors, warning = neighbor_loader(data, frame, pixels)
    if cancelled():
        raise ScanCancelled()
    if warning or len(neighbors) != 2:
        return ('skipped', passes, warning or 'Both original neighboring frames are required')
    for neighbor in neighbors:
        reason = _quality_reason(neighbor)
        if reason:
            return ('skipped', passes, f'Neighbor: {reason}')
    for neighbor in neighbors:
        found, count = _scan_pixels(proposer, neighbor, dense=False, flipped=False, cancelled=cancelled)
        passes += count
        if found:
            return ('kept', passes, 'A Head or Body was detected in a neighboring source frame')
    return ('removed', passes, 'No targets found across strict model views and two source neighbors; not a confirmed negative')

def run_empty_scan(manifest, *, registry=None, store_factory=None, proposer_factory=None, image_reader=None, verifier=None):
    manifest = Path(manifest).resolve()
    job = read_json(manifest)
    if not isinstance(job, dict) or not isinstance(job.get('id'), str) or (not JOB_ID.fullmatch(job['id'])):
        raise ValueError('Invalid empty scan job metadata')
    project = Path(job['project']).resolve()
    folder = project / 'empty-scan-jobs' / job['id']
    if manifest != folder / 'job.json' or folder.is_symlink():
        raise ValueError('Empty scan must stay inside its managed folder')
    registry = registry or ModelRegistry(project)

    def update(**fields):
        nonlocal job
        with file_lock(folder / 'status.lock'):
            job = read_json(manifest, job)
            job.update(fields, updated_at=utc_now())
            atomic_json(manifest, job)
    cancelled = lambda: (folder / 'cancel.flag').exists()
    proposer = None
    with file_lock(folder / 'worker.lock', timeout=0.1):
        with file_lock(folder / 'status.lock'):
            job = read_json(manifest, {})
            if job.get('state') != 'queued':
                raise ValueError('This empty scan has already started or finished')
            if _expired(job):
                job.update(state='failed', message='The empty scan worker did not start', updated_at=utc_now())
                atomic_json(manifest, job)
                raise ValueError('This empty scan expired before startup')
            job.update(state='running', worker_pid=os.getpid(), message='Checking untouched pending frames', updated_at=utc_now())
            atomic_json(manifest, job)
        with (folder / 'scan.log').open('a', encoding='utf-8', buffering=1) as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                try:
                    if cancelled():
                        raise ScanCancelled()
                    candidates = _load_candidates(job, folder)
                    confidence = job.get('confidence', CONFIDENCE)
                    if type(confidence) not in (int, float) or not .05 <= confidence <= .95 or not math.isfinite(confidence):
                        raise ValueError('Invalid empty scan confidence')
                    model = registry.validate(job['model_id'])
                    if Path(model['path']).resolve() != Path(job['model_path']).resolve() or hashlib.sha256(Path(model['path']).read_bytes()).hexdigest() != job['model_sha256']:
                        raise ValueError('The selected detailed model changed before empty screening')
                    if min(model['input_shape'][2:]) < 320:
                        raise ValueError('Empty screening requires a detailed model')
                    if store_factory is None:
                        from review_dataset import ReviewStore
                        store_factory = ReviewStore
                    store = store_factory(Path(job['run_path']))
                    if Path(store.root).resolve() != Path(job['run_path']).resolve():
                        raise ValueError('The scan store belongs to another recording')
                    if proposer_factory is None:
                        from gpu_screening import create_screening_proposer
                        proposer_factory = create_screening_proposer
                    proposer = proposer_factory(Path(model['path']), 0, confidence, True, strict=True)
                    execution = getattr(proposer, 'execution', {})
                    if isinstance(execution, dict):
                        update(inference_backend=execution.get('backend'), inference_device=execution.get('device'), runtime_label=execution.get('label'))
                    if image_reader is None:
                        import cv2
                        image_reader = lambda path: cv2.imread(str(path))
                    print(f"Empty scan: {job['total']} untouched candidates; model={model['path']}; confidence={confidence}", flush=True)
                    for candidate in candidates:
                        if cancelled():
                            raise ScanCancelled()
                        frame_context = getattr(store, 'frame_context', None)
                        if callable(frame_context):
                            try:
                                data = frame_context(candidate['id'])
                            except KeyError:
                                data = {'frames': []}
                        else:
                            data = store.manifest()
                        frame = next((frame for frame in data['frames'] if frame['id'] == candidate['id']), None)
                        decision, passes, reason = ('skipped', 0, 'Frame changed or is no longer an untouched pending frame')
                        if frame is not None and _candidate(frame) and (frame['revision'] == candidate['revision']):
                            pixels = image_reader(store.image_path(frame['id']))
                            decision, passes, reason = (verifier or verify_empty)(proposer, pixels, data, frame, cancelled=cancelled)
                            if cancelled():
                                raise ScanCancelled()
                            if decision == 'removed':
                                evidence = dict(decision='empty', model_id=job['model_id'], model_sha256=job['model_sha256'], confidence=confidence, detections=0, temporal_frames=2, passes=passes, checked_at=utc_now(), method='strict-onnx-multiview-v1')
                                if not store.mark_auto_removed(frame['id'], candidate['revision'], evidence):
                                    decision, reason = ('skipped', 'Frame was edited during verification')
                        if decision not in ('removed', 'kept', 'skipped'):
                            raise ValueError('Invalid empty scan decision')
                        reason_key = ('neighbor_detection' if 'neighboring source frame' in reason else 'current_detection') if decision == 'kept' else decision
                        reason_counts = {**job.get('reason_counts', {})}
                        reason_counts[reason_key] = reason_counts.get(reason_key, 0) + 1
                        print(json.dumps(dict(frame_id=candidate['id'], decision=decision, reason=reason, passes=passes, confidence=confidence), ensure_ascii=False), flush=True)
                        update(processed=job['processed'] + 1, reason_counts=reason_counts, **{decision: job[decision] + 1}, message=f"{(frame['id'] if frame else candidate['id'])}: {reason}")
                    update(state='completed', message=f"Screening complete: {job['removed']} automatically grouped; {job['kept']} kept; {job['skipped']} skipped. Nothing was approved.")
                    return 0
                except (ScanCancelled, KeyboardInterrupt):
                    update(state='cancelled', message='Empty screening cancelled. Prior checked results were preserved; nothing was approved.')
                    return 130
                except Exception as exc:
                    update(state='failed', message=f'Empty screening failed: {exc}')
                    traceback.print_exc()
                    return 1
                finally:
                    session = getattr(getattr(proposer, 'model', None), 'session', None)
                    if session is not None and callable(getattr(session, 'close', None)):
                        session.close()

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, required=True)
    return run_empty_scan(parser.parse_args(argv).worker)
if __name__ == '__main__':
    raise SystemExit(main())
