from __future__ import annotations
import argparse
import copy
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import shutil
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
PROJECT = Path(__file__).resolve().parent
MAX_BODY = 1024 * 1024
FRAME_ID = re.compile('frame-[0-9]{6,12}\\Z')
STATUSES = {'pending', 'approved', 'rejected', 'auto_removed'}
MANUAL_STATUSES = {'pending', 'approved', 'rejected'}

class RevisionConflict(ValueError):
    pass

def validate_boxes(boxes):
    if not isinstance(boxes, list) or len(boxes) > 4096:
        raise ValueError('boxes must be a list with at most 4096 rectangles')
    result = []
    for box in boxes:
        if not isinstance(box, dict) or type(box.get('class_id')) is not int or box['class_id'] not in (0, 1):
            raise ValueError('Each box requires class_id 0 (Head) or 1 (Body)')
        values = [box.get(key) for key in ('x1', 'y1', 'x2', 'y2')]
        if any((type(value) not in (int, float) or not math.isfinite(value) or (not 0 <= value <= 1) for value in values)):
            raise ValueError('Box coordinates must be finite numbers normalized to 0..1')
        x1, y1, x2, y2 = values
        if x2 <= x1 or y2 <= y1:
            raise ValueError('Boxes must have positive width and height')
        clean = dict(zip(('x1', 'y1', 'x2', 'y2'), values), class_id=box['class_id'])
        if 'score' in box:
            score = box['score']
            if type(score) not in (int, float) or not math.isfinite(score) or (not 0 <= score <= 1):
                raise ValueError('Box score must be a finite number in 0..1')
            clean['score'] = score
        result.append(clean)
    return result

def yolo_text(boxes):
    rows = []
    for box in validate_boxes(boxes):
        x1, y1, x2, y2 = (box[key] for key in ('x1', 'y1', 'x2', 'y2'))
        rows.append(f"{box['class_id']} {(x1 + x2) / 2:.8f} {(y1 + y2) / 2:.8f} {x2 - x1:.8f} {y2 - y1:.8f}\n")
    return ''.join(rows)

def _file_signature(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

def atomic_write(path: Path, data: bytes):
    temporary = path.with_name(f'review-tmp-{secrets.token_hex(8)}.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            signature = _file_signature(os.fstat(stream.fileno()))
        os.replace(temporary, path)
        return signature
    finally:
        temporary.unlink(missing_ok=True)

class ReviewStore:

    def __init__(self, run_dir):
        self.root = Path(run_dir).expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError('RUN_DIR must be an existing local directory')
        self._thread_lock = threading.RLock()
        self._manifest_cache = None
        self._manifest_signature = None
        self.manifest()

    def _path(self, relative):
        if not isinstance(relative, str) or '\\' in relative or ':' in relative:
            raise ValueError('Expected a known relative run path')
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise ValueError('Path escapes the review run')
        return path

    @contextmanager
    def _locked(self):
        with self._thread_lock:
            path = self._path('review.lock')
            with path.open('a+b') as stream:
                if stream.seek(0, 2) == 0:
                    stream.write(b'0')
                    stream.flush()
                deadline = time.monotonic() + 5
                while True:
                    try:
                        stream.seek(0)
                        if os.name == 'nt':
                            import msvcrt
                            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                        else:
                            import fcntl
                            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise ValueError('Another reviewer is saving; try again')
                        time.sleep(0.05)
                try:
                    yield
                finally:
                    stream.seek(0)
                    if os.name == 'nt':
                        import msvcrt
                        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def manifest(self):
        return self._manifest_snapshot()

    def _manifest_snapshot(self, *, clone=True):
        with self._thread_lock:
            path = self._path('manifest.json')
            for attempt in range(3):
                signature = _file_signature(path.stat())
                if self._manifest_cache is not None and signature == self._manifest_signature:
                    return copy.deepcopy(self._manifest_cache) if clone else self._manifest_cache
                data = self._load_manifest()
                if _file_signature(path.stat()) == signature:
                    self._manifest_cache = data
                    self._manifest_signature = signature
                    return copy.deepcopy(data) if clone else data
            raise RevisionConflict('The recording changed while loading. Reload it again.')

    def _cache_committed_manifest(self, data, signature):
        self._manifest_cache = data
        self._manifest_signature = signature

    def _writable_frame(self, frame_id):
        snapshot = self._manifest_snapshot(clone=False)
        self._complete(snapshot)
        for index, original in enumerate(snapshot['frames']):
            if original['id'] == frame_id:
                data = {**snapshot, 'frames': list(snapshot['frames'])}
                frame = copy.deepcopy(original)
                data['frames'][index] = frame
                return (data, frame)
        raise KeyError('Unknown frame ID')

    def _load_manifest(self):
        path = self._path('manifest.json')
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict) or data.get('schema_version') != 1 or data.get('status') not in ('running', 'complete'):
            raise ValueError('Expected a schema_version 1 extraction manifest (running or complete)')
        if not isinstance(data.get('frames'), list):
            raise ValueError('Manifest frames must be a list')
        self._path('images')
        self._path('draft-labels')
        ids = set()
        for frame in data['frames']:
            if not isinstance(frame, dict) or not isinstance(frame.get('id'), str) or (not FRAME_ID.fullmatch(frame['id'])):
                raise ValueError('Unsafe or invalid frame ID')
            frame_id = frame['id']
            if frame_id in ids:
                raise ValueError('Duplicate frame ID')
            ids.add(frame_id)
            for field, expected in (('image', f'images/{frame_id}.png'), ('label', f'draft-labels/{frame_id}.txt')):
                if frame.get(field) != expected:
                    raise ValueError(f'Frame {field} must use the known run path {expected}')
            if frame.get('status') not in STATUSES or frame.get('split') not in ('train', 'val'):
                raise ValueError('Unknown frame status or split')
            if type(frame.get('revision')) is not int or frame['revision'] < 0:
                raise ValueError('Frame revision must be a non-negative integer')
            if frame.get('width') != 640 or frame.get('height') != 448:
                raise ValueError('Review frames must use the 640 x 448 crop')
            if type(frame.get('empty_confirmed')) is not bool:
                raise ValueError('empty_confirmed must be true or false')
            validate_boxes(frame.get('boxes'))
            if frame['status'] == 'approved' and (not frame['boxes']) and (not frame['empty_confirmed']):
                raise ValueError('An approved empty frame requires explicit confirmation')
            if frame['status'] == 'auto_removed' and (frame['boxes'] or frame['empty_confirmed']):
                raise ValueError('Auto removed frames must have no boxes and are not confirmed negatives')
        return data

    @staticmethod
    def _complete(data):
        if data['status'] != 'complete':
            raise ValueError('Extraction is still running; wait until it is complete before reviewing or exporting')

    @staticmethod
    def _frame(data, frame_id):
        for frame in data['frames']:
            if frame['id'] == frame_id:
                return frame
        raise KeyError('Unknown frame ID')

    def image_path(self, frame_id):
        return self._path(self._frame(self._manifest_snapshot(clone=False), frame_id)['image'])

    def update_frame(self, frame_id, payload):
        if not isinstance(payload, dict) or type(payload.get('revision')) is not int or payload['revision'] < 0:
            raise ValueError('A non-negative integer revision is required')
        if payload.get('status') not in MANUAL_STATUSES or type(payload.get('empty_confirmed', False)) is not bool:
            raise ValueError('Expected pending, approved, or rejected status and a boolean empty_confirmed')
        boxes = validate_boxes(payload.get('boxes'))
        empty = payload.get('empty_confirmed', False) and (not boxes)
        if payload['status'] == 'approved' and (not boxes) and (not empty):
            raise ValueError('Explicitly confirm an empty frame before approving it as a negative example')
        with self._locked():
            data, frame = self._writable_frame(frame_id)
            if payload['revision'] != frame['revision']:
                raise RevisionConflict('This frame changed in another tab. Reload it before saving your correction.')
            frame.update(boxes=boxes, status=payload['status'], empty_confirmed=bool(empty), revision=frame['revision'] + 1)
            label = self._path(frame['label'])
            if not label.parent.is_dir():
                raise ValueError("The run's draft-labels directory is missing")
            old_label = label.read_bytes() if label.exists() else None
            atomic_write(label, yolo_text(boxes).encode('utf-8'))
            try:
                signature = atomic_write(self._path('manifest.json'), (json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))
                self._cache_committed_manifest(data, signature)
            except Exception:
                if old_label is None:
                    label.unlink(missing_ok=True)
                else:
                    atomic_write(label, old_label)
                raise
            return copy.deepcopy(frame)

    def reject_auto_removed(self):
        with self._locked():
            snapshot = self._manifest_snapshot(clone=False)
            self._complete(snapshot)
            frames = []
            count = 0
            for frame in snapshot['frames']:
                if frame['status'] == 'auto_removed':
                    frames.append({**frame, 'status': 'rejected', 'revision': frame['revision'] + 1})
                    count += 1
                else:
                    frames.append(frame)
            if count:
                data = {**snapshot, 'frames': frames}
                signature = atomic_write(self._path('manifest.json'), (json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))
                self._cache_committed_manifest(data, signature)
            return {'rejected': count}

    def mark_auto_removed(self, frame_id, revision, evidence):
        if type(revision) is not int or revision < 0 or (not isinstance(evidence, dict)):
            raise ValueError('An integer revision and strict empty-scan evidence are required')
        confidence = evidence.get('confidence')
        if evidence.get('decision') != 'empty' or evidence.get('method') != 'strict-onnx-multiview-v1' or type(evidence.get('detections')) is not int or (evidence['detections'] != 0) or (type(evidence.get('temporal_frames')) is not int) or (evidence['temporal_frames'] != 2) or (type(evidence.get('passes')) is not int) or (not 3 <= evidence['passes'] <= 10000) or (type(confidence) not in (int, float)) or (not math.isfinite(confidence)) or (not 0 < confidence <= 0.1) or (not isinstance(evidence.get('model_id'), str)) or (not 1 <= len(evidence['model_id']) <= 128) or (not isinstance(evidence.get('model_sha256'), str)) or (not re.fullmatch('[a-f0-9]{64}', evidence['model_sha256'])) or (not isinstance(evidence.get('checked_at'), str)) or (not 1 <= len(evidence['checked_at']) <= 128):
            raise ValueError('Strict multi-view and temporal empty-scan evidence is required')
        clean = {key: evidence[key] for key in ('decision', 'method', 'model_id', 'model_sha256', 'confidence', 'detections', 'temporal_frames', 'passes', 'checked_at')}
        with self._locked():
            data, frame = self._writable_frame(frame_id)
            if frame['status'] != 'pending' or frame['boxes'] or frame['revision'] != revision or (revision != 0) or frame['empty_confirmed']:
                return False
            frame.update(status='auto_removed', revision=revision + 1, empty_confirmed=False, empty_scan=clean)
            signature = atomic_write(self._path('manifest.json'), (json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))
            self._cache_committed_manifest(data, signature)
            return True

    def export_reviewed(self):
        with self._locked():
            data = self.manifest()
            self._complete(data)
            approved = [frame for frame in data['frames'] if frame['status'] == 'approved']
            counts = {split: sum((frame['split'] == split for frame in approved)) for split in ('train', 'val')}
            if not all(counts.values()):
                raise ValueError('Export requires at least one approved image in both train and val')
            digests = {}
            for frame in approved:
                image = self._path(frame['image'])
                if not image.is_file():
                    raise ValueError(f"Missing approved image: {frame['id']}")
                digest = hashlib.sha256(image.read_bytes()).hexdigest()
                if digest in digests and digests[digest] != frame['split']:
                    raise ValueError('Duplicate image content in both train and val; reject one copy before export')
                digests[digest] = frame['split']
            exports = self._path('exports')
            if exports.is_symlink() or (exports.exists() and (not exports.is_dir())):
                raise ValueError('exports must be an ordinary run directory')
            exports.mkdir(exist_ok=True)
            number = 1
            while True:
                destination = self._path(f"exports/reviewed-data{('' if number == 1 else '-' + str(number))}")
                try:
                    destination.mkdir()
                    break
                except FileExistsError:
                    number += 1
            try:
                for split in ('train', 'val'):
                    (destination / 'images' / split).mkdir(parents=True)
                    (destination / 'labels' / split).mkdir(parents=True)
                for frame in approved:
                    split, frame_id = (frame['split'], frame['id'])
                    shutil.copyfile(self._path(frame['image']), destination / 'images' / split / f'{frame_id}.png')
                    (destination / 'labels' / split / f'{frame_id}.txt').write_text(yolo_text(frame['boxes']), encoding='utf-8', newline='\n')
                (destination / 'dataset.yaml').write_text('path: .\ntrain: images/train\nval: images/val\nnames:\n  0: Head\n  1: Body\n', encoding='utf-8', newline='\n')
                report = {'schema_version': 1, 'source_run': str(self.root), 'source': data.get('source', {}), 'settings': data.get('settings', {}), 'counts': {status: sum((frame['status'] == status for frame in data['frames'])) for status in sorted(STATUSES)}, 'exported': counts, 'frames': approved}
                (destination / 'review-report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
            except Exception:
                if destination.is_relative_to(self.root / 'exports') and (not destination.is_symlink()):
                    shutil.rmtree(destination)
                raise
            return {'path': str(destination), **counts}

class BoardProposals:

    def __init__(self, store, board, proposer_factory=None):
        self.store, self.board = (store, board)
        self._factory = proposer_factory
        self._lock = threading.Lock()
        self._model_id = self._proposer = None

    def propose(self, frame_id, payload, *, store=None):
        store = self.store if store is None else store
        revision = payload.get('revision')
        confidence = payload.get('confidence', 0.25)
        tiles = payload.get('tiles', True)
        assist = payload.get('assist', False)
        model_id = payload.get('model_id')
        if type(revision) is not int or revision < 0:
            raise ValueError('A non-negative integer revision is required')
        if not isinstance(model_id, str) or not model_id or len(model_id) > 128:
            raise ValueError('Select a known model')
        if type(confidence) not in (int, float) or not math.isfinite(confidence) or (not 0.05 <= confidence <= 0.95):
            raise ValueError('Suggestion confidence must be in 0.05..0.95')
        if type(tiles) is not bool:
            raise ValueError('Head-detail tiles must be true or false')
        if type(assist) is not bool:
            raise ValueError('Missing-part recovery must be true or false')
        data = store.manifest()
        store._complete(data)
        frame = store._frame(data, frame_id)
        if frame['revision'] != revision:
            raise RevisionConflict('This frame changed. Reload it before requesting suggestions.')
        model = self.board.registry.resolve(model_id)
        assistance = {}
        with self._lock:
            if self._proposer is None or self._model_id != model_id:
                factory = self._factory
                if factory is None:
                    from prepare_dataset import ONNXProposer
                    factory = ONNXProposer
                new = factory(Path(model['path']), 0, confidence, tiles)
                old = self._proposer
                self._proposer, self._model_id = (new, model_id)
                if old is not None:
                    session = getattr(getattr(old, 'model', None), 'session', None)
                    if session is not None and hasattr(session, 'close'):
                        session.close()
            self._proposer.confidence, self._proposer.tiles = (confidence, tiles)
            import cv2
            pixels = cv2.imread(str(store.image_path(frame_id)))
            if pixels is None or pixels.shape[:2] != (448, 640):
                raise ValueError('Could not read the 640x448 review image')
            boxes = validate_boxes(self._proposer(pixels))
            if assist:
                from annotation_assist import assist_proposals
                assistance = assist_proposals(self._proposer, pixels, boxes, data, frame)
                boxes = validate_boxes(assistance.pop('boxes'))
        if store._frame(store.manifest(), frame_id)['revision'] != revision:
            raise RevisionConflict('This frame changed while suggestions were running. Request them again.')
        return {'frame_id': frame_id, 'revision': revision, 'model_id': model_id, 'boxes': boxes, **assistance}

def make_server(store, port=8765, token=None, *, workbench=None, proposals=None, recording_library=None, empty_scans=None):
    token = token or secrets.token_urlsafe(32)
    workbench_lock = threading.Lock()
    recordings_lock = threading.Lock()
    stores_lock = threading.Lock()
    recording_stores = {store.root: store}

    def get_workbench():
        nonlocal workbench
        with workbench_lock:
            if workbench is None:
                from training_board import TrainingBoard
                workbench = TrainingBoard(PROJECT)
            return workbench

    def get_proposals():
        nonlocal proposals
        board = get_workbench()
        with workbench_lock:
            if proposals is None:
                proposals = BoardProposals(store, board)
            return proposals

    def get_recordings():
        nonlocal recording_library
        with recordings_lock:
            if recording_library is None:
                from recording_library import RecordingLibrary
                recording_library = RecordingLibrary(PROJECT, initial_run=store.root, registry=get_workbench().registry)
            return recording_library

    def get_empty_scans():
        nonlocal empty_scans
        board = get_workbench()
        with workbench_lock:
            if empty_scans is None:
                from empty_scan import EmptyScanManager
                empty_scans = EmptyScanManager(PROJECT, registry=getattr(board, 'registry', None))
            return empty_scans

    def selected_id(payload):
        value = payload.get('model_id')
        if not isinstance(value, str) or not value or len(value) > 128:
            raise ValueError('Select a known model')
        return value

    class Handler(BaseHTTPRequestHandler):
        server_version = 'LocalReview'

        def log_message(self, *args):
            pass

        def _send(self, status, body, content_type='application/json; charset=utf-8'):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            try:
                self.end_headers()
                self.wfile.write(body)
            except ConnectionError:
                self.close_connection = True

        def _error(self, status, message):
            self.close_connection = True
            self._send(status, {'error': message})

        def _download(self, path, filename):
            with path.open('rb') as stream:
                self.send_response(200)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Content-Length', str(os.fstat(stream.fileno()).st_size))
                self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Referrer-Policy', 'no-referrer')
                try:
                    self.end_headers()
                    shutil.copyfileobj(stream, self.wfile, length=1024 * 1024)
                except ConnectionError:
                    self.close_connection = True

        def _route(self):
            parsed = urlsplit(self.path)
            if parsed.scheme or parsed.netloc or '%' in parsed.path or ('\\' in parsed.path):
                raise ValueError('Invalid local route')
            host = f'127.0.0.1:{self.server.server_port}'
            if self.headers.get_all('Host') != [host]:
                return (None, self._error(403, 'Host is not this local review server'))
            origins = self.headers.get_all('Origin', [])
            if origins and origins != [f'http://{host}']:
                return (None, self._error(403, 'Origin is not this local review server'))
            if parsed.path == '/' and self.command == 'GET':
                query = parse_qs(parsed.query, keep_blank_values=True)
                supplied = query.get('token', [''])
                authorized = set(query) == {'token'} and len(supplied) == 1 and hmac.compare_digest(supplied[0], token)
            else:
                supplied = self.headers.get_all('X-Review-Token', [])
                authorized = not parsed.query and len(supplied) == 1 and hmac.compare_digest(supplied[0], token)
            if not authorized:
                return (None, self._error(403, 'A valid local review token is required'))
            return (parsed.path, None)

        def _store(self):
            supplied = self.headers.get_all('X-Review-Run', [])
            if not supplied:
                return store
            if len(supplied) != 1 or not re.fullmatch('run-[a-f0-9]{16,64}', supplied[0]):
                raise ValueError('Choose one known recording ID')
            root = Path(get_recordings().resolve(supplied[0])).resolve()
            with stores_lock:
                if root not in recording_stores:
                    recording_stores[root] = ReviewStore(root)
                return recording_stores[root]

        def do_GET(self):
            try:
                route, _ = self._route()
                if route is None:
                    return
                if route == '/':
                    return self._send(200, (PROJECT / 'review_dataset.html').read_bytes(), 'text/html; charset=utf-8')
                if route == '/api/manifest':
                    return self._send(200, self._store().manifest())
                if route == '/api/recordings':
                    return self._send(200, get_recordings().snapshot())
                if route == '/api/workbench':
                    return self._send(200, get_workbench().snapshot())
                download = re.fullmatch(r'/api/models/([a-z0-9-]+)/download/(onnx|pt)', route)
                if download:
                    model_id, format = download.groups()
                    path = get_workbench().registry.artifact_path(model_id, format)
                    return self._download(path, f'{model_id}.{format}')
                if route == '/api/empty-scan':
                    return self._send(200, get_empty_scans().snapshot(self._store()))
                if route.startswith('/image/') and FRAME_ID.fullmatch(route[7:]):
                    return self._send(200, self._store().image_path(route[7:]).read_bytes(), 'image/png')
                self._error(404, 'Unknown review route')
            except (ValueError, json.JSONDecodeError) as exc:
                self._error(400, str(exc))
            except (KeyError, FileNotFoundError):
                self._error(404, 'Unknown frame or missing run file')
            except OSError:
                self._error(500, 'Could not read the local run file')
            except RuntimeError as exc:
                self._error(500, str(exc))

        def do_POST(self):
            try:
                route, _ = self._route()
                if route is None:
                    return
                if self.headers.get('Transfer-Encoding') or len(self.headers.get_all('Content-Length', [])) != 1:
                    return self._error(400, 'A single Content-Length is required')
                length = int(self.headers['Content-Length'])
                if length < 0 or length > MAX_BODY:
                    return self._error(413, 'Review request exceeds the 1 MB limit')
                if self.headers.get('Content-Type', '').split(';')[0].strip().lower() != 'application/json':
                    return self._error(415, 'Expected application/json')
                payload = json.loads(self.rfile.read(length).decode('utf-8'))
                if not isinstance(payload, dict):
                    raise ValueError('Expected a JSON object')
                if route == '/api/recordings/browse':
                    return self._send(200, get_recordings().browse())
                if route == '/api/recordings/prepare':
                    return self._send(200, get_recordings().start_preparation(payload))
                if route == '/api/empty-scan/start':
                    return self._send(200, get_empty_scans().start(payload, self._store()))
                if route == '/api/empty-scan/cancel':
                    if payload:
                        raise ValueError('Cancel does not accept scan options')
                    return self._send(200, get_empty_scans().cancel(self._store()))
                if route == '/api/export':
                    return self._send(200, self._store().export_reviewed())
                if route == '/api/frames/reject-auto-removed':
                    if payload:
                        raise ValueError('Bulk rejection does not accept frame changes')
                    return self._send(200, self._store().reject_auto_removed())
                if route == '/api/training/start':
                    return self._send(200, get_workbench().start_training(payload, self._store().export_reviewed))
                if route == '/api/models/select':
                    return self._send(200, get_workbench().select_model(selected_id(payload)))
                prefix = '/api/frame/'
                if route.startswith(prefix) and route.endswith('/propose'):
                    frame_id = route[len(prefix):-len('/propose')]
                    if FRAME_ID.fullmatch(frame_id):
                        return self._send(200, get_proposals().propose(frame_id, payload, store=self._store()))
                if route.startswith(prefix) and FRAME_ID.fullmatch(route[len(prefix):]):
                    return self._send(200, self._store().update_frame(route[len(prefix):], payload))
                self._error(404, 'Unknown review route')
            except RevisionConflict as exc:
                self._error(409, str(exc))
            except (ValueError, UnicodeError) as exc:
                self._error(400, str(exc))
            except (KeyError, FileNotFoundError):
                self._error(404, 'Unknown frame or missing run file')
            except OSError:
                self._error(500, 'Could not save the local review file')
            except RuntimeError as exc:
                self._error(500, str(exc))

        def do_OPTIONS(self):
            self._error(405, 'Cross-origin review access is disabled')
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    server.review_token = token
    return server

class EmptyReviewStore(ReviewStore):

    def __init__(self, project):
        self.root = Path(project).resolve() / 'runtime-data' / 'review-empty'

    def manifest(self):
        return dict(schema_version=1, status='complete', source={}, settings={}, frames=[])

    def image_path(self, frame_id):
        raise KeyError('Open a recording before loading frames')

    def export_reviewed(self):
        raise ValueError('Open a recording with approved frames before exporting')

    def update_frame(self, *args, **kwargs):
        raise ValueError('Open a recording before editing frames')

    def reject_auto_removed(self):
        raise ValueError('Open a recording before rejecting frames')

def startup_store(run_dir, *, project=PROJECT):
    requested = Path(run_dir).expanduser().resolve()
    if requested.exists():
        return ReviewStore(requested)
    project = Path(project).resolve()
    parent = project / 'dataset-runs'
    if parent.is_dir() and (not parent.is_symlink()) and (parent.resolve().parent == project):
        for candidate in sorted(parent.iterdir()):
            if candidate.is_dir() and (not candidate.is_symlink()) and (candidate.resolve().parent == parent):
                try:
                    return ReviewStore(candidate)
                except (ValueError, OSError):
                    continue
    return EmptyReviewStore(project)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path, nargs='?', default=PROJECT / 'dataset-runs' / 'recording-data', help='Local draft run containing manifest.json')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--export', action='store_true', help='Export only previously approved frames, then exit; does not train')
    parser.add_argument('--open-browser', action='store_true')
    args = parser.parse_args(argv)
    try:
        store = ReviewStore(args.run_dir) if args.export else startup_store(args.run_dir)
        if args.export:
            print(json.dumps(store.export_reviewed(), indent=2))
            return 0
        if not 1 <= args.port <= 65535:
            raise ValueError('--port must be in 1..65535')
        server = make_server(store, args.port)
        print(f'Review: http://127.0.0.1:{server.server_port}/?token={server.review_token}', flush=True)
        if args.open_browser:
            import webbrowser
            webbrowser.open(f'http://127.0.0.1:{server.server_port}/?token={server.review_token}')
        print('Press Ctrl+C to stop.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0
    except (ValueError, OSError) as exc:
        print(f'Review error: {exc}', file=sys.stderr)
        return 2
if __name__ == '__main__':
    raise SystemExit(main())
