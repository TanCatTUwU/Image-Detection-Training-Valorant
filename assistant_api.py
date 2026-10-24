from __future__ import annotations
import copy
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from assistant_progress import ProgressStore, validate_event

PROVIDERS = ('antigravity', 'custom')
CONNECTION_ID = re.compile(r'assistant-[a-f0-9]{24}\Z')
FRAME_ID = re.compile(r'frame-[0-9]{6,12}\Z')
TOKEN = re.compile(r'assistant-[A-Za-z0-9_-]{43}\Z')
TOKEN_LIFETIME = 7 * 86400


class AssistantConflict(ValueError):
    pass


class AssistantUnauthorized(ValueError):
    pass


def _text(value, maximum, label):
    if not isinstance(value, str) or not 1 <= len(value) <= maximum or value != value.strip() or any(ord(char) < 32 or 127 <= ord(char) < 160 for char in value):
        raise ValueError(f'{label} must be a nonempty single-line string of at most {maximum} characters')
    return value


def validate_assistant(value):
    if not isinstance(value, dict) or set(value) != {'provider', 'name', 'connection_id'}:
        raise ValueError('Assistant provenance requires provider, name, and connection_id')
    provider = _text(value['provider'], 32, 'Assistant provider')
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,31}', provider):
        raise ValueError('Assistant provider must use a known safe identifier')
    name = _text(value['name'], 80, 'Assistant name')
    connection_id = value['connection_id']
    if not isinstance(connection_id, str) or not CONNECTION_ID.fullmatch(connection_id):
        raise ValueError('Invalid assistant connection ID')
    return dict(provider=provider, name=name, connection_id=connection_id)


def _utc(timestamp=None):
    return datetime.fromtimestamp(time.time() if timestamp is None else timestamp, timezone.utc).isoformat(timespec='seconds')


def _recording_id(root):
    canonical = os.path.normcase(str(Path(root).resolve()))
    return 'run-' + hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:24]


def _atomic_json(path, value):
    temporary = path.with_name(f'assistant-tmp-{secrets.token_hex(8)}.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write((json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load(path, default):
    if not path.exists():
        return default
    if path.is_symlink() or path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError('Assistant state must be an ordinary bounded local file')
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict) or value.get('schema_version') != 1:
        raise ValueError('Invalid assistant state')
    return value


def _submission(payload):
    if not isinstance(payload, dict) or set(payload) - {'frame_id', 'revision', 'boxes', 'request_id'} or not {'frame_id', 'revision', 'boxes'} <= set(payload):
        raise ValueError('Expected frame_id, revision, boxes, and optional request_id')
    frame_id, revision, boxes = payload['frame_id'], payload['revision'], payload['boxes']
    if not isinstance(frame_id, str) or not FRAME_ID.fullmatch(frame_id) or type(revision) is not int or revision < 0:
        raise ValueError('Expected a valid frame ID and non-negative integer revision')
    if not isinstance(boxes, list) or len(boxes) > 128:
        raise ValueError('Assistant boxes must be a list with at most 128 rectangles')
    clean = []
    for box in boxes:
        if not isinstance(box, dict) or set(box) - {'class_id', 'x1', 'y1', 'x2', 'y2', 'score'} or type(box.get('class_id')) is not int or box['class_id'] not in (0, 1):
            raise ValueError('Assistant boxes require class_id 0 or 1 and normalized coordinates only')
        values = [box.get(key) for key in ('x1', 'y1', 'x2', 'y2')]
        if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 for value in values) or values[2] <= values[0] or values[3] <= values[1]:
            raise ValueError('Assistant boxes require finite positive rectangles normalized to 0..1')
        if 'score' in box and (type(box['score']) not in (int, float) or not math.isfinite(box['score']) or not 0 <= box['score'] <= 1):
            raise ValueError('Assistant score must be finite and in 0..1')
        clean.append(dict(box))
    normalized = dict(frame_id=frame_id, revision=revision, boxes=clean)
    fingerprint = hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
    request_id = payload.get('request_id', fingerprint)
    if not isinstance(request_id, str) or not re.fullmatch(r'[A-Za-z0-9._:-]{1,128}', request_id):
        raise ValueError('request_id must contain 1..128 letters, digits, dots, underscores, colons, or hyphens')
    return normalized, fingerprint, request_id


def _usage(value):
    if not isinstance(value, dict) or set(value) - {'windows', 'source', 'updated_at', 'error', 'loading', 'stale'}:
        raise ValueError('Expected a bounded usage snapshot')
    windows = value.get('windows', [])
    if not isinstance(windows, list) or len(windows) > 16:
        raise ValueError('Usage windows must be a list of at most 16 entries')
    clean = []
    for window in windows:
        if not isinstance(window, dict) or set(window) - {'id', 'label', 'used_percent', 'reset_at', 'window_minutes'}:
            raise ValueError('Invalid usage window')
        used, minutes, reset = window.get('used_percent'), window.get('window_minutes'), window.get('reset_at')
        if used is not None and (type(used) not in (int, float) or not math.isfinite(used) or not 0 <= used <= 100):
            raise ValueError('Usage percent must be null or a finite number in 0..100')
        if minutes is not None and (type(minutes) is not int or not 1 <= minutes <= 525600):
            raise ValueError('Usage window_minutes must be a positive bounded integer or null')
        if reset is not None:
            _text(reset, 64, 'Usage reset time')
        clean.append(dict(id=_text(window.get('id'), 80, 'Usage window ID'), label=_text(window.get('label'), 120, 'Usage window label'), used_percent=used, reset_at=reset, window_minutes=minutes))
    source, updated, error = value.get('source'), value.get('updated_at'), value.get('error')
    for item, maximum, label in ((source, 160, 'Usage source'), (updated, 64, 'Usage update time'), (error, 500, 'Usage error')):
        if item is not None:
            _text(item, maximum, label)
    for flag in ('loading', 'stale'):
        if flag in value and type(value[flag]) is not bool:
            raise ValueError(f'Usage {flag} must be boolean')
    result = dict(windows=clean, source=source, updated_at=updated, error=error)
    for flag in ('loading', 'stale'):
        if flag in value:
            result[flag] = value[flag]
    return result


class AssistantHub:
    def __init__(self, project, store_factory):
        self.project = Path(project).resolve()
        self._store_factory = store_factory
        self._lock = threading.RLock()
        self._stores = {}
        self._monitor = None
        self._monitor_error = None
        self._progress = ProgressStore(self.project)
        self._state_cache = {}
        self._migration_signatures = {}
        self._path = self.project / 'runtime-data' / 'assistant-connections.json'
        self._registry = _load(self._path, dict(schema_version=1, connections={}))
        if not isinstance(self._registry.get('connections'), dict):
            raise ValueError('Invalid assistant connection registry')

    def _save(self):
        parent = self._path.parent
        if parent.is_symlink() or parent.resolve().parent != self.project:
            raise ValueError('Assistant runtime-data must stay inside the project')
        parent.mkdir(exist_ok=True)
        _atomic_json(self._path, self._registry)

    def _status(self, connection):
        if connection.get('revoked'):
            return 'disconnected'
        if connection['expires_at'] <= time.time():
            return 'expired'
        if connection.get('last_seen_at') is None:
            return 'idle'
        return 'connected' if time.time() - connection['last_seen_at'] <= 90 else 'offline'

    def _valid(self, connection):
        return not connection.get('revoked') and connection['expires_at'] > time.time()

    def _seen(self, connection):
        connection.update(last_seen=_utc(), last_seen_at=time.time())

    def _store(self, connection):
        root = Path(connection['root']).resolve(strict=True)
        if _recording_id(root) != connection['recording_id']:
            raise AssistantUnauthorized('The assistant recording scope is invalid')
        if root not in self._stores:
            self._stores[root] = self._store_factory(root)
        return self._stores[root]

    def _state(self, store, *, clone=True):
        path = store._path('assistant-state.json')
        signature = self._signature(path)
        cached = self._state_cache.get(store.root)
        if cached is not None and cached[0] == signature:
            return copy.deepcopy(cached[1]) if clone else cached[1]
        state = _load(path, dict(schema_version=1, requests={}))
        if not isinstance(state.get('requests'), dict):
            raise ValueError('Invalid assistant request state')
        self._state_cache[store.root] = (signature, state)
        return copy.deepcopy(state) if clone else state

    def _save_state(self, store, state):
        _atomic_json(store._path('assistant-state.json'), state)
        self._state_cache[store.root] = (self._signature(store._path('assistant-state.json')), copy.deepcopy(state))

    @staticmethod
    def _signature(path):
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    @staticmethod
    def _manifest(store):
        return store._manifest_snapshot(clone=False) if hasattr(store, '_manifest_snapshot') else store.manifest()

    def _migrate_progress(self, store, state):
        signature = self._signature(store._path('assistant-state.json'))
        if self._migration_signatures.get(store.root, object()) == signature:
            return
        completed = {}
        for entry in state['requests'].values():
            if entry['state'] == 'applied':
                key = entry['connection_id'], entry['frame_id']
                completed[key] = max(completed.get(key, -1), entry['revision'] + int(entry.get('added', 0) > 0))
        if completed:
            with self._progress.transaction() as database:
                for (connection_id, fid), revision in completed.items():
                    self._progress.success(database, connection_id, fid, revision, _utc())
        self._migration_signatures[store.root] = signature

    def _usage(self, connection):
        if self._monitor is None and self._monitor_error is None:
            try:
                from assistant_usage import UsageMonitor
                self._monitor = UsageMonitor(self.project)
            except (ImportError, OSError, ValueError, RuntimeError):
                self._monitor_error = 'Provider usage monitoring is unavailable'
        result = dict(windows=[], source=None, updated_at=None, error=self._monitor_error)
        if self._monitor is not None:
            try:
                result = _usage(self._monitor.snapshot(connection['provider']))
            except (OSError, ValueError, RuntimeError, TypeError):
                result['error'] = 'Could not read provider usage'
        supplied = connection.get('usage')
        if not result['windows'] and supplied and time.time() - connection.get('usage_received_at', 0) < 120:
            result = copy.deepcopy(supplied)
        if not result['windows'] and not result.get('error') and not result.get('loading'):
            result['error'] = 'Provider usage is unavailable'
        return result

    def _public(self, connection, state=None, manifest=None):
        store = self._store(connection)
        state = state if state is not None else self._state(store, clone=False)
        self._migrate_progress(store, state)
        requests = state['requests'].values()
        owned = [entry for entry in requests if entry['connection_id'] == connection['id']]
        manifest = manifest if manifest is not None else self._manifest(store)
        progress = self._progress.snapshot(connection['id'], {frame['id']: frame for frame in manifest['frames']})
        return dict(id=connection['id'], name=connection['name'], provider=connection['provider'], recording_id=connection['recording_id'], status=self._status(connection), last_seen=connection.get('last_seen'), processed=progress['processed'], queued=sum(entry['state'] == 'queued' for entry in owned), last_error=connection.get('last_error'), usage=self._usage(connection), progress=progress)

    def connect(self, store, payload):
        if not isinstance(payload, dict) or set(payload) != {'provider', 'name'} or payload.get('provider') not in PROVIDERS:
            raise ValueError('Choose the antigravity or custom provider and a connection name')
        name = _text(payload['name'], 80, 'Connection name')
        if not (store.root / 'manifest.json').is_file() or not hasattr(store, '_thread_lock'):
            raise ValueError('Open a recording before connecting an assistant')
        with self._lock:
            if sum(self._valid(item) for item in self._registry['connections'].values()) >= 32:
                raise ValueError('Disconnect an assistant before creating more than 32 active connections')
            if len(self._registry['connections']) >= 256:
                raise ValueError('The local assistant connection history is full')
            token = 'assistant-' + secrets.token_urlsafe(32)
            connection_id = 'assistant-' + secrets.token_hex(12)
            connection = dict(id=connection_id, name=name, provider=payload['provider'], recording_id=_recording_id(store.root), root=str(store.root), token_hash=hashlib.sha256(token.encode('utf-8')).hexdigest(), expires_at=time.time() + TOKEN_LIFETIME, revoked=False, created_at=_utc(), last_seen=None, last_error=None)
            updated = copy.deepcopy(self._registry)
            updated['connections'][connection_id] = connection
            previous, self._registry = self._registry, updated
            try:
                self._save()
            except Exception:
                self._registry = previous
                raise
            self._stores[store.root] = store
            return dict(connection=self._public(connection), token=token, recording_id=connection['recording_id'])

    def authenticate(self, token, recording=None):
        if not isinstance(token, str) or not TOKEN.fullmatch(token):
            raise AssistantUnauthorized('A valid assistant token is required')
        digest = hashlib.sha256(token.encode('ascii')).hexdigest()
        with self._lock:
            connection = next((entry for entry in self._registry['connections'].values() if hmac.compare_digest(entry['token_hash'], digest)), None)
            if connection is None or not self._valid(connection):
                raise AssistantUnauthorized('The assistant token is invalid, expired, or disconnected')
            if recording is not None and recording != connection['recording_id']:
                raise AssistantUnauthorized('The assistant token is scoped to another recording')
            self._store(connection)
            return connection['id']

    def _connection(self, connection_id):
        connection = self._registry['connections'].get(connection_id)
        if connection is None or not self._valid(connection):
            raise AssistantUnauthorized('The assistant token is invalid, expired, or disconnected')
        return connection

    def snapshot(self, store):
        with self._lock:
            self.flush(store)
            recording_id = _recording_id(store.root)
            state = self._state(store, clone=False) if hasattr(store, '_path') else dict(requests={})
            manifest = self._manifest(store)
            signature = self._signature(store._path('manifest.json')) if hasattr(store, '_path') else None
            version = hashlib.sha256(repr(signature).encode('ascii')).hexdigest()[:24] if signature else None
            return dict(connections=[self._public(entry, state, manifest) for entry in self._registry['connections'].values() if entry['recording_id'] == recording_id], providers=list(PROVIDERS), annotation_version=version)

    def disconnect(self, store, payload):
        if not isinstance(payload, dict) or set(payload) != {'connection_id'} or not isinstance(payload['connection_id'], str):
            raise ValueError('Expected one connection_id')
        with self._lock:
            connection = self._registry['connections'].get(payload['connection_id'])
            if connection is None or connection['recording_id'] != _recording_id(store.root):
                raise KeyError('Unknown assistant connection in this recording')
            previous = connection['revoked']
            connection['revoked'] = True
            try:
                self._save()
            except Exception:
                connection['revoked'] = previous
                raise
            self.flush(store)
            return dict(connection=self._public(connection))

    def revoke_recording(self, root):
        root = Path(root).resolve()
        with self._lock:
            previous = copy.deepcopy(self._registry)
            changed = False
            for connection in self._registry['connections'].values():
                if Path(connection['root']).resolve() == root and not connection['revoked']:
                    connection['revoked'] = True
                    changed = True
            if changed:
                try:
                    self._save()
                except Exception:
                    self._registry = previous
                    raise
            self._stores.pop(root, None)

    def status(self, connection_id, payload=None):
        with self._lock:
            connection = self._connection(connection_id)
            if payload is not None:
                if not isinstance(payload, dict) or set(payload) - {'usage'}:
                    raise ValueError('Heartbeat accepts only an optional usage snapshot')
                if 'usage' in payload:
                    connection['usage'] = _usage(payload['usage'])
                    connection['usage_received_at'] = time.time()
                self._seen(connection)
                self._save()
            store = self._store(connection)
            self.flush(store)
            return dict(connection=self._public(connection), recording_status=self._manifest(store)['status'])

    def frames(self, connection_id, payload):
        if not isinstance(payload, dict) or set(payload) != {'limit', 'cursor'} or type(payload['limit']) is not int or not 1 <= payload['limit'] <= 16 or type(payload['cursor']) is not int or payload['cursor'] < 0:
            raise ValueError('Expected integer limit 1..16 and non-negative integer cursor')
        with self._lock:
            connection = self._connection(connection_id)
            store = self._store(connection)
            self.flush(store)
            manifest = self._manifest(store)
            self._migrate_progress(store, self._state(store, clone=False))
            completed = self._progress.completed(connection_id)
            frames, cursor = [], min(payload['cursor'], len(manifest['frames']))
            while cursor < len(manifest['frames']) and len(frames) < payload['limit']:
                frame = manifest['frames'][cursor]
                cursor += 1
                if frame['status'] == 'pending' and completed.get(frame['id']) != frame['revision'] and store._path(frame['image']).is_file():
                    frames.append({key: copy.deepcopy(frame.get(key)) for key in ('id', 'revision', 'width', 'height', 'status', 'boxes', 'time_seconds')})
                    frames[-1]['image_url'] = '/api/assistant/image/' + frame['id']
            self._seen(connection)
            self._save()
            return dict(frames=frames, next_cursor=cursor if cursor < len(manifest['frames']) else None, recording_status=manifest['status'])

    def image_path(self, connection_id, frame_id):
        with self._lock:
            connection = self._connection(connection_id)
            store = self._store(connection)
            with store._thread_lock:
                frame = store._frame(self._manifest(store), frame_id)
                if frame['status'] != 'pending':
                    raise AssistantConflict('This frame is no longer pending')
                return store._path(frame['image'])

    def _completion_revision(self, connection_id, frame, revision, manifest, state):
        if frame['status'] != 'pending':
            raise AssistantConflict('This frame is no longer pending')
        if frame['revision'] == revision:
            return revision
        receipts = manifest.get('assistant_receipts', {})
        for key, entry in state['requests'].items():
            if entry['connection_id'] != connection_id or entry['frame_id'] != frame['id'] or entry['revision'] != revision:
                continue
            receipt = receipts.get(key)
            if receipt and receipt.get('fingerprint') == entry['fingerprint'] and frame['revision'] == revision + int(receipt.get('added', 0) > 0):
                return frame['revision']
        raise AssistantConflict('This pending frame changed. Reload it before reporting completion.')

    def progress(self, connection_id, payload):
        event, batch_id, entries, total, error = validate_event(payload, FRAME_ID)
        with self._lock:
            connection = self._connection(connection_id)
            store = self._store(connection)
            with store._locked():
                manifest = self._manifest(store)
                frames = {frame['id']: frame for frame in manifest['frames']}
                state = self._state(store, clone=False)
                with self._progress.transaction() as database:
                    batch = self._progress.batch(database, connection_id, batch_id)
                    is_new = batch is None
                    if batch is None:
                        if event != 'start':
                            raise AssistantConflict('Start this assistant batch before reporting progress')
                        batch = dict(batch_id=batch_id, items={}, total=total or len(entries), finished=False)
                    items = batch['items']
                    now = _utc()
                    revisions = {}
                    for fid, revision in entries:
                        frame = frames.get(fid)
                        if frame is None:
                            raise KeyError('Unknown frame ID')
                        item = items.get(fid)
                        if item is not None and item['revision'] != revision:
                            raise AssistantConflict('This batch already started a different frame revision')
                        if event == 'start':
                            if frame['status'] != 'pending' or frame['revision'] != revision:
                                raise AssistantConflict('This pending frame changed. Reload it before starting work.')
                            if not store._path(frame['image']).is_file():
                                raise ValueError('The requested frame image has not been extracted')
                            if batch['finished'] and item is None:
                                raise AssistantConflict('This assistant batch already finished')
                        else:
                            if item is None:
                                raise AssistantConflict('Start this frame before reporting completion or failure')
                            if event == 'complete':
                                revisions[fid] = self._completion_revision(connection_id, frame, revision, manifest, state)
                                if item['state'] == 'failed':
                                    raise AssistantConflict('Start a new batch to retry a failed frame')
                    if event == 'start':
                        if len(set(items) | {fid for fid, _ in entries}) > 32:
                            raise ValueError('Assistant batches accept at most 32 unique frames')
                        if total is not None and total < len(set(items) | {fid for fid, _ in entries}):
                            raise ValueError('Batch total cannot be smaller than started frames')
                        if not batch['finished']:
                            batch['total'] = max(batch['total'], total or 0, len(set(items) | {fid for fid, _ in entries}))
                        for fid, revision in entries:
                            if fid not in items:
                                items[fid] = dict(revision=revision, started_at=now, state='processing')
                    elif event == 'complete':
                        for fid, _ in entries:
                            items[fid].update(state='completed', completed_revision=revisions[fid])
                            self._progress.success(database, connection_id, fid, revisions[fid], now)
                    elif event == 'fail':
                        for fid, _ in entries:
                            if items[fid]['state'] != 'completed':
                                items[fid]['state'] = 'failed'
                                self._progress.failure(database, connection_id, fid)
                        if error and any(items[fid]['state'] == 'failed' for fid, _ in entries):
                            connection['last_error'] = error
                    else:
                        batch.update(finished=True, total=len(items))
                        for fid, item in items.items():
                            if item['state'] == 'processing':
                                item['state'] = 'failed'
                                self._progress.failure(database, connection_id, fid)
                    active = self._progress.active_batch(database, connection_id)
                    activate = active is None or (event == 'start' and is_new)
                    self._progress.save_batch(database, connection_id, batch, now, activate=activate)
                self._seen(connection)
                if event == 'complete':
                    connection['last_error'] = None
                return dict(connection=self._public(connection, state, manifest))

    def _result(self, store, entry):
        result = dict(state=entry['state'], added=entry.get('added', 0), duplicate=entry.get('duplicate', 0))
        if entry['state'] == 'applied':
            frame = store._frame(self._manifest(store), entry['frame_id'])
            if frame['status'] == 'pending':
                result['frame'] = copy.deepcopy(frame)
        return result

    def _apply(self, store, key, entry):
        snapshot = self._manifest(store)
        receipt = snapshot.get('assistant_receipts', {}).get(key)
        if receipt is not None:
            if receipt['fingerprint'] != entry['fingerprint']:
                raise AssistantConflict('This assistant request already has a different payload')
            result = dict(state='applied', added=receipt['added'], duplicate=receipt['duplicate'])
            frame = store._frame(snapshot, entry['frame_id'])
            if frame['status'] == 'pending':
                result['frame'] = copy.deepcopy(frame)
        else:
            result = store.merge_assistant(entry['frame_id'], entry['revision'], entry['boxes'], key, entry['fingerprint'])
        entry.update(state='applied', added=result['added'], duplicate=result['duplicate'])
        entry.pop('boxes', None)
        with self._progress.transaction() as database:
            self._progress.success(database, entry['connection_id'], entry['frame_id'], entry['revision'] + int(result['added'] > 0), _utc())
        return result

    def submit(self, connection_id, payload):
        normalized, fingerprint, request_id = _submission(payload)
        with self._lock:
            connection = self._connection(connection_id)
            store = self._store(connection)
            self.flush(store)
            state = self._state(store)
            key = hashlib.sha256((connection_id + ':' + request_id).encode('utf-8')).hexdigest()
            entry = state['requests'].get(key)
            if entry is not None:
                if entry['fingerprint'] != fingerprint:
                    raise AssistantConflict('This request_id was already used for a different submission')
                if entry['state'] == 'failed':
                    raise AssistantConflict(entry['error'])
                return self._result(store, entry)
            manifest = self._manifest(store)
            frame = store._frame(manifest, normalized['frame_id'])
            if frame['status'] != 'pending' or frame['revision'] != normalized['revision']:
                raise AssistantConflict('This pending frame changed. Reload it before submitting boxes.')
            if not store._path(frame['image']).is_file():
                raise ValueError('The requested frame image has not been extracted')
            if len(state['requests']) >= 4096 or sum(item['state'] == 'queued' for item in state['requests'].values()) >= 256:
                raise ValueError('The recording assistant request queue is full')
            provenance = dict(provider=connection['provider'], name=connection['name'], connection_id=connection_id)
            boxes = [dict(box, assistant=provenance) for box in normalized['boxes']]
            entry = dict(connection_id=connection_id, fingerprint=fingerprint, frame_id=normalized['frame_id'], revision=normalized['revision'], boxes=boxes, state='queued', created_at=_utc())
            state['requests'][key] = entry
            self._save_state(store, state)
            if manifest['status'] == 'complete':
                try:
                    result = self._apply(store, key, entry)
                except (ValueError, KeyError) as exc:
                    entry.update(state='failed', error=str(exc))
                    entry.pop('boxes', None)
                    connection['last_error'] = str(exc)
                    self._save_state(store, state)
                    self._save()
                    raise
                self._save_state(store, state)
            else:
                result = self._result(store, entry)
            self._seen(connection)
            connection['last_error'] = None
            self._save()
            return result

    def flush(self, store):
        if not hasattr(store, '_thread_lock'):
            return
        with self._lock:
            snapshot = self._state(store, clone=False)
            if not any(entry['state'] == 'queued' for entry in snapshot['requests'].values()) or self._manifest(store)['status'] != 'complete':
                return
            state = copy.deepcopy(snapshot)
            changed = False
            for key, entry in state['requests'].items():
                if entry['state'] != 'queued':
                    continue
                connection = self._registry['connections'].get(entry['connection_id'])
                try:
                    if connection is None or not self._valid(connection) or connection['recording_id'] != _recording_id(store.root):
                        raise AssistantConflict('The queued assistant connection expired or was disconnected')
                    self._apply(store, key, entry)
                    connection['last_error'] = None
                except (ValueError, KeyError) as exc:
                    entry.update(state='failed', error=str(exc))
                    entry.pop('boxes', None)
                    if connection is not None:
                        connection['last_error'] = str(exc)
                changed = True
            if changed:
                self._save_state(store, state)
                self._save()
