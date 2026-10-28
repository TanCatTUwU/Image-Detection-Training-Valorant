from __future__ import annotations
import contextlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
import time
from assistant_api import AssistantUnauthorized, _recording_id
from assistant_loop import REASONS
from model_registry import file_lock

SESSION_ID = re.compile(r'session-[a-f0-9]{24}\Z')
MODEL = 'gemini-3.8-flash-low'
MESSAGES = dict(starting='Opening the approved Antigravity session.', running='Antigravity is reviewing pending frames.', paused='Paused. Resume keeps the same recording authorization.', restarting='Reopening Antigravity within the approved session.', server_unavailable='Waiting for the local board to return.', quota_exhausted='Provider usage limit reached. Resume after quota is available.', access_unavailable='Assistant access ended. Check sign-in and recording authorization.', provider_unavailable='Antigravity stopped. Check sign-in, model and image access.', invalid_output='Antigravity returned invalid review output. Human action is required.', stopped='The assistant worker stopped.', disconnected='Assistant disconnected. Recording access was revoked.', expired='The recording authorization expired.')
TERMINAL = frozenset(('quota_exhausted', 'access_unavailable', 'provider_unavailable', 'invalid_output', 'stopped', 'disconnected', 'expired'))
CONSOLE_COMMAND = 'cmd.exe /d /v:off /s /c ""%TRAINING_BOARD_PYTHON%" "%TRAINING_BOARD_SERVICE%" --connection-file "%TRAINING_BOARD_CONNECTION%""'


def native_console():
    if os.name != 'nt':
        return None
    import ctypes
    system = ctypes.WinDLL('kernel32', use_last_error=True)
    system.GetSystemDirectoryW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint]
    system.GetSystemDirectoryW.restype = ctypes.c_uint
    buffer = ctypes.create_unicode_buffer(32768)
    length = system.GetSystemDirectoryW(buffer, len(buffer))
    if not 0 < length < len(buffer):
        return None
    path = Path(buffer.value) / 'cmd.exe'
    return str(path.resolve()) if path.is_file() else None


def console_launch(python, service, connection_file):
    environment = os.environ.copy()
    for key, value in (('TRAINING_BOARD_PYTHON', python), ('TRAINING_BOARD_SERVICE', service), ('TRAINING_BOARD_CONNECTION', connection_file)):
        path = Path(value)
        if not path.is_absolute() or path.is_symlink() or not path.is_file() or any(ord(character) < 32 or character == '"' for character in str(path)):
            raise ValueError('Assistant console paths must be existing ordinary absolute files')
        environment[key] = str(path.resolve(strict=True))
    return CONSOLE_COMMAND, environment


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError('Assistant session state must be an ordinary bounded local file')
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError('Invalid assistant session state')
    return value


def write_json(path, value):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Assistant session state must stay inside its private folder')
    temporary = path.with_name('session-write-' + secrets.token_hex(8) + '.tmp')
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class AssistantLauncher:
    def __init__(self, project, hub, api_base, *, spawn=None, clock=None, python=None, powershell=None, console=None):
        from assistant_bridge import AssistantClient
        AssistantClient(api_base, 'assistant-' + 'a' * 43)
        self.project = Path(project).resolve(strict=True)
        self.hub = hub
        self.api_base = api_base.rstrip('/')
        self.spawn = spawn or subprocess.Popen
        self.clock = clock or time.time
        executable = Path(python or sys.executable).resolve()
        console_python = executable.with_name('python.exe')
        if executable.name.lower() == 'pythonw.exe' and console_python.is_file():
            executable = console_python
        self.python = str(executable)
        command_console = console or powershell or native_console()
        self.console = str(Path(command_console).resolve()) if command_console else None
        self.service = Path(__file__).resolve().with_name('assistant_service.py')
        self.runtime = self.project / 'runtime-data'
        self.path = self.runtime / 'assistant-launcher.json'
        self._lock = threading.RLock()
        self._processes = {}
        self._sessions = {}

    def _folder(self, path):
        path = Path(path)
        if not path.is_relative_to(self.project) or path.is_symlink() or not path.resolve().is_relative_to(self.project):
            raise ValueError('Assistant sessions must stay inside the project runtime-data folder')
        if path != self.runtime:
            self._folder(path.parent)
        path.mkdir(exist_ok=True)
        return path

    @contextlib.contextmanager
    def _locked(self):
        with self._lock:
            self._folder(self.runtime)
            lock = self.runtime / 'assistant-launcher.lock'
            if lock.is_symlink():
                raise ValueError('Invalid assistant session lock')
            with file_lock(lock, timeout=5):
                state = read_json(self.path, dict(schema_version=1, sessions={}))
                if state.get('schema_version') != 1 or not isinstance(state.get('sessions'), dict):
                    raise ValueError('Invalid assistant launcher state')
                self._sessions = state['sessions']
                for recording, entry in self._sessions.items():
                    if not isinstance(entry, dict) or entry.get('recording_id') != recording or not isinstance(entry.get('session_id'), str) or not SESSION_ID.fullmatch(entry['session_id']) or type(entry.get('active')) is not bool or entry.get('phase') not in MESSAGES:
                        raise ValueError('Invalid assistant launcher session')
                yield

    def _save(self):
        write_json(self.path, dict(schema_version=1, sessions=self._sessions))

    def _config_path(self, entry):
        return self._folder(self.runtime / 'assistant-sessions' / entry['session_id']) / 'connection.json'

    def _valid(self, entry):
        try:
            config = read_json(self._config_path(entry))
            if config is None or config.get('session_id') != entry['session_id'] or config.get('recording_id') != entry['recording_id'] or config.get('connection_id') != entry['connection_id']:
                return False
            cid = self.hub.authenticate(config.get('headers', {}).get('X-Assistant-Token'), entry['recording_id'])
            connection = self.hub._connection(cid)
            return cid == entry['connection_id'] and connection['provider'] == 'antigravity'
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def _set(self, entry, phase, *, active=None, reason=None):
        entry.update(phase=phase, message=MESSAGES[phase], reason=reason or (None if phase in ('starting', 'running') else phase))
        if active is not None:
            entry['active'] = active

    def _processing_paused(self, entry, paused):
        setter = getattr(self.hub, 'set_processing_paused', None)
        if callable(setter):
            try:
                setter(entry['connection_id'], paused)
            except AssistantUnauthorized:
                pass

    def _worker_status(self, entry):
        status = read_json(self._config_path(entry).parent / 'worker-status.json', {})
        if status.get('session_id') != entry['session_id']:
            return {}
        launch_id = entry.get('launch_id')
        if launch_id is not None and status.get('launch_id') != launch_id:
            return {}
        if entry.get('ignore_status') and status.get('launch_id') == entry.get('ignored_launch_id'):
            return {}
        return status

    def _reconcile(self, entry, status=None):
        status = self._worker_status(entry) if status is None else status
        changed = False
        healthy = status.get('healthy_progress_at')
        if type(healthy) in (int, float) and math.isfinite(healthy) and healthy > entry.get('healthy_progress_at', 0):
            entry.update(healthy_progress_at=healthy, launch_attempts=1, next_launch_at=self.clock() + 1)
            changed = True
        state = status.get('state')
        accepts = entry['phase'] not in ('paused', 'disconnected', 'expired') and isinstance(state, str) and state in MESSAGES and state != 'paused' and (entry['active'] or entry['phase'] == state)
        if accepts:
            reason = status.get('reason', state)
            reason = reason if isinstance(reason, str) and reason in REASONS.keys() | MESSAGES.keys() else None
            attempts = status.get('retry_attempts', entry.get('retry_attempts', 0))
            if type(attempts) is int and 0 <= attempts <= 5:
                if entry.get('retry_attempts') != attempts:
                    entry['retry_attempts'] = attempts
                    changed = True
            if entry.get('reason') != reason:
                entry['reason'] = reason
                changed = True
        if accepts and state in TERMINAL and (entry['active'] or entry['phase'] != state):
            self._processing_paused(entry, True)
            self._set(entry, state, active=False, reason=entry.get('reason'))
            changed = True
        elif accepts and entry['active'] and state not in TERMINAL and state != entry['phase']:
            self._set(entry, state, reason=entry.get('reason'))
            changed = True
        if changed:
            self._save()
        return status

    def _expire(self, entry):
        phase = 'expired'
        try:
            connection = self.hub._registry['connections'].get(entry['connection_id'])
            if connection is None or connection.get('revoked'):
                phase = 'disconnected'
        except (KeyError, TypeError):
            phase = 'access_unavailable'
        self._processing_paused(entry, True)
        self._set(entry, phase, active=False)
        self._save()

    def _entry(self, connection_id):
        return next((entry for entry in self._sessions.values() if entry['connection_id'] == connection_id), None)

    def _public(self, entry):
        valid = self._valid(entry)
        reason = entry.get('reason', entry['phase'])
        reason = reason if isinstance(reason, str) and reason in REASONS.keys() | MESSAGES.keys() else None
        result = {key: entry[key] for key in ('session_id', 'connection_id', 'active', 'phase')}
        message = MESSAGES[entry['phase']]
        if reason in REASONS:
            message = REASONS[reason]
            if entry['phase'] == 'provider_unavailable' and entry.get('retry_attempts') == 5:
                message += ' Five automatic retries were exhausted. Resume to retry pending frames.'
        attempts = entry.get('retry_attempts', 0)
        worker_attempts = entry.get('launch_attempts', 0)
        result.update(message=message, authorized=valid, resumable=bool(valid and not entry['active']), reason=reason, retry_attempts=attempts if type(attempts) is int and 0 <= attempts <= 5 else 0, retry_limit=5, worker_attempts=max(worker_attempts - 1, 0) if type(worker_attempts) is int and 0 <= worker_attempts <= 6 else 0, worker_retry_limit=5, last_progress_at=entry.get('healthy_progress_at'))
        return result

    def _control(self, entry, valid):
        return dict(authorized=bool(valid), active=bool(valid and entry['active']), session_id=entry['session_id'] if entry else None, launch_id=entry.get('launch_id') if entry else None, model=MODEL)

    def _snapshot(self, store):
        entry = self._sessions.get(_recording_id(store.root))
        if entry and not self._valid(entry) and entry['phase'] not in ('disconnected', 'expired'):
            self._expire(entry)
        return dict(available=bool(self.console and Path(self.console).is_file() and Path(self.python).is_file() and self.service.is_file()), session=self._public(entry) if entry else None)

    def snapshot(self, store):
        with self._locked():
            entry = self._sessions.get(_recording_id(store.root))
            if entry:
                self._reconcile(entry)
            if entry and entry['active'] and self._valid(entry):
                self._spawn(entry)
            return self._snapshot(store)

    def _legacy(self, store):
        path = self.project / 'assistant-connection.json'
        try:
            config = read_json(path)
            recording = _recording_id(store.root)
            if config is None or config.get('recording_id') != recording:
                return None
            if not isinstance(config.get('headers'), dict):
                return None
            token = config.get('headers', {}).get('X-Assistant-Token')
            cid = self.hub.authenticate(token, recording)
            connection = self.hub._connection(cid)
            if connection['provider'] != 'antigravity':
                return None
            return connection, token
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _spawn(self, entry):
        status = self._reconcile(entry)
        if not entry['active']:
            return
        process = self._processes.get(entry['session_id'])
        if process is not None and process.poll() is None:
            return
        folder = self._config_path(entry).parent
        lock = folder / 'worker.lock'
        if lock.is_symlink():
            raise ValueError('Invalid assistant worker lock')
        try:
            with file_lock(lock, timeout=.01):
                pass
        except ValueError:
            return
        cache_root = self.runtime / 'assistant-cache'
        cache = cache_root / entry['connection_id']
        bridge_lock = cache / 'bridge.lock'
        if any(path.is_symlink() or not path.resolve().is_relative_to(self.runtime) for path in (cache_root, cache, bridge_lock)):
            raise ValueError('Assistant bridge locks must stay inside their recording cache')
        if bridge_lock.exists():
            try:
                with file_lock(bridge_lock, timeout=.01):
                    pass
            except ValueError:
                if entry['phase'] != 'running':
                    self._set(entry, 'running')
                    self._save()
                return
        finished = process is not None or status.get('session_id') == entry['session_id'] and status.get('state') in TERMINAL | {'server_unavailable'}
        heartbeat = status.get('heartbeat_at')
        if not finished and status.get('session_id') == entry['session_id'] and type(heartbeat) in (int, float) and 0 <= self.clock() - heartbeat < 30:
            return
        launched = entry.get('launched_at')
        if not finished and type(launched) in (int, float) and 0 <= self.clock() - launched < 15:
            return
        attempts = entry.get('launch_attempts', 0)
        if type(attempts) is not int or not 0 <= attempts <= 6:
            raise ValueError('Invalid assistant worker launch budget')
        if attempts >= 6:
            self._processing_paused(entry, True)
            self._set(entry, 'provider_unavailable', active=False, reason='worker_retry_exhausted')
            self._save()
            return
        next_launch = entry.get('next_launch_at', 0)
        if type(next_launch) not in (int, float) or not math.isfinite(next_launch):
            raise ValueError('Invalid assistant worker restart delay')
        if self.clock() < next_launch:
            if finished and entry['phase'] != 'restarting':
                self._set(entry, 'restarting', reason='worker_exited')
                self._save()
            return
        launch_id = 'worker-' + secrets.token_hex(12)
        config_path = self._config_path(entry)
        config = read_json(config_path)
        write_json(config_path, dict(config, launch_id=launch_id))
        arguments, environment = console_launch(self.python, self.service, self._config_path(entry))
        self._set(entry, 'starting', active=True)
        entry.update(launched_at=self.clock(), launch_id=launch_id, launch_attempts=attempts + 1, next_launch_at=self.clock() + min(2 ** attempts, 8), ignore_status=False)
        self._save()
        try:
            self._processes[entry['session_id']] = self.spawn(arguments, executable=self.console, env=environment, cwd=str(self.project), stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, 'CREATE_NEW_CONSOLE', 0), shell=False)
        except (OSError, ValueError, subprocess.SubprocessError):
            self._set(entry, 'restarting', reason='worker_launch_failed')
            self._save()
            raise ValueError('Could not open the assistant command console') from None

    def _public_connection(self, entry):
        with getattr(self.hub, '_lock', contextlib.nullcontext()):
            return self.hub._public(self.hub._connection(entry['connection_id']))

    def start(self, store, payload):
        if not isinstance(payload, dict) or payload:
            raise ValueError('Start accepts an empty object for the selected recording')
        with self._locked():
            if not self._snapshot(store)['available']:
                raise ValueError('Python and the Windows command console are required for automatic assistant sessions')
            recording = _recording_id(store.root)
            entry = self._sessions.get(recording)
            if entry:
                self._reconcile(entry)
            if entry is None or not self._valid(entry):
                adopted = self._legacy(store)
                if adopted is None:
                    created = self.hub.connect(store, dict(provider='antigravity', name='Antigravity board session'))
                    cid, token = created['connection']['id'], created['token']
                else:
                    connection, token = adopted
                    cid = connection['id']
                entry = dict(session_id='session-' + secrets.token_hex(12), connection_id=cid, recording_id=recording, active=True, phase='starting', message=MESSAGES['starting'])
                write_json(self._config_path(entry), dict(api_base=self.api_base, recording_id=recording, session_id=entry['session_id'], connection_id=cid, headers={'X-Assistant-Token': token}, retry_epoch=0))
                self._sessions[recording] = entry
                entry.update(launch_attempts=0, next_launch_at=0, retry_attempts=0)
            else:
                if not entry['active'] and entry['phase'] != 'paused':
                    config_path = self._config_path(entry)
                    config = read_json(config_path)
                    epoch = config.get('retry_epoch', 0)
                    if type(epoch) is not int or epoch < 0:
                        raise ValueError('Invalid assistant retry generation')
                    write_json(config_path, dict(config, retry_epoch=epoch + 1))
                    entry.update(launch_attempts=0, next_launch_at=0, retry_attempts=0)
                self._set(entry, 'running', active=True)
            status = self._worker_status(entry)
            if status.get('state') in TERMINAL:
                entry.update(ignore_status=True, ignored_launch_id=entry.get('launch_id'))
            self._processing_paused(entry, False)
            self._save()
            self._spawn(entry)
            return dict(connection=self._public_connection(entry), launcher=self._snapshot(store))

    def pause(self, store, payload):
        if not isinstance(payload, dict) or set(payload) != {'connection_id'} or not isinstance(payload['connection_id'], str):
            raise ValueError('Pause requires one connection_id')
        with self._locked():
            entry = self._sessions.get(_recording_id(store.root))
            if entry is None or entry['connection_id'] != payload['connection_id'] or not self._valid(entry):
                raise AssistantUnauthorized('Unknown approved assistant session in this recording')
            self._processing_paused(entry, True)
            self._set(entry, 'paused', active=False)
            self._save()
            return dict(connection=self._public_connection(entry), launcher=self._snapshot(store))

    def control(self, connection_id):
        with self._locked():
            entry = self._entry(connection_id)
            if entry:
                self._reconcile(entry)
            valid = entry is not None and self._valid(entry)
            if entry and not valid and entry['phase'] not in ('disconnected', 'expired'):
                self._expire(entry)
            return self._control(entry, valid)

    def report(self, connection_id, payload):
        if not isinstance(payload, dict) or set(payload) - {'session_id', 'launch_id', 'state', 'message'} or not {'session_id', 'launch_id', 'state'} <= set(payload) or not isinstance(payload.get('launch_id'), str) or not re.fullmatch(r'worker-[a-f0-9]{24}', payload['launch_id']) or not isinstance(payload.get('state'), str) or payload['state'] not in MESSAGES or ('message' in payload and (not isinstance(payload['message'], str) or len(payload['message']) > 500)):
            raise ValueError('Expected a session_id, worker launch_id, known state and optional bounded message')
        with self._locked():
            entry = self._entry(connection_id)
            if entry is None or entry['session_id'] != payload['session_id']:
                raise ValueError('The assistant report belongs to another logical session')
            valid = self._valid(entry)
            if not valid:
                self._expire(entry)
            elif payload['launch_id'] != entry.get('launch_id') or entry.get('ignore_status') and payload['launch_id'] == entry.get('ignored_launch_id'):
                return self._control(entry, valid)
            elif payload['state'] in TERMINAL and entry['phase'] != 'paused':
                self._processing_paused(entry, True)
                self._set(entry, payload['state'], active=False)
            elif entry['active']:
                self._set(entry, payload['state'])
            self._reconcile(entry)
            self._save()
            return self._control(entry, valid)

    def allows(self, connection_id):
        with self._locked():
            entry = self._entry(connection_id)
            if entry:
                self._reconcile(entry)
            return True if entry is None else bool(entry['active'] and self._valid(entry))

    def revoke(self, connection_id):
        with self._locked():
            entry = self._entry(connection_id)
            if entry:
                self._processing_paused(entry, True)
                self._set(entry, 'disconnected', active=False)
                self._config_path(entry).unlink(missing_ok=True)
                self._save()

    def supervise(self, stop_event, *, interval=5):
        if type(interval) not in (int, float) or not math.isfinite(interval) or not .01 <= interval <= 60:
            raise ValueError('Invalid assistant supervision interval')
        while not stop_event.is_set():
            try:
                self.resume(stop_event=stop_event)
            except (OSError, ValueError, RuntimeError):
                pass
            if stop_event.wait(interval):
                return

    def resume(self, *, stop_event=None):
        if stop_event is not None and stop_event.is_set():
            return
        with self._locked():
            for entry in self._sessions.values():
                if stop_event is not None and stop_event.is_set():
                    return
                self._reconcile(entry)
                if not entry['active']:
                    self._processing_paused(entry, True)
                    continue
                if not self._valid(entry):
                    self._expire(entry)
                    continue
                config_path = self._config_path(entry)
                config = read_json(config_path)
                if config['api_base'] != self.api_base:
                    write_json(config_path, dict(config, api_base=self.api_base))
                if stop_event is not None and stop_event.is_set():
                    return
                self._spawn(entry)
