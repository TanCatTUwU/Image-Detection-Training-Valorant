from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import threading
import time


def _text(value, limit=160):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError('Invalid usage text')
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError('Invalid usage text')
    return value.strip()


def _reset(value):
    value = _text(value, 64)
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})', value):
        raise ValueError('Invalid usage reset time')
    reset = datetime.fromisoformat(value.replace('Z', '+00:00'))
    return reset.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def parse_antigravity_usage(payload):
    if isinstance(payload, bytes):
        payload = payload.decode('utf-8-sig')
    if isinstance(payload, str):
        payload = json.loads(payload.lstrip('\ufeff'))
    if not isinstance(payload, dict) or payload.get('status') != 'SUCCESS':
        raise ValueError('Antigravity usage is unavailable')
    command = payload.get('command')
    if not isinstance(command, dict) or command.get('name') != 'usage':
        raise ValueError('Invalid Antigravity usage command')
    data = command.get('data')
    groups = data.get('groups') if isinstance(data, dict) else None
    if not isinstance(groups, list) or not 1 <= len(groups) <= 32:
        raise ValueError('Missing Antigravity quota groups')
    windows = []
    identifiers = set()
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError('Invalid Antigravity quota group')
        name = _text(group.get('name'))
        buckets = group.get('buckets')
        if not isinstance(buckets, list) or not 1 <= len(buckets) <= 64:
            raise ValueError('Missing Antigravity quota windows')
        for bucket in buckets:
            if not isinstance(bucket, dict):
                raise ValueError('Invalid Antigravity quota window')
            identifier = _text(bucket.get('id'), 128)
            if identifier in identifiers:
                raise ValueError('Duplicate Antigravity quota window')
            identifiers.add(identifier)
            window = bucket.get('window')
            if window not in ('weekly', '5h'):
                raise ValueError('Unknown Antigravity quota window')
            remaining = bucket.get('remaining_fraction')
            if isinstance(remaining, bool) or not isinstance(remaining, (float, int)):
                raise ValueError('Invalid Antigravity quota fraction')
            if not 0 <= remaining <= 1 or not math.isfinite(remaining):
                raise ValueError('Invalid Antigravity quota fraction')
            windows.append({
                'id': identifier,
                'label': f'{name} · {"Weekly" if window == "weekly" else "Five-hour"}',
                'used_percent': (1 - remaining) * 100,
                'reset_at': _reset(bucket.get('reset_time')),
                'window_minutes': 10080 if window == 'weekly' else 300,
            })
    return windows


class _UsageError(Exception):
    pass


class UsageMonitor:
    def __init__(self, project, *, loader=None, run=None, clock=None, now=None, cache_seconds=60):
        self.project = Path(project).resolve()
        self._run = run or subprocess.run
        self._clock = clock or time.monotonic
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._loader = loader or self._load_usage
        self._cache_seconds = max(60.0, float(cache_seconds))
        if not math.isfinite(self._cache_seconds):
            raise ValueError('Invalid usage cache duration')
        self._lock = threading.Lock()
        self._windows = []
        self._updated_at = None
        self._updated_clock = None
        self._last_attempt = None
        self._error = None
        self._loading = False

    def _safe_path(self, path):
        try:
            if not path.resolve().is_relative_to(self.project):
                raise _UsageError('Antigravity usage log path is unsafe.')
            if path.is_symlink():
                raise _UsageError('Antigravity usage log path is unsafe.')
            if path.exists() and getattr(path.lstat(), 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 1024):
                raise _UsageError('Antigravity usage log path is unsafe.')
        except (OSError, RuntimeError, ValueError):
            raise _UsageError('Antigravity usage log path is unsafe.') from None

    def _log_path(self):
        folder = self.project
        for name in ('runtime-data', 'assistants'):
            folder = folder / name
            self._safe_path(folder)
            folder.mkdir(exist_ok=True)
            self._safe_path(folder)
        log = folder / 'antigravity-usage.log'
        self._safe_path(log)
        if log.exists() and not log.is_file():
            raise _UsageError('Antigravity usage log path is unsafe.')
        return log

    def _executable(self):
        executable = shutil.which('agy')
        if executable:
            return executable
        local = os.environ.get('LOCALAPPDATA')
        if local and Path(local).is_absolute():
            candidate = Path(local) / 'agy' / 'bin' / 'agy.exe'
            if candidate.is_file():
                return str(candidate)
        raise _UsageError('Antigravity CLI is not installed or is unavailable.')

    def _load_usage(self):
        executable = self._executable()
        log = self._log_path()
        result = self._run(
            [executable, '--log-file', str(log), '--print-timeout', '20s',
             '--output-format', 'json', '--print', '/usage'],
            cwd=str(self.project), capture_output=True, text=True, encoding='utf-8',
            errors='replace', stdin=subprocess.DEVNULL, timeout=25, shell=False,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0) if os.name == 'nt' else 0,
        )
        if result.returncode:
            raise _UsageError('Antigravity CLI usage request failed.')
        return result.stdout

    def _expired(self, windows, now):
        return any(datetime.fromisoformat(window['reset_at'].replace('Z', '+00:00')) <= now
                   for window in windows)

    def _refresh(self):
        windows = None
        updated_at = None
        error = None
        try:
            windows = parse_antigravity_usage(self._loader())
            now = self._now().astimezone(timezone.utc)
            if self._expired(windows, now):
                raise _UsageError('Antigravity CLI returned expired usage data.')
            updated_at = now.isoformat().replace('+00:00', 'Z')
        except _UsageError as exc:
            error = str(exc)
        except subprocess.TimeoutExpired:
            error = 'Antigravity CLI usage request timed out.'
        except FileNotFoundError:
            error = 'Antigravity CLI is not installed or is unavailable.'
        except (ValueError, TypeError, UnicodeError):
            error = 'Antigravity CLI returned invalid usage data.'
        except OSError:
            error = 'Antigravity CLI usage request could not be started.'
        except Exception:
            error = 'Antigravity usage is unavailable.'
        with self._lock:
            self._last_attempt = self._clock()
            self._loading = False
            self._error = error
            if error is None:
                self._windows = windows
                self._updated_at = updated_at
                self._updated_clock = self._last_attempt

    def snapshot(self, provider):
        if provider != 'antigravity':
            return {'windows': [], 'source': None, 'updated_at': None,
                    'error': 'Usage is unavailable for this provider.', 'loading': False, 'stale': False}
        with self._lock:
            clock = self._clock()
            if not self._loading and (self._last_attempt is None or clock - self._last_attempt >= self._cache_seconds):
                self._last_attempt = clock
                self._loading = True
                try:
                    threading.Thread(target=self._refresh, name='assistant-usage', daemon=True).start()
                except RuntimeError:
                    self._loading = False
                    self._error = 'Antigravity usage refresh could not be started.'
            expired = self._expired(self._windows, self._now().astimezone(timezone.utc))
            stale = bool(self._windows) and (expired or self._error is not None or
                    clock - self._updated_clock >= self._cache_seconds)
            error = self._error
            if expired and error is None:
                error = 'Antigravity usage data has expired; waiting for refreshed limits.'
            return {'windows': [dict(window) for window in self._windows],
                    'source': 'Antigravity CLI', 'updated_at': self._updated_at,
                    'error': error, 'loading': self._loading, 'stale': stale}
