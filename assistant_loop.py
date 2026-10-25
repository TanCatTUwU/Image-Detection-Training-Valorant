from __future__ import annotations
from datetime import datetime, timezone
import math
from pathlib import Path
import re
import threading


class AssistantRequestError(ValueError):
    def __init__(self, status, message):
        self.status = status
        super().__init__(f'{status}: {message}')


class ServerUnavailable(ConnectionError):
    pass


class QuotaExhausted(ValueError):
    pass

class ProviderUnavailable(ValueError):
    pass


def quota_exhausted(status, model):
    usage = status.get('connection', {}).get('usage') or {}
    if usage.get('stale') or usage.get('error'):
        return False
    family = 'gemini' if model.startswith('gemini') else '3p' if model.startswith(('claude', 'gpt')) else None
    if family is None:
        return False
    for window in usage.get('windows', []):
        identifier = str(window.get('id', '')).lower()
        used = window.get('used_percent')
        if not identifier.startswith(family + '-') or type(used) not in (int, float) or not math.isfinite(used) or used < 100:
            continue
        reset = window.get('reset_at')
        if reset:
            try:
                if datetime.fromisoformat(reset.replace('Z', '+00:00')).astimezone(timezone.utc) <= datetime.now(timezone.utc):
                    continue
            except (TypeError, ValueError):
                continue
        return True
    return False


def connection_cache(cache_root, connection):
    if not isinstance(connection, str) or not re.fullmatch(r'assistant-[a-f0-9]{16,64}', connection):
        raise ValueError('Unknown assistant connection')
    root = Path(cache_root).resolve()
    cache = root / connection
    if cache.is_symlink() or cache.resolve().parent != root:
        raise ValueError('Assistant cache must stay in its connection folder')
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def run_continuously(client, runner, cache_root, *, process_batch, batch_size=4, idle_seconds=10,
                     cursor=0, stop_event=None, on_batch=None):
    if type(batch_size) is not int or not 1 <= batch_size <= 8:
        raise ValueError('Assistant batch size must be 1..8')
    if type(idle_seconds) not in (int, float) or not math.isfinite(idle_seconds) or not 1 <= idle_seconds <= 300:
        raise ValueError('Idle polling must be 1..300 seconds')
    if type(cursor) is not int or cursor < 0:
        raise ValueError('Queue cursor must be a nonnegative integer')
    stop = stop_event or threading.Event()
    result = dict(state='stopped', processed=0, completed=0, failed=0, added=0, queued=0, batches=0, errors=[])
    status = client.request('/status')
    connection = status.get('connection', {}).get('id')
    root = Path(cache_root).resolve()
    cache = connection_cache(root, connection)
    def merge(batch):
        for key in ('processed', 'completed', 'failed', 'added', 'queued'):
            result[key] += batch.get(key, 0)
        result['batches'] += int(batch.get('processed', 0) > 0)
        result['errors'] = (result['errors'] + batch.get('errors', []))[-10:]
    from model_registry import file_lock
    with file_lock(cache / 'bridge.lock', timeout=.1):
        try:
            while not stop.is_set():
                if status.get('connection', {}).get('id') != connection:
                    raise ValueError('Assistant connection changed during processing')
                if quota_exhausted(status, runner.model):
                    result['state'] = 'quota_exhausted'
                    break
                if status.get('recording_status', 'complete') == 'complete':
                    batch = process_batch(client, runner, root, frame_limit=batch_size, cursor=cursor, status=status,batch_size=batch_size)
                    merge(batch)
                    next_cursor = batch.get('next_cursor')
                    cursor = next_cursor if next_cursor is not None else 0
                    if on_batch and batch.get('processed', 0):
                        on_batch({**batch, 'tokens_used': runner.total_tokens})
                    idle = not batch.get('processed') or (not batch.get('completed') and next_cursor is None)
                else:
                    idle = True
                if idle and stop.wait(idle_seconds):
                    break
                status = client.request('/status')
        except (QuotaExhausted, ServerUnavailable, ProviderUnavailable) as error:
            merge(getattr(error, 'summary', {}))
            result['state'] = 'quota_exhausted' if isinstance(error,QuotaExhausted) else 'server_unavailable' if isinstance(error,ServerUnavailable) else 'provider_unavailable'
            result['message'] = str(error)
        except AssistantRequestError as error:
            if error.status not in (401, 403, 404, 410):
                raise
            result['state'] = 'access_unavailable'
        except KeyboardInterrupt:
            result['state'] = 'interrupted'
    result['tokens_used'] = runner.total_tokens
    return result
