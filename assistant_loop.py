from __future__ import annotations
from datetime import datetime, timezone
import math
from pathlib import Path
import re
import threading
import time

REASONS = dict(provider_temporary_error='Antigravity had a temporary service error.', provider_empty_output='Antigravity returned no review output.', provider_session_ended='The Antigravity CLI session ended.', provider_timeout='The Antigravity CLI session timed out.', worker_exited='The assistant worker exited unexpectedly.', worker_launch_failed='The assistant command console could not open.', worker_retry_exhausted='The assistant worker could not stay running after five reconnect attempts.')


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

class ProviderTemporaryError(ProviderUnavailable):
    def __init__(self, message, *, reason='provider_temporary_error'):
        self.reason = reason if reason in REASONS else 'provider_temporary_error'
        super().__init__(message)

class ProviderSessionEnded(ProviderUnavailable):
    def __init__(self, message, *, reason='provider_session_ended'):
        self.reason = reason if reason in REASONS else 'provider_session_ended'
        super().__init__(message)


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
                     cursor=0, stop_event=None, on_batch=None, on_retry=None, retry_epoch=0):
    if type(batch_size) is not int or not 1 <= batch_size <= 8:
        raise ValueError('Assistant batch size must be 1..8')
    if type(idle_seconds) not in (int, float) or not math.isfinite(idle_seconds) or not 1 <= idle_seconds <= 300:
        raise ValueError('Idle polling must be 1..300 seconds')
    if type(cursor) is not int or cursor < 0:
        raise ValueError('Queue cursor must be a nonnegative integer')
    if type(retry_epoch) is not int or retry_epoch < 0:
        raise ValueError('Invalid assistant retry generation')
    stop = stop_event or threading.Event()
    result = dict(state='stopped', processed=0, completed=0, failed=0, added=0, edited=0, queued=0, batches=0, errors=[])
    status = client.request('/status')
    connection = status.get('connection', {}).get('id')
    root = Path(cache_root).resolve()
    cache = connection_cache(root, connection)
    def merge(batch):
        for key in ('processed', 'completed', 'failed', 'added', 'edited', 'queued'):
            result[key] += batch.get(key, 0)
        result['batches'] += int(batch.get('processed', 0) > 0)
        result['errors'] = (result['errors'] + batch.get('errors', []))[-10:]
    from model_registry import file_lock
    from assistant_launcher import read_json, write_json
    with file_lock(cache / 'bridge.lock', timeout=.1):
        state_path = cache / 'loop-state.json'
        saved = read_json(state_path, {})
        if saved.get('connection_id') != connection or saved.get('retry_epoch', 0) != retry_epoch:
            saved = {}
        provider_failures = saved.get('provider_failures', 0)
        retry_reason = saved.get('reason', 'provider_temporary_error')
        next_retry_at = saved.get('next_retry_at', 0)
        if type(provider_failures) is not int or not 0 <= provider_failures <= 6 or retry_reason not in REASONS or type(next_retry_at) not in (int, float) or not math.isfinite(next_retry_at):
            raise ValueError('Invalid saved assistant retry state')
        def save_retries():
            write_json(state_path, dict(schema_version=1, connection_id=connection, retry_epoch=retry_epoch, provider_failures=provider_failures, reason=retry_reason, next_retry_at=next_retry_at))
        def retry_status():
            return dict(reason=retry_reason, retry_attempts=min(provider_failures, 5), retry_exhausted=provider_failures >= 6)
        try:
            while not stop.is_set():
                if provider_failures >= 6:
                    result.update(state='provider_unavailable', **retry_status())
                    break
                if next_retry_at > time.time():
                    if stop.wait(min(next_retry_at - time.time(), 8)):
                        break
                    next_retry_at = 0
                    save_retries()
                if status.get('connection', {}).get('id') != connection:
                    raise ValueError('Assistant connection changed during processing')
                if quota_exhausted(status, runner.model):
                    result['state'] = 'quota_exhausted'
                    break
                if status.get('recording_status', 'complete') == 'complete':
                    try:
                        batch = process_batch(client, runner, root, frame_limit=batch_size, cursor=cursor, status=status,batch_size=batch_size)
                    except ProviderTemporaryError as error:
                        partial = getattr(error, 'summary', {})
                        merge(partial)
                        if partial.get('completed', 0) > 0:
                            provider_failures = 0
                            if on_batch:
                                on_batch({**partial, 'tokens_used': runner.total_tokens})
                        provider_failures += 1
                        retry_reason = error.reason
                        next_retry_at = time.time() + min(2 ** (provider_failures - 1), 8) if provider_failures < 6 else 0
                        save_retries()
                        if on_retry:
                            on_retry(retry_status())
                        if provider_failures >= 6:
                            result.update(state='provider_unavailable', **retry_status())
                            break
                        if stop.wait(min(2 ** (provider_failures - 1), 8)):
                            break
                        next_retry_at = 0
                        save_retries()
                        status=client.request('/status')
                        continue
                    if batch.get('completed', 0) > 0:
                        provider_failures = 0
                        next_retry_at = 0
                        save_retries()
                        if on_retry:
                            on_retry(dict(reason=None, retry_attempts=0, retry_exhausted=False))
                    merge(batch)
                    next_cursor = batch.get('next_cursor')
                    cursor = next_cursor if next_cursor is not None else 0
                    if on_batch and batch.get('processed', 0):
                        on_batch({**batch, 'tokens_used': runner.total_tokens})
                    idle = not batch.get('processed') or (not batch.get('completed') and next_cursor is None)
                else:
                    idle = True
                if stop.is_set() or idle and stop.wait(idle_seconds):
                    break
                status = client.request('/status')
        except (QuotaExhausted, ServerUnavailable, ProviderUnavailable) as error:
            merge(getattr(error, 'summary', {}))
            result['state'] = 'quota_exhausted' if isinstance(error,QuotaExhausted) else 'server_unavailable' if isinstance(error,ServerUnavailable) else 'session_ended' if isinstance(error,ProviderSessionEnded) else 'provider_unavailable'
            result['message'] = str(error)
            result['reason'] = getattr(error, 'reason', result['state'])
        except AssistantRequestError as error:
            if error.status not in (401, 403, 404, 410):
                raise
            result['state'] = 'access_unavailable'
        except KeyboardInterrupt:
            result['state'] = 'interrupted'
        result.setdefault('retry_attempts', min(provider_failures, 5))
    result['tokens_used'] = runner.total_tokens
    return result
