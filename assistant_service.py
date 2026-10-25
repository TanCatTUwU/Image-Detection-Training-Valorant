from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import threading
import time
from assistant_bridge import AssistantClient, AntigravityRunner, process_frames
from assistant_launcher import MESSAGES, SESSION_ID, read_json, write_json
from assistant_loop import AssistantRequestError, ServerUnavailable, run_continuously
from model_registry import file_lock

PROJECT = Path(__file__).resolve().parent


class SessionAccessEnded(AssistantRequestError):
    def __init__(self, message):
        super().__init__(403, message)


class SessionClient:
    def __init__(self, client, gate):
        self.client = client
        self.gate = gate

    def request(self, route, payload=None):
        if route == '/session':
            return self.client.request(route, payload)
        self.gate()
        try:
            return self.client.request(route, payload)
        except AssistantRequestError as error:
            if error.status == 403:
                self.gate()
                return self.client.request(route, payload)
            raise

    def image(self, route):
        self.gate()
        try:
            return self.client.image(route)
        except AssistantRequestError as error:
            if error.status == 403:
                self.gate()
                return self.client.image(route)
            raise


def load_config(path, *, project=None):
    project = Path(project or PROJECT).resolve(strict=True)
    path = Path(path).absolute()
    if not path.is_relative_to(project):
        raise ValueError('Assistant configuration must stay inside the project runtime-data folder')
    relative = path.relative_to(project)
    if len(relative.parts) != 4 or relative.parts[:2] != ('runtime-data', 'assistant-sessions') or not SESSION_ID.fullmatch(relative.parts[2]) or relative.parts[3] != 'connection.json':
        raise ValueError('Use the private approved assistant session configuration')
    current = project
    for part in relative.parts:
        current = current / part
        if current.is_symlink() or not current.resolve().is_relative_to(project):
            raise ValueError('Assistant session files must stay inside their ordinary private folder')
    config = read_json(path)
    if config is None or set(config) != {'session_id', 'connection_id', 'api_base', 'recording_id', 'headers', 'launch_id'} or config.get('session_id') != relative.parts[2] or not isinstance(config.get('connection_id'), str) or not re.fullmatch(r'assistant-[a-f0-9]{24}', config['connection_id']) or not isinstance(config.get('recording_id'), str) or not re.fullmatch(r'run-[a-f0-9]{24}', config['recording_id']):
        raise ValueError('Invalid approved assistant session configuration')
    if 'launch_id' in config and (not isinstance(config['launch_id'], str) or not re.fullmatch(r'worker-[a-f0-9]{24}', config['launch_id'])):
        raise ValueError('Invalid assistant worker launch identity')
    if not isinstance(config.get('headers'), dict) or set(config['headers']) != {'X-Assistant-Token'}:
        raise ValueError('A private scoped assistant connection is required')
    AssistantClient(config.get('api_base'), config['headers']['X-Assistant-Token'])
    return config


def run_service(connection_file, *, project=None, client_factory=None, runner_factory=None, loop=None, wait=None, stop_event=None):
    project = Path(project or PROJECT).resolve(strict=True)
    path = Path(connection_file)
    if path.is_symlink():
        raise ValueError('Use the private approved assistant connection file')
    config = load_config(path, project=project)
    path = path.resolve(strict=True)
    factory = client_factory or AssistantClient
    create_runner = runner_factory or AntigravityRunner
    run_loop = loop or run_continuously
    stop = stop_event or threading.Event()
    pause = wait or stop.wait
    client = factory(config['api_base'], config['headers']['X-Assistant-Token'])
    sid = config['session_id']
    status_path = path.parent / 'worker-status.json'
    phase = ['starting']
    last_status = [None, 0]
    healthy_progress_at = [None]
    heartbeat_stop = threading.Event()
    state_lock = threading.RLock()
    runner = None
    retries = 0
    cache_root = project / 'runtime-data' / 'assistant-cache'
    if cache_root.is_symlink() or not cache_root.resolve().is_relative_to(project):
        raise ValueError('Assistant cache must stay inside its ordinary runtime-data folder')

    def local_status(state, *, force=False):
        with state_lock:
            phase[0] = state
            now = time.time()
            if not force and last_status[0] == state and now - last_status[1] < 5:
                return
            changed = last_status[0] != state
            write_json(status_path, dict(schema_version=1, session_id=sid, connection_id=config['connection_id'], launch_id=config.get('launch_id'), state=state, heartbeat_at=time.time(), healthy_progress_at=healthy_progress_at[0]))
            last_status[:] = [state, now]
            if changed and state in ('starting', 'running', 'paused', 'restarting'):
                print(MESSAGES[state], flush=True)

    def report(state):
        local_status(state)
        try:
            client.request('/session', dict(session_id=sid, launch_id=config['launch_id'], state=state))
        except (ServerUnavailable, AssistantRequestError):
            pass

    def control():
        current = client.request('/session')
        if not isinstance(current, dict) or current.get('authorized') is not True or current.get('session_id') != sid or current.get('launch_id') != config['launch_id'] or current.get('model') != 'gemini-3.8-flash-low':
            raise SessionAccessEnded('The approved assistant session is no longer available')
        if type(current.get('active')) is not bool:
            raise SessionAccessEnded('Invalid assistant session control')
        return current

    def gate():
        while not stop.is_set():
            current = control()
            if current['active']:
                local_status('running')
                return current
            local_status('paused')
            if pause(1):
                raise SessionAccessEnded('The assistant worker stopped')
        raise SessionAccessEnded('The assistant worker stopped')

    def heartbeat():
        while not heartbeat_stop.wait(5):
            try:
                local_status(phase[0], force=True)
            except (OSError, ValueError):
                stop.set()
                break

    def completed(batch):
        nonlocal retries
        if batch.get('completed', 0) > 0:
            retries = 0
            healthy_progress_at[0] = time.time()
            local_status('running', force=True)
        if type(batch.get('processed')) is int and batch['processed'] > 0:
            values = {key: value if type(value := batch.get(key, 0)) is int and value >= 0 else 0 for key in ('processed', 'completed', 'added')}
            print(f"Batch: {values['processed']} processed, {values['completed']} completed, {values['added']} boxes added.", flush=True)

    lock_path = path.parent / 'worker.lock'
    if lock_path.is_symlink():
        raise ValueError('Invalid assistant worker lock')
    with file_lock(lock_path, timeout=.1):
        local_status('starting')
        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()
        result = dict(state='stopped')
        try:
            while not stop.is_set():
                try:
                    current = gate()
                    if runner is None:
                        try:
                            runner = create_runner(cache_root, model=current['model'])
                        except (OSError, ValueError):
                            result = dict(state='provider_unavailable')
                            break
                    report('running')
                    result = run_loop(SessionClient(client, gate), runner, cache_root, process_batch=process_frames, stop_event=stop, on_batch=completed)
                    state = result.get('state')
                    if state == 'server_unavailable':
                        local_status('server_unavailable')
                        break
                    if state != 'session_ended':
                        if state not in MESSAGES:
                            result = dict(state='invalid_output')
                        break
                    runner.close()
                    runner = None
                    if retries >= 5:
                        result = dict(state='provider_unavailable')
                        break
                    report('restarting')
                    if pause(min(2 ** retries, 8)):
                        result = dict(state='stopped')
                        break
                    retries += 1
                except ServerUnavailable:
                    result = dict(state='server_unavailable')
                    local_status('server_unavailable')
                    break
                except (SessionAccessEnded, AssistantRequestError):
                    result = dict(state='access_unavailable')
                    break
                except (ValueError, KeyError, TypeError):
                    result = dict(state='invalid_output')
                    break
                except OSError:
                    result = dict(state='access_unavailable')
                    break
            if result['state'] != 'server_unavailable':
                report(result['state'])
            return result
        except KeyboardInterrupt:
            result = dict(state='stopped')
            report('stopped')
            return result
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=.2)
            close = getattr(runner, 'close', None)
            if callable(close):
                close()
            local_status(result['state'])


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run an already approved local board assistant session')
    parser.add_argument('--connection-file', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_service(args.connection_file)
    except (OSError, ValueError, KeyError, TypeError):
        result = dict(state='access_unavailable')
    state = result.get('state', 'stopped')
    print(json.dumps(dict(state=state, message=MESSAGES.get(state, MESSAGES['stopped'])), allow_nan=False), flush=True)
    return 0 if state == 'stopped' else 2


if __name__ == '__main__':
    raise SystemExit(main())
