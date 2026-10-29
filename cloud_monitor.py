from __future__ import annotations
import argparse
import ctypes
import csv
from contextlib import contextmanager
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen
from training_metrics import read_metrics


class ResourceMonitor:
    def __init__(self, proc=Path('/proc'), cgroup=Path('/sys/fs/cgroup')):
        self.proc, self.cgroup = Path(proc), Path(cgroup)
        self.lock = threading.Lock()
        self.updated = -math.inf
        self.previous = None
        self.value = {'cpu': {}, 'ram': {}, 'gpus': [], 'gpu_status': 'Sampling'}

    def read(self, path):
        try:
            return path.read_text(encoding='utf-8')[:65536].strip()
        except OSError:
            return ''

    def numeric(self, value):
        try:
            number = float(value)
            return number if math.isfinite(number) and number >= 0 else None
        except (ValueError, TypeError):
            return None

    def group(self, controller, filename):
        for line in self.read(self.proc / 'self/cgroup').splitlines():
            fields = line.split(':', 2)
            if len(fields) != 3 or controller and controller not in fields[1].split(',') or not controller and fields[1]:
                continue
            relative = Path(fields[2].lstrip('/'))
            if '..' in relative.parts or relative.is_absolute():
                continue
            bases = [self.cgroup] if not controller else [self.cgroup / fields[1], self.cgroup / controller]
            for base in bases:
                for path in (base / relative, base):
                    if (path / filename).is_file():
                        return path, base
        if (self.cgroup / filename).is_file():
            return self.cgroup, self.cgroup
        return None, None

    def system(self, now):
        cpu = {'percent': None, 'scope': 'unavailable', 'cores': None}
        ram = {'percent': None, 'used_bytes': None, 'total_bytes': None, 'scope': 'unavailable'}
        memory = {}
        for line in self.read(self.proc / 'meminfo').splitlines():
            parts = line.split()
            if len(parts) >= 2:
                memory[parts[0].rstrip(':')] = (self.numeric(parts[1]) or 0) * 1024
        host_total = memory.get('MemTotal')
        if host_total and 'MemAvailable' in memory:
            ram.update(used_bytes=max(0, host_total - memory['MemAvailable']), total_bytes=host_total, scope='host')
        group, base = self.group('', 'memory.current')
        version = 2
        if group is None:
            group, base = self.group('memory', 'memory.usage_in_bytes')
            version = 1
        if group is not None:
            used = self.numeric(self.read(group / ('memory.current' if version == 2 else 'memory.usage_in_bytes')))
            limits = [host_total] if host_total else []
            path = group
            bounded = False
            while True:
                limit = self.numeric(self.read(path / ('memory.max' if version == 2 else 'memory.limit_in_bytes')))
                if limit and (not host_total or limit < host_total):
                    limits.append(limit)
                    bounded = True
                if path == base:
                    break
                path = path.parent
            if used is not None:
                ram.update(used_bytes=used, total_bytes=min(limits) if limits else None, scope='container', limited=bounded)
        try:
            cores = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            cores = os.cpu_count() or 1
        group, base = self.group('', 'cpu.stat')
        version = 2
        if group is None:
            group, base = self.group('cpuacct', 'cpuacct.usage')
            version = 1
        sample = None
        if group is not None:
            quota_group, quota_base = (group, base) if version == 2 else self.group('cpu', 'cpu.cfs_quota_us')
            path = quota_group
            while path is not None:
                values = self.read(path / 'cpu.max').split() if version == 2 else [self.read(path / 'cpu.cfs_quota_us'), self.read(path / 'cpu.cfs_period_us')]
                if len(values) == 2:
                    quota, period = map(self.numeric, values)
                    if quota and period:
                        cores = min(cores, quota / period)
                if path == quota_base:
                    break
                path = path.parent
            usage = dict(line.split()[:2] for line in self.read(group / 'cpu.stat').splitlines() if len(line.split()) >= 2) if version == 2 else {}
            counter = self.numeric(usage.get('usage_usec') if version == 2 else self.read(group / 'cpuacct.usage'))
            if counter is not None:
                sample = ('container', counter / (1e6 if version == 2 else 1e9), now, cores)
        if sample is None:
            fields = self.read(self.proc / 'stat').splitlines()
            values = [self.numeric(item) for item in fields[0].split()[1:9]] if fields and fields[0].startswith('cpu ') else []
            if len(values) >= 4 and all(item is not None for item in values):
                total = sum(values)
                idle = values[3] + (values[4] if len(values) > 4 else 0)
                sample = ('host', total - idle, total, cores)
        if self.proc == Path('/proc') and sys.platform == 'win32':
            class MemoryStatus(ctypes.Structure):
                _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [(name, ctypes.c_ulonglong) for name in ('total', 'available', 'page_total', 'page_available', 'virtual_total', 'virtual_available', 'extended')]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                ram.update(used_bytes=status.total - status.available, total_bytes=status.total, scope='host')
            idle, kernel, user = ctypes.c_ulonglong(), ctypes.c_ulonglong(), ctypes.c_ulonglong()
            if ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
                sample = ('host', kernel.value + user.value - idle.value, kernel.value + user.value, cores)
        if sample:
            scope, busy, elapsed, cores = sample
            cpu.update(scope=scope, cores=cores)
            if self.previous and self.previous[0] == scope:
                delta, duration = busy - self.previous[1], elapsed - self.previous[2]
                if duration > 0 and delta >= 0:
                    cpu['percent'] = min(100, 100 * delta / duration / (cores if scope == 'container' else 1))
            self.previous = sample
        if ram['used_bytes'] is not None and ram['total_bytes']:
            ram['percent'] = min(100, 100 * ram['used_bytes'] / ram['total_bytes'])
        return cpu, ram

    def gpu(self):
        try:
            options = {'creationflags': subprocess.CREATE_NO_WINDOW} if sys.platform == 'win32' else {}
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2, check=True, **options)
            gpus = []
            for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
                if len(row) != 6:
                    continue
                index, identity, name, percent, used, total = [item.strip() for item in row]
                percent, used, total = map(self.numeric, (percent, used, total))
                gpus.append({'index': index, 'uuid': identity, 'name': name[:120], 'percent': min(100, percent) if percent is not None else None, 'used_bytes': used * 1024**2 if used is not None else None, 'total_bytes': total * 1024**2 if total else None})
            for key in ('NVIDIA_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'):
                selection = os.environ.get(key)
                if selection is not None and selection.strip().lower() != 'all':
                    available = gpus
                    gpus = [gpu for position, gpu in enumerate(available) if any(item.strip() == (str(position) if key == 'CUDA_VISIBLE_DEVICES' else gpu['index']) or item.strip().startswith('GPU-') and gpu['uuid'].startswith(item.strip()) for item in selection.split(','))]
            return gpus, '' if gpus else 'No visible NVIDIA GPU'
        except (OSError, subprocess.SubprocessError, UnicodeError):
            return [], 'NVIDIA metrics unavailable'

    def get(self):
        now = time.monotonic()
        if now - self.updated < 5 or not self.lock.acquire(blocking=False):
            return self.value
        try:
            cpu, ram = self.system(now)
            gpus, status = self.gpu()
            self.value = {'cpu': cpu, 'ram': ram, 'gpus': gpus, 'gpu_status': status, 'sampled_at': time.time()}
            self.updated = now
            return self.value
        finally:
            self.lock.release()


def mapped_ports(environ):
    host = environ.get('PUBLIC_IPADDR', '').strip()
    try:
        ipaddress.ip_address(host)
    except ValueError:
        if not re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?', host):
            return []
    ports = []
    for name, value in environ.items():
        match = re.fullmatch(r'VAST_TCP_PORT_(\d+)', name)
        if not match or not str(value).isdigit():
            continue
        alias, external = int(match[1]), int(value)
        internal = external if alias >= 70000 else alias
        if alias in (22, 80, 443, 1111, 6006, 8080, 8384) or not 1024 <= internal <= 65535 or not 1 <= external <= 65535:
            continue
        ports.append(dict(alias=alias, port=internal, external_port=external, hostname=host))
    return sorted(ports, key=lambda item: (item['alias'] != 72299, item['alias'] != 8081, item['alias']))


def safe_path(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if not path.is_relative_to(root) or any(item.is_symlink() for item in (path, *path.parents) if item != root):
        raise ValueError('Dashboard paths must stay inside the training folder')
    return path


def read_json(root, relative):
    try:
        path = safe_path(root, relative)
        if path.stat().st_size > 1024 * 1024:
            return {}
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def clean_log(root, text):
    for path in (str(Path(root).resolve()), str(Path.home())):
        text = text.replace(path, '<local>')
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text).replace('\r\n', '\n').replace('\r', '\n')
    return re.sub(r'''(?i)((?:token|api[_-]?key|authorization|password|secret)["']?\s*[=:]\s*["']?)(?:Bearer\s+)?[^\s,;"']+''', r'\1<redacted>', text)


def worker_alive(pid):
    if type(pid) is not int or pid < 1:
        return True
    if sys.platform == 'win32':
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            kernel.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
            return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def snapshot(root, logs=False):
    state = read_json(root, 'outputs/status.json')
    settings = read_json(root, 'settings.json')
    try:
        history, best = read_metrics(safe_path(root, 'outputs/run/results.csv'))
    except (OSError, ValueError):
        history, best = [], None
    value = {'state': state.get('state', 'waiting'), 'epochs': settings.get('epochs'), 'updated_at': state.get('updated_at'), 'error': clean_log(root, str(state.get('error', ''))), 'metrics': {'history': history, 'latest': history[-1] if history else None, 'best_epoch': best}, 'resolution': [settings.get('model_width'), settings.get('model_height')]}
    worker = read_json(root, 'outputs/worker.json')
    if value['state'] in ('bootstrapping', 'training', 'exporting', 'packaging') and not worker_alive(worker.get('pid')):
        value.update(state='interrupted', error='Training worker stopped. Check the logs before resuming.')
    if logs:
        try:
            path = safe_path(root, 'outputs/worker.log')
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size - 65536))
                value['log'] = clean_log(root, stream.read(65536).decode('utf-8', errors='replace'))
        except (OSError, ValueError):
            value['log'] = ''
    return value


def monitor_id(root, token):
    return hashlib.sha256((str(Path(root).resolve()) + token).encode()).hexdigest()


class MonitorServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 16

    def __init__(self, *args, **kwargs):
        self.slots = threading.BoundedSemaphore(8)
        super().__init__(*args, **kwargs)

    def get_request(self):
        request, address = super().get_request()
        request.settimeout(5)
        return request, address

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()

    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            super().handle_error(request, client_address)


def make_server(root, *, token=None, port=0, host='127.0.0.1'):
    root = Path(root).resolve()
    safe_path(root, 'outputs').mkdir(exist_ok=True)
    token = token or secrets.token_urlsafe(32)
    identity = monitor_id(root, token)
    html = Path(__file__).with_suffix('.html').read_bytes()
    resources = ResourceMonitor()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query, keep_blank_values=True)
            values = query.get('token', [])
            auth = self.headers.get('Authorization', '')
            supplied = auth[7:] if auth.startswith('Bearer ') else values[0] if len(values) == 1 else ''
            if not hmac.compare_digest(supplied.encode('utf-8'), token.encode('utf-8')):
                return self.send_body(403, b'Access denied', 'text/plain')
            if parsed.path == '/':
                return self.send_body(200, html, 'text/html')
            if parsed.path == '/api/health':
                return self.send_body(200, json.dumps({'monitor_id': identity}).encode(), 'application/json')
            if parsed.path == '/api/status':
                value = snapshot(root, logs=query.get('logs') == ['1'])
                value['monitor_id'] = identity
                value['resources'] = resources.get()
                return self.send_body(200, json.dumps(value, allow_nan=False).encode(), 'application/json')
            self.send_body(404, b'Not found', 'text/plain')

        def do_POST(self):
            self.send_body(405, b'Read only', 'text/plain')

        def send_body(self, code, body, content_type):
            self.send_response(code)
            self.send_header('Content-Type', content_type + '; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('X-Frame-Options', 'DENY')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
    server = MonitorServer((host, port), Handler)
    server.daemon_threads = True
    server.monitor_token = token
    server.monitor_id = identity
    return server


def alive(root, record):
    try:
        port = record.get('port')
        token = record.get('token')
        if type(port) is not int or not 1 <= port <= 65535 or not isinstance(token, str):
            return False
        request = Request(f'http://127.0.0.1:{port}/api/health', headers={'Authorization': 'Bearer ' + token})
        with urlopen(request, timeout=1) as response:
            value = json.loads(response.read(65536))
        return value.get('monitor_id') == monitor_id(root, token)
    except Exception:
        return False


def print_address(record):
    print(f'Dashboard hostname: {record["hostname"]}\nDashboard internal port: {record["port"]}', flush=True)
    if record.get('external_port'):
        print(f'Dashboard external port: {record["external_port"]}', flush=True)
    print('Dashboard: ' + record['url'], flush=True)
    if not record.get('external_port'):
        print(f'Forward this port through SSH: -L {record["port"]}:127.0.0.1:{record["port"]}', flush=True)


@contextmanager
def startup_lock(root):
    path = safe_path(root, 'outputs/monitor.lock')
    path.parent.mkdir(exist_ok=True)
    with path.open('a+b') as stream:
        if sys.platform == 'win32':
            import msvcrt
            if path.stat().st_size == 0:
                stream.write(b'0')
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if sys.platform == 'win32':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def serve(root, port=None):
    root = Path(root).resolve()
    with startup_lock(root):
        record_path = safe_path(root, 'outputs/monitor.json')
        record_path.parent.mkdir(exist_ok=True)
        existing = read_json(root, 'outputs/monitor.json')
        if alive(root, existing):
            print_address(existing)
            return
        server = None
        mapping = None
        for candidate in mapped_ports(os.environ):
            if port is not None and candidate['port'] != port and candidate['alias'] != port:
                continue
            try:
                server = make_server(root, host='0.0.0.0', port=candidate['port'])
                mapping = candidate
                break
            except OSError:
                continue
        if server is None:
            server = make_server(root, host='127.0.0.1', port=port if port and port <= 65535 else 0)
        host = mapping['hostname'] if mapping else '127.0.0.1'
        public = mapping['external_port'] if mapping else None
        authority = '[' + host + ']' if ':' in host else host
        record = {'pid': os.getpid(), 'port': server.server_port, 'hostname': host, 'external_port': public, 'token': server.monitor_token, 'url': f'http://{authority}:{public or server.server_port}/?token={server.monitor_token}'}
        temporary = safe_path(root, 'outputs/monitor.json.tmp')
        descriptor = os.open(temporary, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(record, stream)
        temporary.replace(record_path)
    print_address(record)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def background(root, port=None):
    root = Path(root).resolve()
    existing = read_json(root, 'outputs/monitor.json')
    if alive(root, existing):
        print_address(existing)
        return existing
    log = safe_path(root, 'outputs/monitor.log')
    log.parent.mkdir(exist_ok=True)
    command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--root', str(root)]
    if port is not None:
        command += ['--port', str(port)]
    options = {'creationflags': subprocess.CREATE_NO_WINDOW} if sys.platform == 'win32' else {'start_new_session': True}
    with log.open('a', encoding='utf-8') as stream:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, close_fds=True, **options)
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        record = read_json(root, 'outputs/monitor.json')
        if alive(root, record):
            print_address(record)
            return record
        if child.poll() is not None:
            raise RuntimeError('Dashboard could not start; see outputs/monitor.log')
        time.sleep(.1)
    raise RuntimeError('Dashboard startup timed out; see outputs/monitor.log')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.')
    parser.add_argument('--port', type=int)
    parser.add_argument('--background', action='store_true')
    args = parser.parse_args()
    if args.port is not None and not (1024 <= args.port <= 65535 or 70000 <= args.port <= 99999):
        parser.error('Choose a valid listening port or Vast mapping alias')
    if args.background:
        background(Path(args.root), args.port)
    else:
        serve(Path(args.root), args.port)


if __name__ == '__main__':
    main()
