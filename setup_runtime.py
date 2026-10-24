from __future__ import annotations
import os
from pathlib import Path
import shutil
import subprocess
import sys


def _run_probe(device):
    code = (
        'import torch, torchvision; '
        f'device={device!r}; '
        'x=torch.ones(1, device=device).add(1); '
        'assert float(x.cpu()[0]) == 2; '
        'boxes=torch.tensor([[0., 0., 4., 4.]], device=device); '
        'scores=torch.ones(1, device=device); '
        'assert torchvision.ops.nms(boxes, scores, .5).cpu().tolist() == [0]'
    )
    try:
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                                timeout=90, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def probe_gpu():
    return _run_probe('cuda:0')


def probe_cpu():
    return _run_probe('cpu')


def has_nvidia():
    windows = Path(os.environ.get('SystemRoot', 'C:/Windows'))
    program_files = Path(os.environ.get('ProgramFiles', 'C:/Program Files'))
    candidates = [shutil.which('nvidia-smi'), windows / 'System32/nvidia-smi.exe',
                  program_files / 'NVIDIA Corporation/NVSMI/nvidia-smi.exe']
    for executable in candidates:
        if not executable or not Path(executable).is_file():
            continue
        try:
            result = subprocess.run([str(executable), '--query-gpu=name', '--format=csv,noheader'],
                                    capture_output=True, text=True, timeout=10,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            if result.returncode == 0 and result.stdout.strip():
                return True
        except (OSError, subprocess.TimeoutExpired):
            pass
    return False


def install(gpu):
    variant = 'cu126' if gpu else 'cpu'
    command = [sys.executable, '-m', 'pip', 'install', '--upgrade',
               f'torch==2.14.1+{variant}', f'torchvision==0.29.1+{variant}',
               '--index-url', f'https://download.pytorch.org/whl/{variant}']
    try:
        return subprocess.run(command).returncode == 0
    except OSError:
        return False


def main(argv=None):
    if argv:
        print('Run setup.bat without arguments.', file=sys.stderr)
        return 1
    print('Checking training hardware…', flush=True)
    if probe_gpu():
        print('Keeping the verified NVIDIA CUDA runtime.', flush=True)
        return 0
    if has_nvidia():
        print('NVIDIA detected. Installing and checking the CUDA runtime…', flush=True)
        if install(True) and probe_gpu():
            print('NVIDIA CUDA training is ready.', flush=True)
            return 0
        print('CUDA is unavailable on this machine. Using the CPU fallback.', flush=True)
    else:
        print('No NVIDIA CUDA device detected. Using the CPU fallback.', flush=True)
    if not install(False) or not probe_cpu():
        print('CPU runtime setup failed. Resolve the installation error and run setup.bat again.', file=sys.stderr)
        return 1
    print('CPU training is ready.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
