from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

def pick_local_video(runtime_python, *, runner=None):
    python = Path(runtime_python)
    if not python.is_file():
        raise ValueError('The prepared runtime is missing; run setup.bat first')
    command = [str(python), str(Path(__file__).resolve()), '--pick']
    env = dict(os.environ, PYTHONUTF8='1')
    env.pop('QT_QPA_PLATFORM', None)
    try:
        result = (runner or subprocess.run)(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace', env=env, check=False, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except OSError as exc:
        raise ValueError(f'Could not open the native video picker: {exc}') from exc
    try:
        payload = json.loads(result.stdout)
    except (ValueError, TypeError) as exc:
        raise ValueError('The native video picker did not return a valid result') from exc
    if result.returncode != 0 or not isinstance(payload, dict) or 'error' in payload:
        message = payload.get('error') if isinstance(payload, dict) else None
        raise ValueError(f"Could not open the native video picker: {message or 'the helper exited unexpectedly'}")
    if payload == {'cancelled': True}:
        return None
    selected = payload.get('path')
    if not isinstance(selected, str) or not selected or '://' in selected or (not Path(selected).is_absolute()):
        raise ValueError('The native picker must return an absolute local video path')
    return Path(selected)

def choose_video():
    from PySide6.QtWidgets import QApplication, QFileDialog
    app = QApplication.instance() or QApplication([])
    app.setApplicationName('Local recording picker')
    selected, _ = QFileDialog.getOpenFileName(None, 'Choose a local recording', '', 'Video files (*.mp4 *.mkv *.avi *.mov *.wmv *.webm *.m4v *.mpg *.mpeg *.ts *.mts *.m2ts);;All files (*)', options=QFileDialog.Option(0))
    return {'path': selected} if selected else {'cancelled': True}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pick', action='store_true', required=True)
    parser.parse_args(argv)
    try:
        result = choose_video()
        print(json.dumps(result, ensure_ascii=True), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({'error': str(exc)}, ensure_ascii=True), flush=True)
        return 1
if __name__ == '__main__':
    raise SystemExit(main())
