from __future__ import annotations
import os
from pathlib import Path
import re
import secrets
import stat
from model_registry import atomic_json, file_lock, read_json, utc_now

TRASH_ID = re.compile(r'trash-[a-f0-9]{24}\Z')

def _ordinary(path):
    path = Path(path).absolute()
    if path.is_symlink() or path.resolve() != path:
        raise ValueError('Managed paths must not contain symbolic links or junctions')
    try:
        attributes = path.lstat().st_file_attributes if os.name == 'nt' else 0
    except FileNotFoundError:
        return path
    if attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        raise ValueError('Managed paths must not contain junctions or reparse points')
    return path

def checked_managed_path(project, kind, path, *, exists=True):
    project = Path(project).resolve()
    names = {'model': 'model-library', 'recording': 'dataset-runs'}
    if kind not in names:
        raise ValueError('Unknown deleted item kind')
    parent = _ordinary(project / names[kind])
    path = Path(path).absolute()
    if path.parent != parent or not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', path.name):
        raise ValueError('Only managed model-library or dataset-runs folders can be deleted or restored')
    if kind == 'model' and not re.fullmatch(r'model-[a-f0-9]{24}', path.name):
        raise ValueError('Invalid managed model path')
    _ordinary(path)
    if exists and not path.is_dir():
        raise ValueError('The managed item folder is unavailable')
    return path

def _ordinary_tree(path):
    _ordinary(path)
    for parent, directories, files in os.walk(path, followlinks=False):
        for name in directories + files:
            _ordinary(Path(parent) / name)

class LocalTrash:

    def __init__(self, project):
        self.project = Path(project).resolve()
        self.root = self.project / 'deleted-items'

    def _prepare(self):
        _ordinary(self.root)
        self.root.mkdir(exist_ok=True)

    def resolve(self, trash_id):
        self._prepare()
        if not isinstance(trash_id, str) or not TRASH_ID.fullmatch(trash_id):
            raise ValueError('Choose a known deleted item ID')
        folder = _ordinary(self.root / trash_id)
        metadata = _ordinary(folder / 'item.json')
        record = read_json(metadata)
        if not isinstance(record, dict) or record.get('id') != trash_id or record.get('kind') not in ('model', 'recording'):
            raise ValueError('The deleted item metadata is unavailable or invalid')
        original = checked_managed_path(self.project, record['kind'], record.get('original_path', ''), exists=False)
        payload = _ordinary(folder / 'payload')
        if not payload.is_dir():
            raise ValueError('This deleted item was already restored or is unavailable')
        if record['kind'] == 'model' and record.get('model_id') != original.name:
            raise ValueError('The deleted model metadata does not match its managed path')
        if record['kind'] == 'recording':
            from recording_library import recording_id
            if record.get('recording_id') != recording_id(original):
                raise ValueError('The deleted recording metadata does not match its managed path')
        return {**record, 'original_path': str(original), 'payload_path': str(payload)}

    def list(self):
        self._prepare()
        items = []
        for folder in self.root.iterdir():
            if not TRASH_ID.fullmatch(folder.name):
                continue
            try:
                record = self.resolve(folder.name)
            except (ValueError, OSError):
                continue
            items.append({key: record[key] for key in ('id', 'kind', 'label', 'original_path', 'deleted_at')})
        return sorted(items, key=lambda item: item['deleted_at'], reverse=True)

    def put(self, kind, path, label, **metadata):
        self._prepare()
        with file_lock(self.root / 'trash.lock'):
            path = checked_managed_path(self.project, kind, path)
            _ordinary_tree(path)
            for _ in range(100):
                trash_id = 'trash-' + secrets.token_hex(12)
                folder = self.root / trash_id
                try:
                    folder.mkdir(exist_ok=False)
                    break
                except FileExistsError:
                    continue
            else:
                raise ValueError('Could not reserve a unique deleted item folder')
            record = {**metadata, 'id': trash_id, 'kind': kind, 'label': label, 'original_path': str(path), 'deleted_at': utc_now()}
            atomic_json(folder / 'item.json', record)
            os.replace(path, folder / 'payload')
            return {**record, 'payload_path': str(folder / 'payload')}

    def restore(self, trash_id, *, kind=None, commit=None):
        self._prepare()
        with file_lock(self.root / 'trash.lock'):
            record = self.resolve(trash_id)
            if kind is not None and record['kind'] != kind:
                raise ValueError('The deleted item has a different kind')
            source = Path(record['payload_path'])
            target = Path(record['original_path'])
            _ordinary_tree(source)
            if target.exists():
                raise ValueError('The original path already exists; the replacement and deleted item were preserved')
            if not target.parent.is_dir():
                raise ValueError('The original managed directory is unavailable')
            os.replace(source, target)
            try:
                return commit(record, target) if commit is not None else record
            except BaseException:
                os.replace(target, source)
                raise
