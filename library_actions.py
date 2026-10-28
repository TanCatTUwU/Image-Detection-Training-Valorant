from __future__ import annotations
from contextlib import nullcontext
from pathlib import Path
import re
from model_registry import process_alive, read_json

ACTIVE = {'preparing', 'queued', 'running', 'exporting'}


def _inside(value, root):
    if not isinstance(value, (str, Path)) or not str(value):
        return False
    try:
        return Path(value).resolve().is_relative_to(root)
    except (ValueError, OSError, RuntimeError):
        return False


class LibraryActions:

    def __init__(self, project, board, recordings, *, proposals, assistants, stores, stores_lock):
        self.project = Path(project).resolve()
        self.board, self.recordings = board, recordings
        self.proposals, self.assistants = proposals, assistants
        self.stores, self.stores_lock = stores, stores_lock

    def _active(self, folder, pointer='latest'):
        root = self.project / folder
        index = read_json(root / f'{pointer}.json', {})
        jid = index.get('id') if isinstance(index, dict) else None
        if not isinstance(jid, str) or not re.fullmatch(r'(?:job|video|empty)-[a-f0-9]{16}', jid):
            return None
        path = root / jid
        if root.is_symlink() or path.is_symlink() or path.resolve().parent != root:
            raise ValueError('The active job directory leaves the application')
        job = read_json(path / 'job.json', {})
        if not isinstance(job, dict) or job.get('state') not in ACTIVE:
            return None
        pid = job.get('preparation_pid') if job.get('state') == 'preparing' else job.get('worker_pid') or job.get('launch_pid')
        if folder == 'bundle-jobs' and type(pid) is not int:
            return None
        return None if type(pid) is int and not process_alive(pid) else job

    def _refresh(self):
        for owner in (self.board, self.recordings):
            latest = getattr(owner, '_latest', None)
            if callable(latest):
                latest()

    def _close_proposer(self, *, model_id=None, root=None):
        manager = self.proposals()
        if manager is None:
            return
        with manager._lock:
            if model_id is not None and manager._model_id != model_id:
                return
            if root is not None and getattr(getattr(manager, 'store', None), 'root', None) != root:
                return
            model = manager._proposer
            session = getattr(getattr(model, 'model', None), 'session', None)
            if callable(getattr(session, 'close', None)):
                session.close()
            manager._proposer = manager._model_id = None

    def delete_model(self, model_id):
        self._refresh()
        entry = self.board.registry.resolve(model_id)
        if model_id.startswith('builtin-'):
            raise ValueError('Included models are kept with the application')
        if entry.get('is_active'):
            raise ValueError('Choose another default model before deleting this model')
        imports = getattr(self.board, 'model_imports', None)
        job = imports.snapshot() if imports is not None else None
        if isinstance(job, dict) and job.get('state') in ACTIVE:
            raise ValueError('Wait for model import or conversion to finish before deleting a model')
        preparation = self._active('video-jobs')
        if preparation and (preparation.get('model_id') == model_id or preparation.get('model_path') == entry['path']):
            raise ValueError('This model is in use by recording preparation; cancel it first')
        training = self._active('training-jobs')
        model_root = self.project / 'model-library' / model_id
        if training and training.get('state') == 'exporting':
            raise ValueError('Wait for training model export and registration to finish before deleting a model')
        if training and (_inside(training.get('checkpoint'), model_root) or training.get('options', {}).get('checkpoint_id') == model_id):
            raise ValueError('This model is in use by training; wait for training to stop')
        bundle = self._active('bundle-jobs')
        if bundle and (_inside(bundle.get('checkpoint'), model_root) or bundle.get('checkpoint_id') == model_id or bundle.get('options', {}).get('checkpoint_id') == model_id):
            raise ValueError('This model is in use by a training ZIP bundle; wait for the export to finish')
        scan = self._active('empty-scan-jobs', 'active')
        if scan and (scan.get('model_id') == model_id or scan.get('model_path') == entry['path']):
            raise ValueError('This model is in use by empty screening; cancel the scan first')
        directml = getattr(self.board, '_directml', None)
        overlay = directml() if callable(directml) else read_json(self.project / 'training-jobs/directml.json', {})
        if isinstance(overlay, dict) and overlay.get('running') and overlay.get('model_id') == model_id:
            raise ValueError('Stop the DirectML overlay before deleting the model it uses')
        self._close_proposer(model_id=model_id)
        return self.board.registry.delete(model_id)

    def delete_recording(self, recording_id):
        self._refresh()
        root = Path(self.recordings.resolve(recording_id)).resolve()
        preparation = self._active('video-jobs')
        if preparation and _inside(preparation.get('run_path'), root):
            raise ValueError('This recording is in use by preparation; cancel it first')
        training = self._active('training-jobs')
        if training and training.get('state') == 'preparing' and recording_id in training.get('source_recording_ids', []):
            raise ValueError('This recording is in use by training preparation; wait for preparation to finish')
        if training and _inside(training.get('data'), root):
            raise ValueError('This recording export is in use by training; wait for training to stop')
        bundle = self._active('bundle-jobs')
        if bundle and recording_id in bundle.get('source_recording_ids', []):
            raise ValueError('This recording is in use by a training ZIP bundle; wait for the export to finish')
        scan = self._active('empty-scan-jobs', 'active')
        if scan and _inside(scan.get('run_path'), root):
            raise ValueError('This recording is in use by an empty scan; cancel the scan first')
        self.assistants().revoke_recording(root)
        self._close_proposer(root=root)
        with self.stores_lock:
            store = self.stores.get(root)
        with getattr(store, '_thread_lock', nullcontext()):
            result = self.recordings.delete(recording_id)
            with self.stores_lock:
                old = self.stores.pop(root, None)
                if old is not None:
                    old._retired = True
            return result

    def deleted_items(self):
        from local_trash import LocalTrash
        return {'items': LocalTrash(self.project).list()}

    def restore(self, trash_id):
        from local_trash import LocalTrash
        item = LocalTrash(self.project).resolve(trash_id)
        if item['kind'] == 'model':
            return self.board.registry.restore(trash_id)
        if item['kind'] == 'recording':
            restored = self.recordings.restore(trash_id)
            root = Path(restored['path']).resolve()
            with self.stores_lock:
                self.stores.pop(root, None)
            return restored
        raise ValueError('Unknown deleted item type')
