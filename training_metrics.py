from __future__ import annotations
from collections import OrderedDict
import csv
import io
import math
from pathlib import Path
import threading

FIELDS = {
    'train_box_loss': ('train/box_loss',),
    'val_box_loss': ('val/box_loss',),
    'val_cls_loss': ('val/cls_loss',),
    'map50': ('metrics/mAP50(B)', 'metrics/mAP50'),
    'precision': ('metrics/precision(B)', 'metrics/precision'),
}

def finite_number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None

def valid_epoch(value):
    number = finite_number(value)
    return int(number) if not isinstance(value, bool) and number is not None and number.is_integer() and 1 <= number <= 1000 else None

def read_metrics(path):
    with path.open('rb') as stream:
        raw = stream.read(4 * 1024 * 1024 + 1)
    raw = raw[:4 * 1024 * 1024]
    end = raw.rfind(b'\n')
    if end < 0:
        return [], None
    reader = csv.DictReader(io.StringIO(raw[:end + 1].decode('utf-8-sig', errors='replace')))
    if not reader.fieldnames:
        return [], None
    reader.fieldnames = [name.strip() for name in reader.fieldnames]
    rows = {}
    fitness = {}
    try:
        for values in reader:
            epoch = valid_epoch(values.get('epoch'))
            if epoch is None:
                continue
            row = {'epoch': epoch}
            for field, aliases in FIELDS.items():
                row[field] = next((finite_number(values[name]) for name in aliases if name in values), None)
            rows[epoch] = row
            score = next((finite_number(values[name]) for name in ('metrics/mAP50-95(B)', 'metrics/mAP50-95') if name in values), row['map50'])
            if score is not None:
                fitness[epoch] = score
    except csv.Error:
        pass
    history = [rows[epoch] for epoch in sorted(rows)]
    best = max(fitness, key=fitness.get) if fitness else None
    return history, best

class TrainingMetrics:

    def __init__(self, project):
        self.project = Path(project).resolve()
        self.root = self.project / 'training-runs'
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def _path(self, run_path):
        if not isinstance(run_path, (str, Path)) or not str(run_path):
            return None
        try:
            if self.root.resolve() != self.root:
                return None
            run = Path(run_path)
            run = (run if run.is_absolute() else self.project / run).resolve()
            path = (run / 'results.csv').resolve()
            if run == self.root or not run.is_relative_to(self.root) or not path.is_relative_to(self.root):
                return None
            return path
        except (OSError, RuntimeError, ValueError):
            return None

    def _read(self, path):
        return read_metrics(path)

    def snapshot(self, run_path, *, stopped_early=False, best_epoch=None):
        history, best = [], None
        path = self._path(run_path)
        if path is not None:
            with self.lock:
                try:
                    stat = path.stat()
                    signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
                    cached = self.cache.get(path)
                    if cached is None or cached[0] != signature:
                        parsed = self._read(path)
                        self.cache[path] = (signature, parsed)
                        self.cache.move_to_end(path)
                        while len(self.cache) > 8:
                            self.cache.popitem(last=False)
                    history, best = self.cache[path][1]
                except (OSError, ValueError):
                    self.cache.pop(path, None)
        history = [dict(row) for row in history]
        return {'history': history, 'latest': dict(history[-1]) if history else None,
                'best_epoch': valid_epoch(best_epoch) or best, 'stopped_early': bool(stopped_early)}
