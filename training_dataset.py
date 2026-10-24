from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil

RECORDING_ID = re.compile(r'run-[a-f0-9]{16,64}\Z')


def validate_recording_selection(recording_ids, validation_ids=()):
    if not isinstance(recording_ids, (list, tuple)) or not 1 <= len(recording_ids) <= 128:
        raise ValueError('Select between 1 and 128 recordings')
    if any(not isinstance(value, str) or not RECORDING_ID.fullmatch(value) for value in recording_ids):
        raise ValueError('Choose known recording IDs from the recording library')
    if len(set(recording_ids)) != len(recording_ids):
        raise ValueError('Select each recording only once')
    if not isinstance(validation_ids, (list, tuple)) or any(not isinstance(value, str) for value in validation_ids):
        raise ValueError('Validation recordings must be a list of selected IDs')
    if len(set(validation_ids)) != len(validation_ids) or not set(validation_ids).issubset(recording_ids):
        raise ValueError('Validation recordings must be unique and selected for training')
    if validation_ids and len(validation_ids) == len(recording_ids):
        raise ValueError('Keep at least one training recording outside the validation holdout')


def export_recordings(project, stores, validation_ids=()):
    from review_dataset import yolo_text
    project = Path(project).resolve(strict=True)
    ids = [rid for rid, _ in stores]
    validate_recording_selection(ids, validation_ids)
    holdout = set(validation_ids)
    sources, source_splits, timestamps = [], {}, {}
    seen_roots = set()
    for rid, store in stores:
        if store.root in seen_roots:
            raise ValueError('Select each recording only once')
        seen_roots.add(store.root)
        with store._locked():
            data = store.manifest()
            store._complete(data)
            frames = [frame for frame in data['frames'] if frame['status'] == 'approved']
        if not frames:
            raise ValueError(f'{store.root.name} has no approved frames')
        role = ('val' if rid in holdout else 'train') if holdout else None
        source = data.get('source', {})
        original_path = source.get('path') if isinstance(source, dict) else None
        if isinstance(original_path, str) and original_path.strip():
            identity = os.path.normcase(str(Path(original_path).expanduser().resolve()))
            if holdout:
                if identity in source_splits and source_splits[identity] != role:
                    raise ValueError('Training and validation recordings come from the same source video; hold out a different recording')
                source_splits[identity] = role
            else:
                settings = data.get('settings') or {}
                gap = settings.get('gap_seconds', 5) if isinstance(settings, dict) else 5
                gap = gap if type(gap) in (int, float) and math.isfinite(gap) and gap >= 0 else 5
                start, end = (settings.get(key) for key in ('start_seconds', 'end_seconds')) if isinstance(settings, dict) else (None, None)
                if all(type(value) in (int, float) and math.isfinite(value) for value in (start, end)) and end > start:
                    gap = min(gap, (end - start) * .1)
                for frame in frames:
                    timestamp = frame.get('time_seconds')
                    if type(timestamp) in (int, float) and math.isfinite(timestamp) and timestamp >= 0:
                        timestamps.setdefault(identity, []).append((timestamp, frame['split'], gap, rid))
        sources.append((rid, store, frames, role))
    for frames in timestamps.values():
        frames.sort()
        previous = {}
        for timestamp, split, gap, rid in frames:
            for (old_split, old_rid), (latest, expires) in previous.items():
                if old_split != split and old_rid != rid and (timestamp - latest <= 1e-6 or timestamp < max(expires, latest + gap) - 1e-6):
                    raise ValueError('Nearby frames from the same video cross the temporal train/validation gap; select disjoint clips or a different validation recording')
            _, expires = previous.get((split, rid), (timestamp, timestamp))
            previous[split, rid] = timestamp, max(expires, timestamp + gap)
    counts = dict(train=0, val=0)
    for _, _, frames, role in sources:
        for frame in frames:
            counts[role or frame['split']] += 1
    if not all(counts.values()):
        raise ValueError('Selected recordings need approved training and validation images; approve both saved splits or choose a validation holdout')
    parent = project / 'training-exports'
    if parent.is_symlink() or parent.resolve().parent != project:
        raise ValueError('Training exports must stay inside the project')
    parent.mkdir(exist_ok=True)
    for _ in range(100):
        destination = parent / ('export-' + secrets.token_hex(8))
        try:
            destination.mkdir(exist_ok=False)
            break
        except FileExistsError:
            continue
    else:
        raise ValueError('Could not reserve a unique training export')
    counts = dict(train=0, val=0)
    negatives = dict(train=0, val=0)
    digests, report, skipped = {}, [], 0
    try:
        for split in ('train', 'val'):
            (destination / 'images' / split).mkdir(parents=True)
            (destination / 'labels' / split).mkdir(parents=True)
        for rid, store, frames, role in sources:
            exported = dict(train=0, val=0)
            provenance = []
            for frame in frames:
                split = role or frame['split']
                name = f"{rid}-{frame['id']}"
                text = ''.join(sorted(yolo_text(frame['boxes']).splitlines(keepends=True)))
                image = store._path(frame['image'])
                target = destination / 'images' / split / f'{name}.png'
                digest = hashlib.sha256()
                with image.open('rb') as incoming, target.open('xb') as outgoing:
                    signature = os.fstat(incoming.fileno())
                    for block in iter(lambda: incoming.read(1024 * 1024), b''):
                        digest.update(block)
                        outgoing.write(block)
                    after = os.fstat(incoming.fileno())
                    if (signature.st_size, signature.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise ValueError('An approved image changed during export; try again')
                key = digest.digest()
                if key in digests:
                    previous_split, previous_text = digests[key]
                    if previous_split != split:
                        raise ValueError('Duplicate image content in both train and val; remove overlapping footage from the selection')
                    if previous_text != text:
                        raise ValueError('Identical frames have conflicting approved labels; correct them before training')
                    target.unlink()
                    skipped += 1
                    continue
                digests[key] = split, text
                (destination / 'labels' / split / f'{name}.txt').write_text(text, encoding='utf-8', newline='\n')
                counts[split] += 1
                negatives[split] += int(not frame['boxes'])
                exported[split] += 1
                provenance.append(dict(id=frame['id'], revision=frame['revision'], split=split))
            report.append(dict(id=rid, path=str(store.root), role=role or 'saved', exported=exported, frames=provenance))
        if not all(counts.values()):
            raise ValueError('The combined dataset needs unique approved images in both train and val')
        (destination / 'dataset.yaml').write_text('path: .\ntrain: images/train\nval: images/val\nnames:\n  0: Head\n  1: Body\n', encoding='utf-8', newline='\n')
        result = dict(path=str(destination), **counts, backgrounds=negatives, duplicates_skipped=skipped,
                      split_mode='recording' if holdout else 'saved', recordings=report)
        (destination / 'review-report.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
        return {**result, 'recordings': [{key: value for key, value in item.items() if key != 'frames'} for item in report]}
    except BaseException:
        if destination.is_relative_to(parent) and not destination.is_symlink():
            shutil.rmtree(destination)
        raise
