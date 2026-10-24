from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import cv2
import numpy as np
PROJECT = Path(__file__).resolve().parent

def sample_times(start, end, sample_fps, max_images):
    if not all((math.isfinite(v) for v in (start, end, sample_fps))):
        raise ValueError('Sampling times and FPS must be finite')
    if start < 0 or end <= start or sample_fps <= 0 or (max_images < 1):
        raise ValueError('Choose a positive duration, sampling FPS and maximum image count')
    step = max(1 / sample_fps, (end - start) / max_images)
    return [start + n * step for n in range(min(max_images, math.ceil((end - start) / step))) if start + n * step < end]

def temporal_split(timestamp, start, end, val_fraction, gap_seconds):
    if not math.isfinite(val_fraction) or not math.isfinite(gap_seconds) or (not 0 < val_fraction < 1) or (gap_seconds < 0):
        raise ValueError('Validation fraction must be between 0 and 1; gap must be nonnegative')
    boundary = end - (end - start) * val_fraction
    gap = min(gap_seconds, (end - start) * 0.1)
    if timestamp < boundary - gap / 2:
        return 'train'
    if timestamp >= boundary + gap / 2:
        return 'val'
    return None

class FrameDeduplicator:

    def __init__(self):
        self.exact = set()
        self.signatures = []

    def accept(self, frame):
        digest = hashlib.sha256(np.ascontiguousarray(frame).tobytes()).digest()
        if digest in self.exact:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        reduced = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        packed = np.packbits(reduced[:, 1:] > reduced[:, :-1]).tobytes()
        hash_value = int.from_bytes(packed, 'little')
        thumbnail = cv2.resize(gray, (32, 24), interpolation=cv2.INTER_AREA).astype(np.float32)
        for old_hash, old_thumbnail in self.signatures:
            if (hash_value ^ old_hash).bit_count() <= 3 and np.abs(thumbnail - old_thumbnail).mean() < 0.8:
                return False
        self.exact.add(digest)
        self.signatures.append((hash_value, thumbnail))
        return True

def map_boxes(boxes, region, frame_shape):
    left, top, width, height = region
    full_height, full_width = frame_shape[:2]
    return [{**box, 'x1': (left + box['x1'] * width) / full_width, 'y1': (top + box['y1'] * height) / full_height, 'x2': (left + box['x2'] * width) / full_width, 'y2': (top + box['y2'] * height) / full_height} for box in boxes]

def merge_proposals(boxes):
    result = []
    for box in sorted(boxes, key=lambda b: -b['score']):
        area = (box['x2'] - box['x1']) * (box['y2'] - box['y1'])
        duplicate = False
        for old in result:
            if old['class_id'] != box['class_id']:
                continue
            intersection = max(0, min(box['x2'], old['x2']) - max(box['x1'], old['x1'])) * max(0, min(box['y2'], old['y2']) - max(box['y1'], old['y1']))
            old_area = (old['x2'] - old['x1']) * (old['y2'] - old['y1'])
            threshold = 0.55 if box['class_id'] == 0 else 0.65
            if intersection / max(area + old_area - intersection, 1e-12) > threshold:
                duplicate = True
                break
        if not duplicate:
            result.append(box)
        if len(result) >= 300:
            break
    return result

class ONNXProposer:

    def __init__(self, model_path, device_id=0, confidence=0.25, tiles=True, *, strict=False):
        from directml_backend import DirectMLDetector
        self.model = DirectMLDetector(Path(model_path), device_id)
        if self.model.names != {0: 'Head', 1: 'Body'}:
            raise ValueError('Annotation model must use classes exactly 0=Head, 1=Body')
        self.confidence, self.tiles = (confidence, tiles)
        self.strict = strict

    def _pass(self, frame):
        from directml_backend import prepare_frame, decode_output
        tensor, geometry = prepare_frame(frame, self.model.input_size)
        output = self.model.session.run(None, {self.model.input_name: tensor})[0]
        if getattr(self, 'strict', False):
            raw = np.asarray(output)
            valid_shape = raw.ndim == 3 and raw.shape[0] == 1 and (min(raw.shape) > 0) and (raw.shape[2] == 6 if self.model.end_to_end else raw.shape[1] == 4 + len(self.model.names))
            if not valid_shape or not np.issubdtype(raw.dtype, np.floating) or (not np.isfinite(raw).all()):
                raise ValueError('Strict model output must be finite floating-point detections with a supported nonempty shape')
        decoded = decode_output(output, geometry, self.model.names, end_to_end=self.model.end_to_end, confidence=self.confidence, head_iou=0.55, return_scores=True)
        return [{'class_id': 0 if name == 'Head' else 1, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'score': score} for x1, y1, x2, y2, name, label, score in decoded]

    def __call__(self, frame):
        from directml_backend import _plausible_head
        height, width = frame.shape[:2]
        proposals = self._pass(frame)
        if self.tiles:
            tile_width, tile_height = (min(384, width), min(320, height))
            regions = {(x, y, tile_width, tile_height) for x in (0, width - tile_width) for y in (0, height - tile_height)}
            for x, y, w, h in sorted(regions):
                boxes = [b for b in self._pass(frame[y:y + h, x:x + w]) if b['class_id'] == 0 and min(b['x1'], b['y1'], 1 - b['x2'], 1 - b['y2']) > 0.008]
                proposals.extend(map_boxes(boxes, (x, y, w, h), frame.shape))
        bodies = [[b[k] for k in ('x1', 'y1', 'x2', 'y2')] for b in proposals if b['class_id'] == 1]
        proposals = [b for b in proposals if b['class_id'] != 0 or _plausible_head([b[k] for k in ('x1', 'y1', 'x2', 'y2')], bodies, width, height)]
        return merge_proposals(proposals)

def review_flags(boxes, frame_shape):
    height, width = frame_shape[:2]
    flags = set()
    heads = [b for b in boxes if b['class_id'] == 0]
    bodies = [b for b in boxes if b['class_id'] == 1]
    if not boxes:
        flags.add('no_detections')
    for box in boxes:
        if box['score'] < 0.5:
            flags.add('low_confidence')
        if min(box['x1'], box['y1'], 1 - box['x2'], 1 - box['y2']) < 0.005:
            flags.add('crop_edge')
    for head in heads:
        if min((head['x2'] - head['x1']) * width, (head['y2'] - head['y1']) * height) < 12:
            flags.add('small_head')
        if head['x1'] <= 0.5 + 6 / width and head['x2'] >= 0.5 - 6 / width and (head['y1'] <= 0.5 + 6 / height) and (head['y2'] >= 0.5 - 6 / height):
            flags.add('crosshair_area')
    from annotation_assist import pair_parts
    pairs, ambiguous = pair_parts(boxes)
    if len(pairs) < len(bodies):
        flags.add('body_without_head')
    if len(pairs) < len(heads):
        flags.add('head_without_body')
    if ambiguous:
        flags.add('ambiguous_parts')
    return sorted(flags)

def _open_video(path):
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    if not capture.isOpened():
        capture.release()
        raise ValueError(f'Cannot open video with the bundled FFmpeg decoder: {path}')
    return capture

def inspect_video(path):
    capture = _open_video(path)
    try:
        width, height = (round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps, frames = (capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or not math.isfinite(frames) or fps <= 0 or (frames <= 0):
            raise ValueError('Video has no reliable frame rate/duration metadata')
        return {'width': width, 'height': height, 'fps': fps, 'frame_count': round(frames), 'duration_seconds': frames / fps}
    finally:
        capture.release()

def _reserve_run(parent):
    parent = Path(parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    for number in range(1, 10000):
        candidate = parent / ('recording-data' if number == 1 else f'recording-data-{number}')
        if candidate.resolve().parent != parent:
            raise ValueError('Output path leaves the selected output directory')
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise ValueError('Choose another output directory; too many existing recording runs')

def _write_manifest(run, manifest):
    temporary = run / 'manifest.tmp'
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, run / 'manifest.json')

def _label_text(boxes):
    return ''.join((f"{box['class_id']} {(box['x1'] + box['x2']) / 2:.8f} {(box['y1'] + box['y2']) / 2:.8f} {box['x2'] - box['x1']:.8f} {box['y2'] - box['y1']:.8f}\n" for box in boxes))

def _processing_lock(run):
    stream = (run / 'prepare.lock').open('a+b')
    if stream.seek(0, 2) == 0:
        stream.write(b'0')
        stream.flush()
    stream.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        stream.close()
        raise ValueError('This recording run is already being processed') from error
    return stream

def _restore_run(run, source, settings, times):
    manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or manifest.get('status') not in ('running', 'failed', 'interrupted'):
        raise ValueError('Only unfinished runs can resume; complete/reviewed runs are preserved')
    for key in ('path', 'size_bytes', 'width', 'height', 'fps', 'frame_count', 'duration_seconds'):
        if manifest.get('source', {}).get(key) != source[key]:
            raise ValueError('Resume video differs from the original source')
    old_mtime = manifest['source'].get('mtime_ns')
    if old_mtime is not None and old_mtime != source['mtime_ns']:
        raise ValueError('The source recording changed after preparation started')
    if manifest.get('settings') != settings:
        raise ValueError('Resume requires the same model, sampling and split settings')
    identifiers = set()
    for frame in manifest['frames']:
        identifier = frame['id']
        if not identifier.startswith('frame-') or not identifier[6:].isdigit():
            raise ValueError('Invalid frame ID in resume checkpoint')
        number = int(identifier[6:])
        if identifier != f'frame-{number:06d}' or not 1 <= number <= len(times) or identifier in identifiers:
            raise ValueError('Invalid/duplicate frame ID in resume checkpoint')
        identifiers.add(identifier)
        if frame.get('status') != 'pending' or frame.get('revision') != 0 or frame.get('empty_confirmed'):
            raise ValueError('Resume will not replace reviewed annotations')
        if abs(frame['time_seconds'] - times[number - 1]) > 1e-06:
            raise ValueError('Resume timestamps do not match the selected sample schedule')
        for key, expected in (('image', f'images/{identifier}.png'), ('label', f'draft-labels/{identifier}.txt')):
            if frame.get(key) != expected or not (run / expected).resolve().is_relative_to(run):
                raise ValueError('Invalid frame path in resume checkpoint')
            if not (run / expected).is_file():
                raise ValueError(f'Missing checkpoint file: {expected}')
    last_saved = max((int(f['id'][6:]) for f in manifest['frames']), default=0)
    cursor = manifest.get('next_sample_index', last_saved)
    if type(cursor) is not int or not last_saved <= cursor <= len(times):
        raise ValueError('Invalid progress cursor in resume checkpoint')
    manifest['next_sample_index'] = cursor
    manifest['statistics']['saved'] = len(manifest['frames'])
    manifest['statistics']['decoded'] = len(manifest['frames']) + manifest['statistics']['duplicates']
    manifest['resume_count'] = manifest.get('resume_count', 0) + 1
    manifest['status'] = 'running'
    manifest.pop('error', None)
    return manifest

def prepare_video(video_path, output_parent, model_path, *, sample_fps=1, max_images=600, view='focus', val_fraction=0.2, gap_seconds=5, start=0, end=None, device_id=0, confidence=0.25, tiles=True, proposer=None, resume=None, progress_callback=None):
    from directml_backend import detection_view, validate_capture_size
    cv2.setNumThreads(1)
    video_path, model_path = (Path(video_path).resolve(), Path(model_path).resolve())
    if not video_path.is_file() or not model_path.is_file():
        raise ValueError('Video and ONNX model paths must both refer to existing files')
    source = inspect_video(video_path)
    validate_capture_size(source['width'], source['height'])
    end = source['duration_seconds'] if end is None else min(end, source['duration_seconds'])
    times = sample_times(start, end, sample_fps, max_images)
    temporal_split(start, start, end, val_fraction, gap_seconds)
    if not 0 < confidence <= 1 or device_id < 0:
        raise ValueError('Invalid confidence or DirectML device')
    if view != 'focus':
        raise ValueError('Dataset review currently requires the centered 640x448 focus view')
    existing_parent = Path(output_parent).resolve()
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    if shutil.disk_usage(existing_parent).free < len(times) * 1500000 + 50000000:
        raise ValueError('Insufficient free disk space for the selected PNG budget')
    source = {'path': str(video_path), 'size_bytes': video_path.stat().st_size, 'mtime_ns': video_path.stat().st_mtime_ns, **source}
    settings = {'sample_fps': sample_fps, 'max_images': max_images, 'view': view, 'start_seconds': start, 'end_seconds': end, 'val_fraction': val_fraction, 'gap_seconds': gap_seconds, 'model': str(model_path), 'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(), 'confidence': confidence, 'head_detail_tiles': tiles, 'decoder_threads': 1}
    if resume is not None:
        run = Path(resume).resolve(strict=True)
        lock = _processing_lock(run)
        try:
            manifest = _restore_run(run, source, settings, times)
        except BaseException:
            lock.close()
            raise
    else:
        run = _reserve_run(output_parent)
        lock = _processing_lock(run)
        (run / 'images').mkdir()
        (run / 'draft-labels').mkdir()
        manifest = {'schema_version': 1, 'status': 'running', 'source': source, 'settings': settings, 'next_sample_index': 0, 'notice': 'All automatic annotations are unreviewed drafts. Model misses are not confirmed negatives. Train/validation use separate time blocks of one recording, not independent sessions.', 'statistics': {}, 'frames': []}
    stats = {'scheduled': len(times), 'decoded': 0, 'saved': 0, 'duplicates': 0, 'split_gap': 0, 'decode_failures': 0}
    stats.update(manifest['statistics'])
    manifest['statistics'] = stats
    _write_manifest(run, manifest)
    print(f"Output: {run}\nVideo: {source['width']}x{source['height']}, {source['duration_seconds']:.1f}s; {len(times)} scheduled samples; ONNX DirectML only.", flush=True)
    dedup, capture = (FrameDeduplicator(), None)
    started = time.perf_counter()
    previous_elapsed = stats.get('elapsed_seconds', 0)

    def report_progress():
        if progress_callback is not None:
            progress_callback({'run_path': str(run), 'processed': manifest['next_sample_index'], 'total': len(times), 'saved': stats['saved']})
    try:
        report_progress()
        for item in manifest['frames']:
            existing = cv2.imread(str(run / item['image']))
            if existing is None or existing.shape[:2] != (448, 640):
                raise ValueError(f"Cannot restore saved image: {item['image']}")
            dedup.accept(existing)
        if proposer is None:
            from model_inference import create_proposer
            proposer = create_proposer(model_path, device_id, confidence, tiles)
        capture = _open_video(video_path)
        cursor = manifest['next_sample_index']
        if cursor:
            print(f"Resuming at sample {cursor + 1}/{len(times)}; preserving {len(manifest['frames'])} saved frames.", flush=True)
        for index in range(cursor, len(times)):
            timestamp = times[index]
            split = temporal_split(timestamp, start, end, val_fraction, gap_seconds)
            if split is None:
                stats['split_gap'] += 1
                manifest['next_sample_index'] = index + 1
                report_progress()
                continue
            if not capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000):
                raise ValueError(f'Decoder cannot seek to {timestamp:.2f}s')
            ok, frame = capture.read()
            if not ok:
                stats['decode_failures'] += 1
                manifest['next_sample_index'] = index + 1
                report_progress()
                continue
            stats['decoded'] += 1
            if frame.shape[:2] != (source['height'], source['width']):
                raise ValueError('Video changes resolution; split it into fixed-resolution clips first')
            selected, region = detection_view(frame, view)
            if not dedup.accept(selected):
                stats['duplicates'] += 1
                manifest['next_sample_index'] = index + 1
                report_progress()
                continue
            boxes = proposer(selected)
            identifier = f'frame-{index + 1:06d}'
            image_path, label_path = (f'images/{identifier}.png', f'draft-labels/{identifier}.txt')
            if not cv2.imwrite(str(run / image_path), selected, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise OSError(f'Could not save {run / image_path}')
            (run / label_path).write_text(_label_text(boxes), encoding='utf-8')
            flags = review_flags(boxes, selected.shape)
            sharpness = float(cv2.Laplacian(cv2.cvtColor(selected, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var())
            if sharpness < 15:
                flags.append('low_sharpness')
            manifest['frames'].append({'id': identifier, 'image': image_path, 'label': label_path, 'time_seconds': timestamp, 'decoded_time_seconds': capture.get(cv2.CAP_PROP_POS_MSEC) / 1000, 'split': split, 'width': selected.shape[1], 'height': selected.shape[0], 'source_region': list(region), 'boxes': boxes, 'flags': flags, 'sharpness': round(sharpness, 2), 'status': 'pending', 'revision': 0, 'empty_confirmed': False})
            stats['saved'] += 1
            manifest['next_sample_index'] = index + 1
            if stats['saved'] == 1 or stats['saved'] % 25 == 0:
                stats['elapsed_seconds'] = round(previous_elapsed + time.perf_counter() - started, 2)
                _write_manifest(run, manifest)
                counts = Counter((f['split'] for f in manifest['frames']))
                print(f"Processed {index + 1}/{len(times)} samples; saved {stats['saved']} (train {counts['train']}, val {counts['val']}), duplicates {stats['duplicates']}, elapsed {time.perf_counter() - started:.0f}s", flush=True)
            report_progress()
        if not manifest['frames']:
            raise ValueError('No usable frames were decoded in the selected time range')
        manifest['status'] = 'complete'
    except BaseException as error:
        manifest['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        manifest['error'] = str(error)
        raise
    finally:
        if capture is not None:
            capture.release()
        stats['elapsed_seconds'] = round(previous_elapsed + time.perf_counter() - started, 2)
        _write_manifest(run, manifest)
        lock.close()
    print(f'''Complete: {stats['saved']} draft images. Review with:\n  review_data.bat "{run}"''', flush=True)
    return run

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--output', type=Path, default=PROJECT / 'dataset-runs')
    parser.add_argument('--model', type=Path, default=PROJECT / 'best.onnx')
    parser.add_argument('--sample-fps', type=float, default=1)
    parser.add_argument('--max-images', type=int, default=600)
    parser.add_argument('--view', choices=('focus',), default='focus')
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--gap-seconds', type=float, default=5)
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--end', type=float)
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--confidence', type=float, default=0.25)
    parser.add_argument('--tiles', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--inspect', action='store_true', help='Read video metadata only; no model load or output')
    parser.add_argument('--resume', type=Path, help='Continue an unfinished run with exactly the same source/settings')
    args = parser.parse_args(argv)
    try:
        if args.inspect:
            print(json.dumps(inspect_video(args.video), indent=2))
            return 0
        prepare_video(args.video, args.output, args.model, sample_fps=args.sample_fps, max_images=args.max_images, view=args.view, val_fraction=args.val_fraction, gap_seconds=args.gap_seconds, start=args.start, end=args.end, device_id=args.device_id, confidence=args.confidence, tiles=args.tiles, resume=args.resume)
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f'Data preparation failed: {error}', file=sys.stderr)
        return 2
if __name__ == '__main__':
    raise SystemExit(main())
