from __future__ import annotations
import math
from pathlib import Path
import cv2
import numpy as np
COORDS = ('x1', 'y1', 'x2', 'y2')

def iou(a, b):
    intersection = max(0.0, min(a['x2'], b['x2']) - max(a['x1'], b['x1'])) * max(0.0, min(a['y2'], b['y2']) - max(a['y1'], b['y1']))
    area = lambda box: (box['x2'] - box['x1']) * (box['y2'] - box['y1'])
    return intersection / max(area(a) + area(b) - intersection, 1e-12)

def relation(head, body):
    bw, bh = (body['x2'] - body['x1'], body['y2'] - body['y1'])
    return np.array([(head['x2'] - head['x1']) / bw, (head['y2'] - head['y1']) / bh, ((head['x1'] + head['x2']) / 2 - (body['x1'] + body['x2']) / 2) / bw, (head['y1'] - body['y1']) / bh])

def pair_parts(boxes):
    from assignment import _assign
    heads = [i for i, b in enumerate(boxes) if b['class_id'] == 0]
    bodies = [i for i, b in enumerate(boxes) if b['class_id'] == 1]
    costs = np.full((len(heads), len(bodies)), 2.0, dtype=float)
    for h, hi in enumerate(heads):
        for b, bi in enumerate(bodies):
            wr, hr, dx, dy = relation(boxes[hi], boxes[bi])
            if 0.12 <= wr <= 0.85 and 0.055 <= hr <= 0.3 and (abs(dx) <= 0.43) and (-0.13 <= dy <= 0.16):
                costs[h, b] = 0.35 * abs(dx) / 0.43 + 0.25 * abs(dy) / 0.16 + 0.2 * abs(hr - 0.15) / 0.15
    valid = costs < 1
    ambiguous = set()
    for h, b in zip(*np.where(valid)):
        if valid[h].sum() > 1 or valid[:, b].sum() > 1:
            ambiguous.update((heads[h], bodies[b]))
    pairs = [(heads[h], bodies[b]) for h, b in _assign(costs)] if costs.size else []
    return (pairs, ambiguous)

def learn_profile(frames, exclude_id=None):
    samples = []
    sample_frames = []
    frame_ids = set()
    for frame in frames:
        if frame.get('id') == exclude_id or frame.get('status') != 'approved' or frame.get('split') != 'train':
            continue
        boxes = frame['boxes']
        pairs, ambiguous = pair_parts(boxes)
        for hi, bi in pairs:
            if hi in ambiguous or bi in ambiguous:
                continue
            h, b = (boxes[hi], boxes[bi])
            if min(*(h[k] for k in ('x1', 'y1')), 1 - h['x2'], 1 - h['y2'], *(b[k] for k in ('x1', 'y1')), 1 - b['x2'], 1 - b['y2']) < 0.01:
                continue
            if any((i not in (hi, bi) and other['class_id'] == 1 and (iou(other, b) > 0.1) for i, other in enumerate(boxes))):
                continue
            samples.append(relation(h, b))
            sample_frames.append(frame['id'])
            frame_ids.add(frame['id'])
    if len(samples) < 8 or len(frame_ids) < 4:
        return None
    values = np.asarray(samples)
    median = np.median(values, axis=0)
    mad = np.median(abs(values - median), axis=0)
    keep = (abs(values - median) <= np.maximum(3 * 1.4826 * mad, [0.07, 0.035, 0.08, 0.04])).all(axis=1)
    values = values[keep]
    if len(values) < 8 or len({sample_frames[i] for i in np.flatnonzero(keep)}) < 4:
        return None
    median = np.median(values, axis=0)
    spread = np.median(abs(values - median), axis=0)
    if np.any(spread > [0.13, 0.045, 0.12, 0.065]):
        return None
    return dict(ratios=median.tolist(), pairs=len(values))

def _estimate(box, reason, evidence, identifier):
    return {**{k: float(box[k]) for k in COORDS}, 'class_id': box['class_id'], 'estimate_id': identifier, 'reason': reason, 'evidence': evidence}

def _inside(box):
    return all((math.isfinite(box[k]) for k in COORDS)) and 0.005 < box['x1'] < box['x2'] < 0.995 and (0.005 < box['y1'] < box['y2'] < 0.995)

def geometry_estimates(boxes, profile):
    if profile is None:
        return []
    pairs, ambiguous = pair_parts(boxes)
    paired = {i for pair in pairs for i in pair}
    wr, hr, dx, dy = profile['ratios']
    result = []
    for i, anchor in enumerate(boxes):
        if i in paired or i in ambiguous or (not _inside(anchor)):
            continue
        if anchor['class_id'] == 1:
            bw, bh = (anchor['x2'] - anchor['x1'], anchor['y2'] - anchor['y1'])
            hw, hh = (bw * wr, bh * hr)
            hx = (anchor['x1'] + anchor['x2']) / 2 + dx * bw
            hy = anchor['y1'] + dy * bh
            guess = dict(class_id=0, x1=hx - hw / 2, y1=hy, x2=hx + hw / 2, y2=hy + hh)
        else:
            hw, hh = (anchor['x2'] - anchor['x1'], anchor['y2'] - anchor['y1'])
            if min(hw * 640, hh * 448) < 10:
                continue
            bw, bh = (hw / wr, hh / hr)
            bx = (anchor['x1'] + anchor['x2']) / 2 - dx * bw
            by = anchor['y1'] - dy * bh
            guess = dict(class_id=1, x1=bx - bw / 2, y1=by, x2=bx + bw / 2, y2=by + bh)
        if not _inside(guess) or min((guess['x2'] - guess['x1']) * 640, (guess['y2'] - guess['y1']) * 448) < 4:
            continue
        if any((j != i and (iou(other, guess) > 0.12 or (other['class_id'] == anchor['class_id'] and iou(other, anchor) > 0.12)) for j, other in enumerate(boxes))):
            continue
        result.append(_estimate(guess, 'Missing Head' if guess['class_id'] == 0 else 'Missing Body', f"Proportions from {profile['pairs']} isolated reviewed training pairs; position is estimated.", f'geometry-{i}'))
        if len(result) >= 24:
            break
    return result

def patch_translation(source, target, anchor, radius=40):
    if source.shape != target.shape:
        return None
    before = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    after = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    height, width = before.shape
    x1, y1, x2, y2 = [anchor[k] * (width if k[0] == 'x' else height) for k in COORDS]
    bw, bh = (x2 - x1, y2 - y1)
    pw, ph = (min(20, round(bw * 0.32)), min(20, round(bh * 0.15)))
    if pw < 5 or ph < 5:
        return None
    offsets = []
    for fx, fy in ((0.3, 0.25), (0.7, 0.25), (0.3, 0.5), (0.7, 0.5), (0.5, 0.78)):
        x, y = (round(x1 + fx * bw - pw / 2), round(y1 + fy * bh - ph / 2))
        if x < 0 or y < 0 or x + pw > width or (y + ph > height):
            continue
        patch = before[y:y + ph, x:x + pw]
        if patch.std() < 12:
            continue
        sx, sy = (max(0, x - radius), max(0, y - radius))
        ex, ey = (min(width, x + pw + radius), min(height, y + ph + radius))
        scores = cv2.matchTemplate(after[sy:ey, sx:ex], patch, cv2.TM_CCOEFF_NORMED)
        _, peak, _, where = cv2.minMaxLoc(scores)
        mx, my = where
        if not math.isfinite(peak) or peak < 0.83 or mx in (0, scores.shape[1] - 1) or (my in (0, scores.shape[0] - 1)):
            continue
        alternative = scores.copy()
        alternative[max(0, my - 3):my + 4, max(0, mx - 3):mx + 4] = -1
        if peak - float(alternative.max()) < 0.09:
            continue
        offsets.append((sx + mx - x, sy + my - y))
    if len(offsets) < 3:
        return None
    values = np.asarray(offsets)
    agreement = np.max(abs(values[:, None, :] - values[None, :, :]), axis=2) <= 2
    best = int(np.argmax(agreement.sum(axis=1)))
    matched = values[agreement[best]]
    if len(matched) < 3:
        return None
    dx, dy = np.rint(np.median(matched, axis=0)).astype(int)
    return (int(dx), int(dy), len(matched))

def _combine(boxes, additions):
    result = list(boxes)
    for box in additions:
        if not any((other['class_id'] == box['class_id'] and iou(other, box) > 0.55 for other in result)):
            result.append(box)
    return result

def focused_rescan(proposer, pixels, boxes, anchors):
    if not callable(getattr(proposer, '_pass', None)):
        return (list(boxes), 0)
    from prepare_dataset import map_boxes
    context = _combine(anchors, boxes)
    pairs, _ = pair_parts(context)
    paired = {i for pair in pairs for i in pair}
    result = list(boxes)
    passes = 0
    for i, anchor in enumerate(context):
        if i in paired or passes >= 8:
            continue
        aw, ah = (anchor['x2'] - anchor['x1'], anchor['y2'] - anchor['y1'])
        cx = (anchor['x1'] + anchor['x2']) / 2
        if anchor['class_id'] == 1:
            rect = (cx - aw * 0.8, anchor['y1'] - ah * 0.16, cx + aw * 0.8, anchor['y1'] + ah * 0.5)
        else:
            rect = (cx - aw * 2.5, anchor['y1'] - ah * 0.6, cx + aw * 2.5, anchor['y1'] + ah * 8)
        x1, y1, x2, y2 = [max(0, min(size, round(value * size))) for value, size in zip(rect, (640, 448, 640, 448))]
        if x2 - x1 < 24 or y2 - y1 < 24:
            continue
        detected = proposer._pass(pixels[y1:y2, x1:x2])
        passes += 1
        mapped = map_boxes(detected, (x1, y1, x2 - x1, y2 - y1), pixels.shape)
        accepted = []
        for candidate, local in zip(mapped, detected):
            if candidate['class_id'] == anchor['class_id']:
                continue
            test = [anchor, candidate]
            matches, _ = pair_parts(test)
            if matches and all((0.01 < local[k] < 0.99 for k in COORDS)):
                accepted.append(candidate)
        result = _combine(result, accepted)
    return (result, passes)

def load_neighbors(data, frame, current, *, strict=False):
    from prepare_dataset import _open_video
    source = data.get('source', {})
    try:
        path = Path(source['path'])
        stat = path.stat()
        if stat.st_size != source['size_bytes'] or ('mtime_ns' in source and stat.st_mtime_ns != source['mtime_ns']):
            return ([], 'Source video changed; temporal recovery was skipped.')
        region = frame['source_region']
        if not isinstance(region, list) or len(region) != 4 or any((type(v) is not int for v in region)) or (region[2:] != [640, 448]):
            return ([], 'Source crop is unavailable; temporal recovery was skipped.')
        x, y, w, h = region
        timestamp = frame['time_seconds']
        if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0:
            return ([], 'Source timestamp is unavailable; temporal recovery was skipped.')
    except (KeyError, TypeError, OSError, ValueError):
        return ([], 'Original video metadata is unavailable; temporal recovery was skipped.')
    try:
        capture = _open_video(path)
    except (ValueError, OSError, cv2.error):
        return ([], 'Original video could not be decoded; temporal recovery was skipped.')
    spacing = 0.08
    if strict:
        fps = source.get('fps')
        if type(fps) not in (int, float) or not math.isfinite(fps) or fps < 3:
            capture.release()
            return ([], 'Reliable nearby source frame indexes are unavailable for strict screening.')
        spacing = max(0.08, 1.01 / fps)

    def read_at(seconds):
        if not capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000):
            return None
        ok, pixels = capture.read()
        if not ok or pixels.shape[:2] != (source['height'], source['width']):
            return None
        crop = pixels[y:y + h, x:x + w]
        position = capture.get(cv2.CAP_PROP_POS_FRAMES)
        index = round(position) - 1 if math.isfinite(position) and position >= 1 else None
        return (crop, index) if crop.shape == current.shape else None
    try:
        reference = read_at(timestamp)
        if reference is None or not np.array_equal(reference[0], current):
            return ([], 'Saved image does not match the original source timestamp; temporal recovery was skipped.')
        reference_index = reference[1]
        if strict and reference_index is None:
            return ([], 'The source decoder did not provide reliable frame indexes.')
        result = []
        for offset in (-spacing, spacing):
            if not 0 <= timestamp + offset < source['duration_seconds']:
                continue
            decoded = read_at(timestamp + offset)
            if decoded is not None:
                pixels, index = decoded
                if strict and (index is None or (offset < 0 and index >= reference_index) or (offset > 0 and index <= reference_index)):
                    return ([], 'Distinct predecessor and successor source frames could not be verified.')
                result.append(pixels)
        if len(result) != 2:
            return ([], 'Both neighboring video frames are needed for temporal recovery.')
        return (result, None)
    finally:
        capture.release()

def temporal_estimates(proposer, pixels, neighbors, current_boxes):
    if len(neighbors) != 2:
        return []
    candidates = []
    for neighbor in neighbors:
        detected = proposer(neighbor)
        pairs, ambiguous = pair_parts(detected)
        side = []
        for hi, bi in pairs[:24]:
            if hi in ambiguous or bi in ambiguous:
                continue
            head, body = (detected[hi], detected[bi])
            moved = patch_translation(neighbor, pixels, body)
            if moved is None:
                continue
            dx, dy, support = moved
            anchored = False
            for part in (body, head):
                shifted = {**part, **{key: part[key] + (dx / 640 if key[0] == 'x' else dy / 448) for key in COORDS}}
                matches = [b for b in current_boxes if b['class_id'] == part['class_id'] and iou(b, shifted) > 0.5]
                if len(matches) != 1:
                    continue
                match = matches[0]
                observed_x = (match['x1'] + match['x2'] - (part['x1'] + part['x2'])) * 320
                observed_y = (match['y1'] + match['y2'] - (part['y1'] + part['y2'])) * 224
                if max(abs(observed_x - dx), abs(observed_y - dy)) <= 2.5:
                    anchored = True
                    break
            if not anchored:
                continue
            for part in (head, body):
                guess = {**part, **{key: part[key] + (dx / 640 if key[0] == 'x' else dy / 448) for key in COORDS}}
                if _inside(guess):
                    side.append((guess, support))
        candidates.append(side)
    result = []
    for first, support in candidates[0]:
        compatible = [(second, count) for second, count in candidates[1] if second['class_id'] == first['class_id'] and iou(first, second) >= 0.65]
        if len(compatible) != 1:
            continue
        second, count = compatible[0]
        if sum((other['class_id'] == second['class_id'] and iou(other, second) >= 0.65 for other, _ in candidates[0])) != 1:
            continue
        guess = {k: (first[k] + second[k]) / 2 for k in COORDS} | {'class_id': first['class_id']}
        if any((b['class_id'] == guess['class_id'] and iou(b, guess) > 0.25 for b in current_boxes + result)):
            continue
        result.append(_estimate(guess, 'Part recovered from nearby frames', f'Before/after frames and a visible target box agree; {min(support, count)} local patches support motion. Hidden part remains estimated.', f'temporal-{len(result)}'))
        if len(result) >= 24:
            break
    return result

def separate_washed_out_heads(pixels, boxes):
    actual, estimates = ([], [])
    height, width = pixels.shape[:2]
    for i, box in enumerate(boxes):
        if box['class_id'] == 0:
            x1, y1, x2, y2 = [max(0, min(size, round(box[key] * size))) for key, size in zip(COORDS, (width, height, width, height))]
            crop = pixels[y1:y2, x1:x2]
            white = float((crop.min(axis=2) > 220).mean()) if crop.size else 0
            if white >= 0.45:
                estimates.append(_estimate(box, 'Head pixels are washed out', f'{white:.0%} of this detector box is nearly white. Inspect the location; a flash may hide the head.', f'washed-out-{i}'))
                continue
        actual.append(box)
    return (actual, estimates)

def assist_proposals(proposer, pixels, boxes, data, frame):
    from prepare_dataset import merge_proposals
    anchors, _ = separate_washed_out_heads(pixels, frame['boxes'])
    actual, passes = focused_rescan(proposer, pixels, boxes, anchors)
    actual = merge_proposals(actual)
    actual, washed_out = separate_washed_out_heads(pixels, actual)
    context = _combine(anchors, actual)
    pairs, ambiguous = pair_parts(context)
    warnings = []
    if washed_out:
        warnings.append('Washed-out head predictions were moved to individual estimates; brightness alone cannot confirm a head.')
    if ambiguous:
        warnings.append('Overlapping targets have ambiguous Head/Body relationships. No geometry estimates were made for those parts.')
    profile = learn_profile(data['frames'], exclude_id=frame['id'])
    if profile is None:
        warnings.append('Not enough consistent isolated reviewed training pairs for proportional estimates (8 pairs across 4 frames required).')
    paired = {i for pair in pairs for i in pair}
    recoverable = any((i not in paired and i not in ambiguous for i in range(len(context))))
    neighbors = []
    if recoverable:
        neighbors, notice = load_neighbors(data, frame, pixels)
        if notice:
            warnings.append(notice)
    estimates = temporal_estimates(proposer, pixels, neighbors, context)
    for estimate in washed_out + geometry_estimates(context, profile):
        if not any((b['class_id'] == estimate['class_id'] and iou(b, estimate) > 0.25 for b in estimates)):
            estimates.append(estimate)
    return dict(boxes=actual, estimates=estimates[:24], warnings=warnings, assistance=dict(pairs=len(pairs), ambiguous_parts=len(ambiguous), focused_passes=passes, profile_pairs=profile['pairs'] if profile else 0, temporal_frames=len(neighbors)))
