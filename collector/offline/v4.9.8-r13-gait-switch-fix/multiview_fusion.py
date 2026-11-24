from __future__ import annotations
import json
import math
from pathlib import Path
KEYS = ('completion', 'foot_motion', 'knee_tracking_proxy', 'lateral_control', 'depth', 'hip_knee_coordination', 'trunk_control', 'indirect_control')
FRONT_ONLY = {'knee_tracking_proxy', 'lateral_control'}
SIDE_ONLY = {'depth', 'hip_knee_coordination'}
GRADE = {3: 'GOOD', 2: 'MID', 1: 'POOR'}
ZH = {3: '\u597d', 2: '\u4e2d', 1: '\u5dee'}

def load_config(path=None):
    cfg = json.loads((Path(__file__).parent / 'multiview_config.json').read_text(encoding='utf-8'))
    if path:
        update = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if set(update) - set(cfg):
            raise ValueError('Unknown fusion options: ' + str(sorted(set(update) - set(cfg))))
        cfg.update(update)
    for key in ('maximum_bottom_difference_s', 'minimum_interval_iou', 'overlap_duplicate_iou', 'maximum_duplicate_bottom_difference_s', 'default_split_ratio', 'minimum_pair_fraction_when_three_each', 'edge_context_window_s', 'edge_moving_speed_leg_s'):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(value)) or (value <= 0):
            raise ValueError('Invalid fusion setting: ' + key)
    if not 0 < cfg['minimum_interval_iou'] <= cfg['overlap_duplicate_iou'] <= 1:
        raise ValueError('Interval IoU settings must satisfy 0 < match <= duplicate <= 1.')
    if not 0.2 <= cfg['default_split_ratio'] <= 0.8:
        raise ValueError('Split ratio must be in [0.2, 0.8].')
    if cfg['minimum_pair_fraction_when_three_each'] > 1:
        raise ValueError('Minimum pair fraction must not exceed one.')
    if cfg['indirect_maximum_level'] != 1 or cfg['require_stable_view_for_fusion'] is not True:
        raise ValueError('Indirect cues must be capped at MID; stable-view evidence is required.')
    expected = {'shared_criterion_policy': 'maximum_supported_severity_once', 'boundary_policy': 'best_observed_view_side_on_tie', 'missing_criterion_policy': 'exclude_and_report_count'}
    for (key, value) in expected.items():
        if cfg[key] != value:
            raise ValueError('Unsupported fusion policy: ' + key)
    return cfg

def number(row, name, default=0.0):
    try:
        value = float(row.get(name, default))
        return value if math.isfinite(value) else default
    except (ValueError, TypeError):
        return default

def criteria(row):
    value = row.get('observable_criteria_json', {}) if row else {}
    return json.loads(value) if isinstance(value, str) else value

def supported_criteria(row, view):
    if not row or number(row, 'video_view_stable') != 1:
        return {}
    if row.get('video_view_hint') != view + '_or_oblique':
        return {}
    result = {}
    for (key, original) in criteria(row).items():
        if key not in KEYS or (key in FRONT_ONLY and view != 'front') or (key in SIDE_ONLY and view != 'side'):
            continue
        level = original.get('level')
        if level not in (0, 1, 2) or original.get('status') != ('PASS', 'WARN', 'FAIL')[level]:
            continue
        item = dict(original, view=view)
        if key == 'indirect_control':
            item['level'] = min(1, level)
            item['status'] = ('PASS', 'WARN')[item['level']]
        result[key] = item
    return result

def combine(front=None, side=None):
    evidence = {'front': supported_criteria(front, 'front'), 'side': supported_criteria(side, 'side')}
    merged = {}
    for key in KEYS:
        items = [cs[key] for cs in evidence.values() if key in cs]
        level = max((c['level'] for c in items), default=None)
        merged[key] = {'level': level, 'status': ('PASS', 'WARN', 'FAIL')[level] if level is not None else 'UNAVAILABLE', 'evidence': items}
    observed = sum((c['level'] is not None for c in merged.values()))
    worst = max((c['level'] for c in merged.values() if c['level'] is not None), default=None)
    code = 3 - worst if worst is not None else None
    reasons = []
    for (key, item) in merged.items():
        if item['level']:
            triggers = [e for e in item['evidence'] if e['level']]
            detail = ', '.join((f"{e['view']}={e.get('value')} {e.get('unit', '')}" for e in triggers))
            reasons.append(f"{key}: {item['status']} ({detail})")
    return {'grade_code': code, 'grade': GRADE.get(code, ''), 'grade_zh': ZH.get(code, ''), 'observed_items': observed, 'possible_items': len(KEYS), 'criteria': merged, 'grade_reasons': '; '.join(reasons) if reasons else f'All {observed} observed items passed' if observed else 'No observable scoring evidence'}

def interval_iou(a, b):
    overlap = max(0.0, min(a['end_video_s'], b['end_video_s']) - max(a['start_video_s'], b['start_video_s']))
    union = max(a['end_video_s'], b['end_video_s']) - min(a['start_video_s'], b['start_video_s'])
    return overlap / union if union > 0 else 0.0

def match_repetitions(front, side, cfg):
    (n, m) = (len(front), len(side))
    scores = [[(0, 0.0) for _ in range(m + 1)] for _ in range(n + 1)]
    choices = {}
    for i in range(n + 1):
        for j in range(m + 1):
            if not (i or j):
                continue
            candidates = []
            if i:
                candidates.append((scores[i - 1][j], 'front'))
            if j:
                candidates.append((scores[i][j - 1], 'side'))
            if i and j:
                (f, s) = (front[i - 1], side[j - 1])
                delta = abs(f['bottom_video_s'] - s['bottom_video_s'])
                overlap = interval_iou(f, s)
                if delta <= cfg['maximum_bottom_difference_s'] and overlap >= cfg['minimum_interval_iou']:
                    prior = scores[i - 1][j - 1]
                    candidates.append(((prior[0] + 1, prior[1] - delta - 0.2 * (1 - overlap)), 'both'))
            (score, action) = max(candidates, key=lambda x: (x[0], x[1] == 'both'))
            (scores[i][j], choices[i, j]) = (score, action)
    pairs = []
    (i, j) = (n, m)
    while i or j:
        action = choices[i, j]
        if action == 'both':
            pairs.append((front[i - 1], side[j - 1]))
            i -= 1
            j -= 1
        elif action == 'front':
            pairs.append((front[i - 1], None))
            i -= 1
        else:
            pairs.append((None, side[j - 1]))
            j -= 1
    pairs.reverse()
    return pairs

def boundary_quality(event, view):
    row = event.get('_grade', {})
    supported = supported_criteria(row, view)
    coverage = number(row, 'side_leg_pose_coverage' if view == 'side' else 'bilateral_pose_coverage')
    gap = number(row, 'video_largest_gap_s', math.inf)
    boundary_flags = sum((bool(number(event, key)) for key in ('side_onset_needs_review', 'body_boundary_needs_review')))
    return (bool(supported), -boundary_flags, coverage, number(row, 'video_pose_coverage'), -gap, view == 'side')

def fuse_events(front, side, cfg=None, *, labels_only=False):
    cfg = cfg or load_config()
    front = sorted(front, key=lambda x: x['start_video_s'])
    side = sorted(side, key=lambda x: x['start_video_s'])
    pairs = match_repetitions(front, side, cfg)
    matched = sum((a is not None and b is not None for (a, b) in pairs))
    if min(len(front), len(side)) >= 3 and matched / min(len(front), len(side)) < cfg['minimum_pair_fraction_when_three_each']:
        raise ValueError('Too few temporally matched repetitions; confirm synchronized panels of the same performance.')
    (records, diagnostics) = ([], [])
    for (f, s) in pairs:
        available = [(x, view) for (x, view) in ((f, 'front'), (s, 'side')) if x is not None]
        (chosen, source) = max(available, key=lambda x: boundary_quality(*x))
        outcome = combine(f.get('_grade') if f else None, s.get('_grade') if s else None)
        if not labels_only and (not outcome['observed_items']):
            diagnostics.append({'kind': 'zero_observed_items', 'front_rep_id': f['rep_id'] if f else None, 'side_rep_id': s['rep_id'] if s else None})
            continue
        row = {key: chosen[key] for key in ('start_video_s', 'end_video_s', 'bottom_video_s')}
        row.update({k: v for (k, v) in outcome.items() if k != 'criteria'})
        row.update(boundary_source=source, view_match='both' if f and s else 'front_only' if f else 'side_only', criteria_json=json.dumps(outcome['criteria'], ensure_ascii=False, allow_nan=False), duration_s=row['end_video_s'] - row['start_video_s'])
        for (view, item) in (('front', f), ('side', s)):
            row[view + '_rep_id'] = int(item['rep_id']) if item else None
            sub = combine(item.get('_grade') if item and view == 'front' else None, item.get('_grade') if item and view == 'side' else None)
            row[view + '_grade'] = sub['grade']
            row[view + '_observed_items'] = sub['observed_items']
            for key in ('start_video_s', 'end_video_s', 'bottom_video_s'):
                row[view + '_' + key] = item[key] if item else None
            row['_' + view] = item
        row['view_bottom_difference_s'] = abs(f['bottom_video_s'] - s['bottom_video_s']) if f and s else None
        row['view_interval_iou'] = interval_iou(f, s) if f and s else None
        records.append(row)
    records.sort(key=lambda x: x['start_video_s'])
    selected = []
    for row in records:
        duplicate = next((other for other in reversed(selected) if interval_iou(row, other) >= cfg['overlap_duplicate_iou'] and abs(row['bottom_video_s'] - other['bottom_video_s']) <= cfg['maximum_duplicate_bottom_difference_s']), None)
        if duplicate is not None:
            raise ValueError(f"Ambiguous overlapping detections failed strict matching; confirm synchronized panels. Bottoms: {duplicate['bottom_video_s']:.3f}s, {row['bottom_video_s']:.3f}s")
        else:
            selected.append(row)
    selected.sort(key=lambda x: x['bottom_video_s'])
    for (a, b) in zip(selected, selected[1:]):
        if a['end_video_s'] > b['start_video_s']:
            lower = max(a['start_video_s'], a['bottom_video_s'], b['start_video_s'])
            upper = min(a['end_video_s'], b['bottom_video_s'], b['end_video_s'])
            if lower > upper:
                raise ValueError('Inconsistent overlapping repetitions; source views may not be synchronized.')
            boundary = min(max((a['bottom_video_s'] + b['bottom_video_s']) / 2, lower), upper)
            diagnostics.append({'kind': 'shared_intercycle_boundary', 'boundary_s': boundary})
            a['end_video_s'] = b['start_video_s'] = boundary
    for (rid, row) in enumerate(selected, 1):
        row['rep_id'] = rid
        row['duration_s'] = row['end_video_s'] - row['start_video_s']
        if not row['start_video_s'] <= row['bottom_video_s'] < row['end_video_s']:
            raise ValueError('Fused bottom lies outside its half-open interval.')
    return (selected, diagnostics)
