"""Distinguish a visible running reversal from an unsupported walk between runs."""
import numpy as np

from activity_rules import runs


# Only overwrite labels that lacked independent walking evidence before the
# R8 recoil classifier. A low-recoil fit alone cannot confirm walking in a turn.
WEAK_STEP = {'translation_with_step_motion_but_unresolved_gait'}
TRUNK_AND_FEET = [5, 6, 11, 12, 15, 16]


def _number(row, key):
    value = row.get(key)
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def _weak_walk(row):
    evidence = row.get('gait_evidence')
    if evidence == 'person_local_cycle_morphology':
        evidence = row.get('pre_cycle_gait_evidence')
    return (evidence in WEAK_STEP and not row.get('action_event_id')
            and not row.get('stair_sequence_supported')
            and row.get('pre_cycle_gait_activity') not in ('standing', 'squat', 'stairs_up', 'stairs_down'))


def _world_xy(t, pixels, camera, start, end):
    indices = np.asarray(camera['source_indices'], int)
    first = int(np.searchsorted(indices, start, side='right')) - 1
    last = int(np.searchsorted(indices, end - 1))
    if first < 0 or last >= len(indices) or not np.all(camera['valid'][first + 1:last + 1]):
        return None
    # The affine at entry j maps the previous camera sample into sample j.
    transform = np.eye(3)
    inverse = [transform.copy()]
    for matrix in camera['matrices'][first + 1:last + 1]:
        if not np.isfinite(matrix).all() or np.linalg.det(matrix[:, :2]) <= 0:
            return None
        transform = np.vstack((matrix, [0., 0., 1.])) @ transform
        inverse.append(np.linalg.inv(transform))
    coeff = np.asarray(inverse)[:, :2].reshape(-1, 6)
    if 'time_s' in camera:
        times = np.asarray(camera['time_s'],float)[first:last+1]
    else:
        if indices[first]<0 or indices[last]>=len(t):
            return None
        times = t[indices[first:last + 1]]
    if not np.isfinite(times).all() or (np.diff(times)<=0).any():
        return None
    if t[start] < times[0]-1e-6 or t[end-1] > times[-1]+1e-6:
        return None
    aligned = np.stack([np.interp(t[start:end], times, x)
                        for x in coeff.T], axis=1).reshape(-1, 2, 3)
    return np.einsum('nij,nkj->nki', aligned[:, :, :2], pixels) + aligned[:, None, :, 2]


def _trend(t, p):
    x = t - t.mean()
    slope = x @ (p - p.mean(axis=0)) / (x @ x)
    residual = p - p.mean(axis=0) - x[:, None] * slope
    # Compare net movement to this person's local pose jitter, not to a
    # preset movement speed, stride length, or absolute pixel distance.
    noise = np.median(np.linalg.norm(residual, axis=1))
    return slope * (t[-1] - t[0]), noise


def _reversal(t, p, a, b, step):
    approach, na = _trend(t[:a], p[:a])
    departure, nb = _trend(t[b:], p[b:])
    if approach @ departure >= 0:
        return None
    axis = approach - departure
    axis /= np.linalg.norm(axis)
    if approach @ axis <= 2 * na or -departure @ axis <= 2 * nb:
        return None
    projected = p @ axis
    # Median filtering over part of the observed step removes contact jitter.
    median_path = np.array([np.median(projected[np.abs(t - x) <= step / 4]) for x in t])
    peak = int(np.argmax(median_path))
    if not a <= peak < b:
        return None
    if (median_path[peak] - median_path[0] <= 2 * na
            or median_path[peak] - median_path[-1] <= 2 * nb):
        return None
    return axis, float(t[peak])


def apply(t, pose, valid, rows, measured, camera):
    labels = np.array([row['activity'] for row in rows], object)
    is_running = labels == 'running'
    if not is_running.any():
        return
    ends = np.asarray([row['frame_end_s'] for row in rows])
    raw_step = np.array([_number(row, 'personal_gait_step_period_s') for row in rows])
    rates = np.array([_number(row, 'gait_step_hz_candidate') for row in rows])
    missing = ~np.isfinite(raw_step) & np.isfinite(rates) & (rates > 0)
    raw_step[missing] = 1 / rates[missing]
    supported_run = is_running & np.array([
        not (r.get('gait_inferred') or r.get('ambiguous_gait') or r.get('locomotion_partial')
             or r.get('action_hypotheses') or r.get('action_event_id')) for r in rows], bool)
    reliable = measured & valid[:, TRUNK_AND_FEET].all(axis=1)
    run_spans = list(runs(is_running & reliable))
    preceding = {b: a for a, b in run_spans}
    following = {a: b for a, b in run_spans}
    for a, b in runs(np.isin(labels, ('walking', 'gait_walk'))):
        if a not in preceding or b not in following or not reliable[a:b].all():
            continue
        if not all(_weak_walk(row) for row in rows[a:b]):
            continue
        # Bounds are local gait-cycle counts, not universal times/frequencies.
        # Each anchor must span at least a step. A separate walking interval
        # longer than three adjacent running steps is not handled by this rule.
        left = slice(preceding[a], a)
        right = slice(b, following[b])
        periods = []
        for span in (left, right):
            span_steps = raw_step[span][supported_run[span]]
            span_steps = span_steps[np.isfinite(span_steps) & (span_steps > 0)]
            if len(span_steps):
                periods.append(float(np.median(span_steps)))
        if len(periods) != 2:
            continue
        pre, post = periods
        step = (pre + post) / 2
        if ends[b - 1] - t[a] > 3 * step:
            continue
        start = max(left.start, int(np.searchsorted(ends, t[a] - 1.5 * pre, side='right')))
        end = min(right.stop, int(np.searchsorted(t, ends[b - 1] + 1.5 * post)))
        weights = ends - t
        if (a - start < 3 or end - b < 3 or np.max(weights[start:end]) > step / 2
                or (weights[start:a] * supported_run[start:a]).sum() < pre
                or (weights[b:end] * supported_run[b:end]).sum() < post):
            continue
        p = pose[start:end, :, :2]
        centers = np.stack((p[:, [11, 12]].mean(axis=1), p[:, [15, 16]].mean(axis=1)), axis=1)
        world = _world_xy(t, centers, camera, start, end)
        if world is None:
            continue
        signals = [_reversal(t[start:end], world[:, j], a - start, b - start, step) for j in range(2)]
        if any(s is None for s in signals) or signals[0][0] @ signals[1][0] <= 0:
            continue
        pivot = (signals[0][1] + signals[1][1]) / 2
        for row in rows[a:b]:
            row.update(pre_turn_activity=row['activity'], pre_turn_gait_evidence=row.get('gait_evidence', ''),
                       activity='running_turn', activity_reason='same_track_running_anchors_with_compensated_hip_and_feet_reversal',
                       gait_evidence='running_reversal_transition', turn_pivot_video_s=pivot,
                       turn_reference_step_s=step, action_hypotheses=(), ambiguous_gait=0,
                       gait_inferred=0, locomotion_partial=0)
