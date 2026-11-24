"""Track-local gait recovery from repeated bilateral steps, even with no torso travel."""

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import find_peaks

from activity_rules import runs
from walk_continuity import _step


WALK = ('walking', 'gait_walk')
UNRESOLVED = ('unknown', 'body_motion')
EVIDENCE = 'same_track_repeated_bilateral_steps_match_observed_gait'
ONSET = 'bilateral_step_onset_before_repeated_walk_support'


def _lower(values):
    x = np.asarray(values)
    return float(np.quantile(x, .10) - np.median(np.abs(x - np.median(x))))


def _upper(values):
    x = np.asarray(values)
    return float(np.quantile(x, .90) + np.median(np.abs(x - np.median(x))))


def _steps(t, p, direct, step):
    """Find alternating extrema and learn their excursions from this track's anchor bouts."""
    xy = p[:, :, :2].astype(float)
    hips = xy[:, 11:13].mean(axis=1)
    unit = np.linalg.norm(xy[:, 5:7].mean(axis=1) - hips, axis=1)
    if np.any(unit <= 1):
        return [], None
    dt = np.median(np.diff(t))
    if not 0 < dt < step / 3:
        return [], None
    smoothing = max(3, int(round(step / (8 * dt))) | 1)
    feet = (xy[:, 15:17] - hips[:, None, :]) / median_filter(unit, size=smoothing)[:, None, None]
    feet = median_filter(feet, size=(smoothing, 1, 1))
    diff = feet[:, 0] - feet[:, 1]
    # Remove slow viewpoint/stance drift. The step interval just sets the observation
    # scale for this track and is never a cutoff separating walking from running.
    trend_span = max(3, int(round(4 * step / dt)) | 1)
    delta = diff - uniform_filter1d(diff, size=trend_span, axis=0, mode='nearest')
    sample = delta[direct]
    _, _, axes = np.linalg.svd(sample - np.median(sample, axis=0), full_matrices=False)
    axis = axes[0]
    signed = delta @ axis
    projected = feet @ axis
    # Reduce sensitivity to quantization and frame-to-frame detector jitter.
    error = np.abs(np.diff(diff @ axis, n=2))
    ref_error = error[direct[1:-1]]
    noise = max(1 / np.median(unit[direct]), _upper(ref_error))

    def peaks(floor):
        return sorted([(i, sign) for sign in (-1, 1)
                       for i in find_peaks(sign * signed, prominence=floor)[0]])

    def pairs(extrema):
        for (a, s), (b, q) in zip(extrema[:-1], extrema[1:]):
            span = t[b] - t[a]
            swing = q * (projected[b] - projected[a]) * [1, -1]
            amplitude = q * (signed[b] - signed[a])
            if s != q and span > 0 and swing.min() > 0 and amplitude > 0:
                yield a, b, span, amplitude, float(swing.min())

    preliminary = list(pairs(peaks(2 * noise)))
    # A sustained direct bout must cover a majority of each calibration handoff.
    # Short islands and inferred labels do not count as anchors.
    def anchored(a,b):
        return any(t[a + j - 1] - t[a + i] > (t[b] - t[a]) / 2
                   for i,j in runs(direct[a:b+1]))

    ref = np.array([(span, amp, swing) for a, b, span, amp, swing in preliminary
                    if anchored(a,b) and step / 2 <= span <= 2 * step and swing > noise])
    if len(ref) < 2:
        return [], None
    low_t = max(step / 2, _lower(ref[:, 0]) - dt)
    high_t = min(2 * step, _upper(ref[:, 0]) + dt)
    # A longer onset handoff may still merge into the next alternating handoffs.
    # Allow variation equal to the observed half-cycle asymmetry, not a global period.
    variability = float(np.ptp(ref[:, 0]) / 2)
    low_t = max(step / 2, low_t - variability)
    high_t = min(2 * step, high_t + variability)
    low_amp = max(2 * noise, _lower(ref[:, 1]) - 2 * noise)
    low_swing = max(noise, _lower(ref[:, 2]) - 2 * noise)
    accepted = [(a, b) for a, b, span, amp, swing in pairs(peaks(max(2 * noise, low_amp / 2)))
                if low_t <= span <= high_t and amp >= low_amp and swing >= low_swing
                and np.max(np.diff(t[a:b + 1])) <= low_t / 2]
    batches = []
    for a, b in accepted:
        if batches and batches[-1][-1] == a:
            batches[-1].append(b)
        else:
            batches.append([a, b])
    return [g for g in batches if len(g) >= 4], (signed, projected, low_amp, low_swing, high_t)


def apply(t, ends, pose, valid, rows, measured):
    """Recover a gait only from a connected run of steps with a measured class anchor."""
    raw = np.array([r['activity'] for r in rows], dtype=object)
    phase_ok = np.asarray(measured, bool) & valid[:, [5, 6, 11, 12, 13, 14, 15, 16]].all(axis=1)
    phase_ok &= np.array([not (r['action_event_id'] or r.get('stair_sequence_supported')
                               or r.get('locomotion_partial')) for r in rows], bool)
    for kind in ('gait_walk', 'running'):
        resolved = WALK if kind == 'gait_walk' else ('running',)
        repairable = (*UNRESOLVED, 'standing', *WALK) if kind == 'gait_walk' else UNRESOLVED
        hypotheses = np.array([all(h in resolved for h in r.get('action_hypotheses', ())) for r in rows])
        eligible = phase_ok & hypotheses & np.isin(raw, (*resolved, *repairable))
        is_class = np.isin(raw, resolved)
        direct = eligible & is_class & np.array([
            not (r.get('ambiguous_gait') or r.get('action_hypotheses') or r.get('gait_inferred'))
            for r in rows], bool)
        weak = eligible & np.isin(raw, repairable) & ~direct
        if not direct.any() or not weak.any():
            continue
        periods = np.array([_step(r) for r in rows])
        for s, e in runs(eligible):
            if not weak[s:e].any():
                continue
            ref_periods = periods[s:e][direct[s:e] & np.isfinite(periods[s:e])]
            if not len(ref_periods):
                continue
            step = float(np.median(ref_periods))
            anchors = direct[s:e].copy()
            for a, b in runs(anchors):
                if ends[s+b-1] - t[s+a] < step:
                    anchors[a:b] = False
            if anchors.sum() < 8:
                continue
            runs_of_steps, details = _steps(t[s:e], pose[s:e], anchors, step)
            for g in runs_of_steps:
                l, r = g[0] + s, g[-1] + s + 1
                if not any(anchors[a:b + 1].all() for a,b in zip(g[:-1], g[1:])):
                    continue
                if kind == 'running':
                    # For running, only repair within a running bout, not at a transition.
                    good = np.flatnonzero(anchors[l-s:r-s]) + l
                    l, r = good[0], good[-1] + 1
                elif l > s and raw[l-1] == 'standing' and not direct[s:l].any():
                    # The partial first step can have weaker evidence than a full handoff.
                    # Trace backwards only while observed motion has departed from the
                    # measured neutral position, never by a preset duration in the video.
                    signed, projected, excursion, limb, span = details
                    a, b = g[:2]
                    direction = np.sign(signed[a] - signed[b])
                    mid = (signed[a] + signed[b]) / 2
                    start = max(0, np.searchsorted(t[s:e], t[l] - span))
                    onset = (signed[start:a] - mid) * direction >= excursion / 4
                    # Both ankles must have advanced away from the pre-step pose.
                    bilateral = (projected[start:a] - projected[start]) * (direction * np.array([1, -1]))
                    onset &= bilateral.min(axis=1) > limb / 4
                    for x, y in runs(onset):
                        if y == a - start:
                            for i in range(s + start + x, l):
                                if weak[i] and raw[i] == 'standing':
                                    _update(rows[i], kind, step, ONSET, inferred=True)
                for i in np.flatnonzero(weak[l:r]) + l:
                    _update(rows[i], kind, step, EVIDENCE)
                    # Do not let an inferred step seed a different class later in this pass.
                    phase_ok[i] = False


def _update(row, kind, step, reason, *, inferred=False):
    row.update(pre_bilateral_gait_activity=row['activity'], pre_bilateral_gait_evidence=row.get('gait_evidence',''),
               activity=kind, activity_reason=reason, gait_evidence='bilateral_cycle_continuity',
               personal_gait_step_period_s=step, gait_step_hz_candidate=1 / step,
               action_hypotheses=(), ambiguous_gait=0, locomotion_partial=0, gait_inferred=int(inferred))
