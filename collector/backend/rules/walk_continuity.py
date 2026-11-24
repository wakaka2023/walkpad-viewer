"""Recover short weak walking spans with measured forward/back depth progression."""
import numpy as np

from .activity_rules import runs
from .adaptive_cycle_gait import _fit
from .running_turn import _world_xy

REASON = 'same_track_walk_with_alternating_legs_and_compensated_depth_progression'


def _step(row):
    p = row.get('personal_gait_step_period_s')
    hz = row.get('gait_step_hz_candidate')
    if p is not None and np.isfinite(p) and p > 0:
        return float(p)
    return float(1 / hz) if hz is not None and np.isfinite(hz) and hz > 0 else np.nan


def _trend(t, v):
    x = t - t.mean()
    centered = v - v.mean()
    slope = (x @ centered) / (x @ x)
    return slope * (t[-1] - t[0]), np.median(np.abs(centered - slope * x))


def _progress(t, log_scale, a, b, step):
    pre, n0 = _trend(t[:a], log_scale[:a])
    post, n1 = _trend(t[b:], log_scale[b:])
    direction = np.sign(pre)
    if abs(pre) <= 2 * n0 or direction * post <= 2 * n1:
        return False
    # Require progression throughout the inner span, not just endpoint growth.
    # Camera compensated torso scale is a depth proxy, not physical stride speed.
    span = t[b] - t[a]
    cuts = np.searchsorted(t, np.linspace(t[a], t[b], max(2, round(span / step) + 1)))
    for s, e in zip(cuts[:-1], cuts[1:]):
        if e - s < 3:
            return False
        trend, jitter = _trend(t[s:e + 1], log_scale[s:e + 1])
        if direction * trend <= 2 * jitter:
            return False
    return True


def bridges(t, ends, labels, rows, inferred, protected, reviews, measured, pose, valid, camera):
    result = np.full(len(t), '', dtype=object)
    # These rows already report body motion, but none of the recognized gait
    # types. Do not reclassify quiet unknowns, rest, other actions or lost tracks.
    moving = (labels == 'unknown') & np.array([
        row['activity'] == 'body_motion' and row.get('gait_evidence') == 'gait_type_unresolved'
        and not row.get('action_hypotheses') for row in rows], bool)
    if not moving.any():
        return result
    walking = (labels == 'walking') & np.array([
        row['activity'] in ('gait_walk', 'walking') and rv['action_candidates'] in ('', 'walking')
        for row, rv in zip(rows, reviews)], bool)
    reliable = measured & valid[:, [5, 6, 11, 12, 15, 16]].all(axis=1) & ~protected & np.array([
        not (row['action_event_id'] or row.get('stair_sequence_supported') or row.get('locomotion_partial')) for row in rows], bool)
    direct = walking & ~inferred & np.array([r['activity_status'] == 'supported' for r in reviews], bool)
    dt = ends - t
    for start, end in runs(reliable & (walking | moving)):
        if not moving[start:end].any():
            continue
        anchors = []
        for l, r in runs(direct[start:end]):
            l, r = l + start, r + start
            ps = np.array([_step(row) for row in rows[l:r]])
            if not np.isfinite(ps).any():
                continue
            step = float(np.nanmedian(ps))
            # Islands shorter than half an observed step are not anchor bouts.
            if ends[r - 1] - t[l] >= step / 2:
                anchors.append((l, r, step))
        for (l, a, pre), (b, r, post) in zip(anchors[:-1], anchors[1:]):
            if b <= a or not moving[a:b].any():
                continue
            step = (pre + post) / 2
            # At most two local strides of uncertainty between substantiated bouts.
            if t[b] - t[a] > 4 * step:
                continue
            s = max(l, np.searchsorted(ends, t[a] - 2 * pre, side='right'))
            e = min(r, np.searchsorted(t, t[b] + 2 * post, side='left'))
            if a - s < 3 or e - b < 3 or np.max(dt[s:e]) > step / 2:
                continue
            xy = pose[s:e, :, :2]
            world = _world_xy(t, xy, camera, s, e)
            if world is None:
                continue
            torso = np.linalg.norm(world[:, 5:7].mean(1) - world[:, 11:13].mean(1), axis=1)
            if not np.all(torso > 1) or not _progress(t[s:e], np.log(torso), a - s, b - s, step):
                continue
            compensated = np.concatenate((world, pose[s:e, :, 2:3]), axis=2)
            fit = _fit(t[s:e], compensated, valid[s:e])
            if fit is None or fit['leg_amp'] <= fit['resolution']:
                continue
            # A meaningful alternate-foot model must fit a compatible cycle; its
            # spectral resolution comes from this observation span, not a cutoff in Hz.
            span = t[e - 1] - t[s]
            if abs(.5 / fit['period'] - .5 / step) > 1 / span:
                continue
            result[np.flatnonzero(moving[a:b]) + a] = REASON
    return result
