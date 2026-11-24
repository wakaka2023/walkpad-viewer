import numpy as np


def _step_signature(t, xy, valid, scale):
    joints = (5, 6, 7, 8, 9, 10, 11, 12, 15, 16)
    if len(t) < 3 or not valid[:, joints].all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        return None
    xy = np.asarray(xy, dtype=np.float64)
    shoulder = xy[:, [5, 6]].mean(axis=1)
    hip = xy[:, [11, 12]].mean(axis=1)
    torso = (shoulder + hip) / 2
    mid = int(np.searchsorted(t, (t[0] + t[-1]) / 2))
    if np.dot(torso[mid] - torso[0], torso[-1] - torso[mid]) <= 0:
        return None
    phase = (t - t[0]) / (t[-1] - t[0])
    unit = np.median(scale)

    def residual(y):
        return (y - (y[0] + phase * (y[-1] - y[0]))) / unit

    hs, ss = residual(hip[:, 1]), residual(shoulder[:, 1])
    if np.dot(hs - hs.mean(), ss - ss.mean()) <= 0:
        return None
    a = xy[:, [5, 6]] - xy[:, [7, 8]]
    b = xy[:, [9, 10]] - xy[:, [7, 8]]
    arm_len = np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2)
    if not np.all(arm_len > 0):
        return None
    elbow = np.sum(a * b, axis=2) / arm_len
    ankles = np.linalg.norm(xy[:, 15] - xy[:, 16], axis=1) / scale
    return np.array([min(np.ptp(hs), np.ptp(ss)), np.quantile(ankles, .9),
                     np.median(elbow.max(axis=1))]), torso[-1] - torso[0]


def detect(t, pose, valid, scale, batches, walked, ran, unresolved):
    result = np.zeros(len(t), dtype=bool)
    if not (walked.any() and ran.any() and unresolved.any()):
        return result
    groups = [[], []]
    candidates = []
    for grid, steps in batches:
        previous = None
        for (j, prev), (k, side), (m, nxt) in zip(steps, steps[1:], steps[2:]):
            if prev == side or side == nxt:
                previous = None
                continue
            lo, hi = np.searchsorted(t, [(grid[j] + grid[k]) / 2, (grid[k] + grid[m]) / 2])
            s = slice(lo, hi)
            desc = _step_signature(t[s], pose[s, :, :2], valid[s], scale[s])
            if desc is None:
                previous = None
                continue
            v, heading = desc
            walk_count, run_count = walked[s].sum(), ran[s].sum()
            if 2 * walk_count > hi - lo and not run_count:
                groups[0].append(v)
            elif 2 * run_count > hi - lo and not walk_count:
                groups[1].append(v)
            elif not (walk_count or run_count) and unresolved[s].any():
                if previous is not None and np.dot(previous, heading) > 0:
                    candidates.append((s, v))
            previous = heading
    if min(map(len, groups)) < 2:
        return result
    walk, run = (np.median(g, axis=0) for g in groups)
    if not np.all(run > walk):
        return result
    max_walk_flexion = np.max(np.asarray(groups[0])[:, 2])
    for s, v in candidates:
        above_walk = v[2] - max_walk_flexion > 128 * np.finfo(float).eps
        if above_walk and np.all(np.abs(v[:2] - run[:2]) < np.abs(v[:2] - walk[:2])):
            result[s] = unresolved[s]
    return result
