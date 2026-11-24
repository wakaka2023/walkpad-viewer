"""Use observed leg dynamics at clipped walk entrances and supported squat edges."""

import numpy as np
from scipy.ndimage import median_filter

from .activity_rules import runs
from .running_turn import _world_xy
from .walk_continuity import _step


def _dispersion(x, axis=0):
    return np.median(np.abs(x - np.median(x, axis=axis)), axis=axis)


def _weak(row):
    return (row['activity'] in ('unknown', 'body_motion', 'stepping', 'fast_stepping')
            and not (row['action_event_id'] or row.get('stair_sequence_supported') or row.get('action_hypotheses')))


def _candidate(row, kind, reason, *, gait=False):
    row.update(pre_leg_boundary_activity=row['activity'], pre_leg_boundary_reason=row['activity_reason'],
               activity=kind, activity_reason=reason, boundary_inferred=1)
    if gait:
        row.update(gait_evidence='clipped_bilateral_thigh_walk_context', locomotion_partial=1, gait_inferred=1)


def recover_clipped_walk(t, ends, pose, valid, rows, measured, camera, height):
    """A later measured walk can support a continuous cropped onset, but not a missing track."""
    weak = np.array([_weak(r) for r in rows])
    if not weak.any():
        return
    seen_legs = measured & valid[:, 11:15].all(axis=1)
    direct = seen_legs & np.array([r['activity'] in ('gait_walk', 'walking') and not
                                  (r['action_event_id'] or r.get('gait_inferred') or r.get('locomotion_partial')
                                   or r.get('ambiguous_gait') or r.get('action_hypotheses')) for r in rows])
    walk = np.array([r['activity'] in ('gait_walk','walking') and
                     not (r['action_event_id'] or any(h != 'walking' for h in r.get('action_hypotheses', ()))) for r in rows])
    for lo, hi in runs(seen_legs & (walk | weak)):
        for a, b in runs(direct[lo:hi]):
            a, b = a+lo, b+lo
            ps = np.array([_step(r) for r in rows[a:b]])
            if not np.isfinite(ps).any():
                continue
            step = float(np.nanmedian(ps))
            if ends[b-1]-t[a] < step/2:
                continue
            s = max(lo, np.searchsorted(t, t[a]-4*step))
            if a-s < 5 or not weak[s:a].any():
                continue
            e = min(hi, max(b, np.searchsorted(t, t[a]+step, side='right')))
            raw = pose[s:e, :, :2]
            v = valid[s:e]
            thigh_px = np.linalg.norm(raw[:, 13:15]-raw[:, 11:13], axis=2)
            outer_knee = raw[:, 13:15, 1].max(axis=1)
            clipped = (~v[:, 15:17].all(axis=1)) & (height-outer_knee < thigh_px.mean(axis=1)/2)
            if not clipped[:a-s].any():
                continue
            world = _world_xy(t, raw, camera, s, e)
            if world is None or np.any(thigh_px <= 2):
                continue
            q = np.log(np.linalg.norm(world[:, 13:15]-world[:, 11:13],axis=2))
            if not np.isfinite(q).all():
                continue
            asymmetry = median_filter(q[:,0]-q[:,1], size=3, mode='nearest')
            dt = np.median(np.diff(t[s:e]))
            err = max(2 / np.median(thigh_px), 2*_dispersion(np.diff(asymmetry, n=2)))
            # Asymmetric projected swing must emerge above detector jitter. Each
            # local window also has to recede on both thighs, excluding stationary
            # weight shifting and flip-flops in the tracker scale.
            hit = np.zeros(a-s, bool)
            progressing = hit.copy()
            for i in range(a-s):
                j = min(e-s, np.searchsorted(t[s:e], t[s+i]+1.5*step, side='right'))
                if j-i < 5 or np.max(np.diff(t[s+i:s+j])) > step/3:
                    continue
                x = t[s+i:s+j] - np.mean(t[s+i:s+j])
                y = q[i:j]-np.mean(q[i:j],axis=0)
                slope = x @ y / (x @ x)
                drift = slope * (x[-1]-x[0])
                scatter = np.median(np.abs(y-x[:,None]*slope), axis=0)
                progressing[i] = np.all(-drift > scatter)
                hit[i] = progressing[i] and np.ptp(asymmetry[i:j]) > err + dt*np.linalg.norm(slope)
            # A low-excursion phase can join only across measured scale progression.
            # It still must be within half a locally observed step of leg evidence.
            times = t[s:a]
            good = times[hit]
            if not len(good):
                continue
            near = (np.abs(times[:,None]-np.r_[good, t[a]][None,:]).min(axis=1) <= step/2)
            for l, h in runs(progressing & near):
                if not clipped[l:h].any() or not hit[l:h].any():
                    continue
                if h != a-s:
                    if t[a]-t[s+h-1] > step/2:
                        continue
                    left = max(l, np.searchsorted(times, t[s+h-1]-step/2))
                    right = min(e-s, np.searchsorted(t[s:e], t[a]+step/2, side='right'))
                    before, after = q[left:h], q[a-s:right]
                    # Ending at a step reversal may give a weak short local fit.
                    # The measured legs on both sides must still show recession.
                    if not len(before) or not len(after) or not np.all(np.median(before,axis=0) > np.median(after,axis=0)+err):
                        continue
                for i in np.flatnonzero(weak[s+l:a]) + s+l:
                    _candidate(rows[i], 'gait_walk', 'visible_thigh_motion_and_depth_progression_to_measured_walk', gait=True)
                    weak[i] = False


def recover_squat_edges(t, pose, valid, rows, measured, f, events, camera):
    """Use planted feet, upright reference and joint motion only next to confirmed reps."""
    if not events:
        return
    knee = f['knee_flexion']
    usable = measured & valid[:, [11, 12, 13, 14, 15, 16]].all(axis=1) & np.isfinite(knee)
    labels = np.array([r['activity'] for r in rows])
    weak = np.array([_weak(r) for r in rows])
    for ev in events:
        if ev['activity'] != 'squat':
            continue
        a, p, b = ev['start_video_frame']-1, ev['peak_video_frame']-1, ev['end_video_frame']
        if not (0 <= a < p < b < len(t)):
            continue
        # Context scale depends on the observed descent/ascent of this repetition.
        lo = max(0, np.searchsorted(t, t[a] - 2*(t[p]-t[a])))
        hi = min(len(t), np.searchsorted(t, t[b] + 2*(t[b]-t[p]), side='right'))
        world = _world_xy(t, pose[lo:hi, :, :2], camera, lo, hi)
        if world is None:
            continue
        feet = world[:,15:17]
        reach = world[:,11:13,1].mean(axis=1) - feet[:,:,1].mean(axis=1)
        stance = feet[a-lo:b-lo][usable[a:b]]
        if len(stance) < 6:
            continue
        center = np.median(stance, axis=0)
        deviation = np.linalg.norm(stance-center, axis=2).max(axis=1)
        tolerance = max(2., float(np.quantile(deviation,.95)+3*_dispersion(deviation)))
        planted = np.linalg.norm(feet-center, axis=2).max(axis=1) <= tolerance
        upright = (labels[lo:hi]=='standing') & usable[lo:hi]
        if upright.sum() < 4:
            continue
        ref_k = knee[lo:hi][upright]
        ref_h = reach[upright]
        k0 = np.quantile(ref_k,.9) + max(_dispersion(ref_k), 1.)
        h0 = np.quantile(ref_h,.9) + max(_dispersion(ref_h), 2.)
        signal = (knee[lo:hi] > k0) & (reach > h0)
        for edge, stop, direction in ((a-1, lo-1, -1), (b, hi, 1)):
            tail = []
            for i in range(edge, stop, direction):
                if not usable[i] or not weak[i]:
                    break
                tail.append(i)
            # An active edge has to touch the established repetition and move away
            # from it toward neutral in both measured signals. No new rep is counted.
            active_count = 0
            for i in tail:
                j = i - direction
                k_adv = knee[j] - knee[i]
                h_adv = reach[j-lo] - reach[i-lo]
                # At the very edge one joint signal can return to its upright
                # range first; its measured motion must still agree with the rise/descent.
                moving = signal[i-lo] or ((knee[i] > k0 or reach[i-lo] > h0)
                                          and k_adv > max(_dispersion(ref_k), 1.)
                                          and h_adv > max(_dispersion(ref_h), 2.))
                if not moving or not planted[i-lo]:
                    break
                active_count += 1
            if active_count:
                inside = edge - direction
                last = tail[active_count - 1]
                if knee[last] <= knee[inside] and reach[last-lo] <= reach[inside-lo]:
                    for i in tail[:active_count]:
                        _candidate(rows[i],'squat','planted_feet_with_knee_hip_motion_at_confirmed_squat_edge')
                        weak[i] = False
                else:
                    active_count = 0
            # Stand only when this neutral fringe really reaches a quiet standing
            # anchor, and neither the knees nor hip height still indicate the squat.
            if tail:
                after = tail[-1] + direction
                if lo <= after < hi and upright[after-lo]:
                    for i in reversed(tail[active_count:]):
                        if (signal[i-lo] or knee[i] > k0 + 2*_dispersion(ref_k) or reach[i-lo] > h0 + 2*_dispersion(ref_h)
                                or np.linalg.norm(feet[i-lo]-feet[after-lo], axis=1).max() > tolerance):
                            break
                        _candidate(rows[i],'standing','measured_upright_planted_recovery_adjacent_to_standing')
                        weak[i] = False


def recover_walk_handoffs(t, ends, pose, valid, rows, measured, camera):
    """Bridge a same-walk gap only when a foot actually advances through it."""
    weak = np.array([_weak(r) for r in rows])
    if not weak.any():
        return
    feet_seen = measured & valid[:,15:17].all(axis=1)
    direct = feet_seen & np.array([r['activity'] in ('gait_walk','walking') and not
                                  (r['action_event_id'] or r.get('ambiguous_gait') or r.get('action_hypotheses')
                                   or r.get('gait_inferred') or r.get('locomotion_partial')) for r in rows])
    for a, b in runs(weak & feet_seen):
        if a == 0 or b == len(t) or not direct[a-1] or not direct[b] or b-a < 5:
            continue
        p0,p1 = _step(rows[a-1]),_step(rows[b])
        if not np.isfinite([p0,p1]).all():
            continue
        step = (p0 + p1) / 2
        if t[b]-t[a] > step or np.max(np.diff(t[a-1:b+1])) > step/3:
            continue
        # Two direct anchors are required. One may be short, but it cannot be
        # an isolated frame, and the other must cover at least half a step.
        l,r = a,b
        while l > 0 and direct[l-1]: l -= 1
        while r < len(t) and direct[r]: r += 1
        if min(a-l,r-b) < 3 or max(t[a]-t[l],ends[r-1]-t[b]) < step/2:
            continue
        world = _world_xy(t, pose[a:b,15:17,:2], camera, a, b)
        if world is None:
            continue
        same = np.sum((world[1:]-world[:-1])**2,axis=(1,2))
        swapped = np.sum((world[1:]-world[:-1,::-1])**2,axis=(1,2))
        if np.count_nonzero(same > swapped) > 1:
            continue
        smooth = median_filter(world,size=(3,1,1),mode='nearest')
        noise = np.median(np.linalg.norm(np.diff(world,n=2,axis=0),axis=2),axis=0)
        n = max(2, len(world)//3)
        movement = np.linalg.norm(np.median(smooth[-n:],axis=0)-np.median(smooth[:n],axis=0),axis=1)
        if not np.any(movement > np.maximum(2.,2*noise)):
            continue
        for i in range(a,b):
            _candidate(rows[i],'gait_walk','same_track_observed_foot_progression_through_walk_gap')
