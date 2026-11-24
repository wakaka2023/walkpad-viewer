import numpy as np
from scipy.signal import find_peaks

from activity_rules import runs, smooth_valid
from front_onset_rules import _fit
from view_body_rules import side_features

def early_lower_body_support(f, start, onset, bottom, args):
    details, supported = {}, False
    for name, group in zip(('hip', 'knee'), f['groups'][2:4]):
        pre = group[start:onset+1, 0]
        full = group[start:bottom+1, 0]
        if not np.isfinite(pre).all() or not np.isfinite(full).all():
            continue
        displacement = float(np.linalg.norm(pre[-1]-pre[0]))
        spread = float(np.linalg.norm(np.ptp(pre, axis=0)))
        full_spread = float(np.linalg.norm(np.ptp(full, axis=0)))
        limit = max(args.side_onset_torso_guard, .10*full_spread)
        details.update({f'side_early_{name}_motion_leg': spread,
                        f'side_early_{name}_limit_leg': limit})
        supported |= spread > limit and displacement >= .75*spread
    details['side_early_lower_body_supported'] = int(supported)
    return supported, details

def supported_torso_onset(f, event, args, lo, hi, old, body_onset, weights):
    t, k, scale = f['t'], f['base']['k'], f['base']['scale']
    shoulder, hip = (5, 11) if f['base']['geometry_side']=='left' else (6, 12)
    bottom = int(event['bottom_video_frame'])-1
    torso = (k[:, shoulder, 0]-k[:, hip, 0])/scale
    direction = np.sign(np.median(torso[max(lo, bottom-1):bottom+1])-np.median(torso[lo:min(lo+3, hi)]))
    x, y = t[lo:hi], direction*torso[lo:hi]
    details = dict(side_torso_status='insufficient_raw_torso_evidence')
    if direction == 0 or not np.isfinite(y).all() or hi-lo < 8:
        return old, details
    first = max(old, int(np.searchsorted(t, t[lo]+args.side_onset_min_context)))
    candidates = []
    for onset in range(first, min(body_onset, hi-4)+1):
        loss, coef = _fit(x, y, t[onset], weights)
        if coef[2] > 0 and abs(coef[1]) <= args.side_onset_pretrend_max:
            candidates.append((loss, onset, coef))
    if not candidates:
        details['side_torso_status'] = 'no_supported_torso_change_point'
        return old, details
    loss, onset, coef = min(candidates, key=lambda item:item[0])
    old_loss, _ = _fit(x, y, t[old], weights)
    gain = (old_loss-loss)/max(old_loss, 1e-12)
    near = [i for value, i, _ in candidates if value <= loss*1.15+1e-6]
    details.update(side_torso_fit_rmse_leg=float(np.sqrt(loss)),
                   side_torso_fit_relative_gain=float(gain),
                   side_torso_pretrend_leg_s=float(coef[1]),
                   side_torso_onset_frame=onset+1,
                   side_torso_model_first_frame=min(near)+1,
                   side_torso_model_last_frame=max(near)+1)
    if np.sqrt(loss) > max(.015, .10*float(np.ptp(y))):
        details['side_torso_status'] = 'poor_raw_torso_fit'
        return old, details
    if t[min(near)]-t[old] <= max(.035, float(np.median(np.diff(t)))) or gain < .05:
        details['side_torso_status'] = 'earlier_torso_onset_supported'
        return old, details
    details['side_torso_status'] = 'supported_raw_torso_change_point'
    return onset, details

def observed_step_boundary(f, event, lo, args):
    t, k, scale = f['t'], f['base']['k'], f['base']['scale']
    c = int(event['hip_core_start_video_frame']) - 1
    end = int(event['hip_core_end_video_frame'])
    dt = float(np.median(np.diff(t)))
    boundary, drivers = lo, []
    for foot in (15, 16):
        valid = np.isfinite(k[:, foot]).all(axis=1) & (k[:, foot, 2] >= args.keypoint_conf)
        xy, _ = smooth_valid(t, k[:, foot, :2], valid, args.smooth_seconds)
        core_valid = valid[c:end] & np.isfinite(xy[c:end]).all(axis=1)
        if core_valid.sum() < 3:
            continue
        anchor = np.median(xy[c:end][core_valid], axis=0)
        distance = np.linalg.norm(xy - anchor, axis=1) / scale
        limit = max(args.side_onset_foot_motion,
                    float(np.max(distance[c:end][core_valid])) + .025)
        moving = valid & np.isfinite(distance) & (distance > limit)
        for a, b in runs(moving[lo:c]):
            a, b = a + lo, b + lo
            if t[b-1] - t[a] + dt < args.side_onset_step_hold:
                continue
            if np.any(np.diff(t[a:b]) > max(.25, 3*dt)):
                continue
            if b > boundary:
                boundary, drivers = b, [foot]
            elif b == boundary:
                drivers.append(foot)
    return int(boundary), drivers

def grounded_descent_onset(f, event, args, previous_end=0, fraction=.25):
    t, k, scale = f['t'], f['base']['k'], f['base']['scale']
    old = int(event['start_video_frame']) - 1
    core = int(event['hip_core_start_video_frame']) - 1
    bottom = int(event['bottom_video_frame']) - 1
    details = dict(side_onset_status='original_onset_retained',
                   side_onset_previous_video_frame=old+1,
                   side_onset_previous_video_s=float(t[old]),
                   side_onset_needs_review=0)

    def retain(status, review=True):
        details.update(side_onset_status=status, side_onset_needs_review=int(review))
        return old, details

    if core <= old:
        return retain('no_precore_extension', False)
    lo = max(previous_end, int(np.searchsorted(t, t[old]-.3)))
    dt = float(np.median(np.diff(t)))
    if lo >= core or not f['valid'][lo:bottom+1].all():
        return retain('missing_visible_chain')
    if np.any(np.diff(t[lo:bottom+1]) > max(.25, 3*dt)):
        return retain('presentation_time_gap')

    lo, drivers = observed_step_boundary(f, event, lo, args)
    details.update(side_onset_support_start_frame=lo+1,
                   side_onset_support_start_s=float(t[lo]),
                   side_onset_step_ankles=','.join(map(str, drivers)))
    signal = f['body_top_signal']
    if core-lo < 3 or not np.isfinite(signal[lo:bottom+1]).all():
        return retain('insufficient_grounded_body_evidence')

    drop_estimate = float(signal[bottom] - np.quantile(signal[lo:core+1], .1))
    prominence = max(args.side_onset_reset_prominence, .05*drop_estimate)
    valleys, _ = find_peaks(-signal[lo:core+1], prominence=prominence)
    if len(valleys):
        lo += int(valleys[-1])
    details.update(side_onset_fit_start_frame=lo+1, side_onset_fit_start_s=float(t[lo]))
    top = float(np.quantile(signal[lo:core+1], .1))
    drop = float(signal[bottom] - top)
    crossing = np.flatnonzero(signal[core:bottom+1] >=
                             signal[core] + fraction*(signal[bottom]-signal[core]))
    hi = core + int(crossing[0]) + 1 if len(crossing) else bottom
    if hi-lo < 8 or hi-core < 3 or drop < .04:
        return retain('insufficient_descent_support')

    shoulder, hip, ankle = (5, 11, 15) if f['base']['geometry_side']=='left' else (6, 12, 16)
    raw = (.6*k[:, shoulder, 1] + .4*k[:, hip, 1] - k[:, ankle, 1]) / scale
    x, y = t[lo:hi], raw[lo:hi]
    if not np.isfinite(y).all():
        return retain('missing_raw_visible_chain')
    weights = np.gradient(x)
    candidates = []
    first = max(old, int(np.searchsorted(t, t[lo]+args.side_onset_min_context)))
    for onset in range(first, min(core, hi-4)+1):
        loss, coef = _fit(x, y, t[onset], weights)
        if coef[2] > 0 and abs(coef[1]) <= args.side_onset_pretrend_max:
            candidates.append((loss, onset, coef))
    if not candidates:
        return retain('no_supported_change_point')
    loss, onset, coef = min(candidates, key=lambda item:item[0])
    old_loss, _ = _fit(x, y, t[old], weights)
    gain = (old_loss-loss) / max(old_loss, 1e-12)
    near = [i for value, i, _ in candidates if value <= loss*1.15 + 1e-6]
    details.update(side_onset_fit_end_s=float(x[-1]), side_onset_fit_rmse_leg=float(np.sqrt(loss)),
                   side_onset_fit_relative_gain=float(gain), side_onset_pretrend_leg_s=float(coef[1]),
                   side_onset_model_first_frame=min(near)+1, side_onset_model_last_frame=max(near)+1,
                   side_onset_model_range_s=float(t[max(near)]-t[min(near)]))
    if np.sqrt(loss) > max(.015, .10*drop):
        return retain('poor_raw_body_fit')
    if t[onset]-t[old] <= max(.035, dt) or gain < .05:
        return retain('original_onset_supported', False)

    details.update(side_body_descent_onset_frame=onset+1, side_body_descent_onset_s=float(t[onset]))
    torso = f['groups'][1][:, 0, 0]
    guard_start = max(old, lo)
    pre_range = float(np.ptp(torso[guard_start:onset+1]))
    core_range = float(np.ptp(torso[guard_start:bottom+1]))
    guard = max(args.side_onset_torso_guard, .10*core_range)
    details.update(side_onset_pre_torso_range_leg=pre_range, side_onset_torso_guard_leg=guard)
    if pre_range > guard:
        lower_body, support_details = early_lower_body_support(f, guard_start, onset, bottom, args)
        details.update(support_details)
        if not lower_body:
            details.update(side_torso_status='preparation_without_supported_lower_body_motion',
                           side_onset_status='grounded_descent_after_torso_preparation', side_onset_needs_review=0)
            return int(onset), details
        torso_onset, torso_details = supported_torso_onset(f, event, args, lo, hi, old, onset, weights)
        details.update(torso_details)
        if torso_onset <= old:
            return retain('earlier_torso_motion_preserved')
        onset = min(onset, torso_onset)
        details.update(side_onset_status='grounded_raw_body_and_torso_descent', side_onset_needs_review=0)
        return int(onset), details
    details.update(side_onset_status='grounded_raw_body_descent', side_onset_needs_review=0)
    return int(onset), details

def refine_side_onsets(data, args, events, frames):
    f = side_features(data, args, args._view_info['visible_side'])
    t, previous_end = f['t'], 0
    for event in events:
        old = int(event['start_video_frame']) - 1
        onset, details = grounded_descent_onset(f, event, args, previous_end)
        event.update(details)
        if onset > old:
            alternatives = [grounded_descent_onset(f, event, args, previous_end, q)[0]
                            for q in (.20, .30)]
            for i in range(old, onset):
                frames[i].update(rep_id=0, squat_label=-1, reason='before_grounded_side_descent', movement_phase='')
            event.update(start_video_frame=onset+1, start_video_s=float(t[onset]),
                         duration_s=float(event['end_video_s']-t[onset]),
                         start_earlier_s=float(event['hip_core_start_video_s']-t[onset]),
                         start_driver='raw_visible_torso_change_point' if details['side_onset_status']=='grounded_raw_body_and_torso_descent' else 'grounded_raw_visible_shoulders_hips',
                         start_sensitivity_first_frame=min(alternatives+[onset])+1,
                         start_sensitivity_last_frame=max(alternatives+[onset])+1)
        previous_end = int(event['end_video_frame'])
    return events, frames
