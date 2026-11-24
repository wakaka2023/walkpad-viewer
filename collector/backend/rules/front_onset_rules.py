import numpy as np
from .body_boundary_rules import body_features

def _fit(t, y, onset, weights):
    x = t - onset
    design = np.c_[np.ones(len(x)), x, np.maximum(0, x)**2]
    w = weights.copy()
    for _ in range(3):
        sw = np.sqrt(w)
        coef = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)[0]
        residual = y - design @ coef
        sigma = max(.003, 1.4826 * float(np.median(np.abs(residual - np.median(residual)))))
        w = weights * np.minimum(1, 1.5 * sigma / np.maximum(np.abs(residual), 1e-12))

    delta = .015
    r = np.abs(residual)
    loss = np.average(np.where(r <= delta, residual**2, 2*delta*r-delta**2), weights=weights)
    return float(loss), coef

def raw_descent_onset(f, event, previous_end=0, fraction=.25):
    t = f['t']; k = f['base']['k']; scale = f['base']['scale']
    a = int(event['start_video_frame'])-1
    c = int(event['hip_core_start_video_frame'])-1
    p = int(event['bottom_video_frame'])-1
    details = dict(onset_fit_status='original_onset_retained', onset_previous_video_s=float(t[a]),
                   onset_previous_video_frame=a+1)
    if c <= a:
        details['onset_fit_status'] = 'no_precore_extension'
        return a, details
    lo = max(previous_end, int(np.searchsorted(t, t[a]-.3)))
    signal = f['body_top_signal']
    if lo > c or not np.isfinite(signal[lo:p+1]).all():
        details['onset_fit_status'] = 'missing_body_evidence'
        return a, details
    top = float(np.quantile(signal[lo:c+1], .1))
    drop = float(signal[p]-top)
    crossing = np.flatnonzero(signal[c:p+1] >= top+fraction*drop)
    hi = c+int(crossing[0])+1 if len(crossing) else p
    dt = float(np.median(np.diff(t)))
    if hi-lo < 8 or hi-c < 3 or drop < .04:
        details['onset_fit_status'] = 'insufficient_descent_support'
        return a, details
    if not f['valid'][lo:hi].all() or np.any(np.diff(t[lo:hi]) > max(.25, 3*dt)):
        details['onset_fit_status'] = 'missing_pose_or_time_gap'
        return a, details
    feet = k[:, [15,16], 1].mean(axis=1)
    raw = (.6*k[:, [5,6], 1].mean(axis=1)+.4*k[:, [11,12], 1].mean(axis=1)-feet)/scale
    xx = t[lo:hi]; yy = raw[lo:hi]
    if not np.isfinite(yy).all():
        details['onset_fit_status'] = 'missing_raw_pose'
        return a, details
    weights = np.gradient(xx)
    candidates = []
    for onset in range(a, min(c, hi-4)+1):
        loss, coef = _fit(xx, yy, t[onset], weights)
        if coef[2] > 0 and abs(coef[1]) <= .08:
            candidates.append((loss, onset, coef))
    if not candidates:
        details['onset_fit_status'] = 'no_valid_drift_to_descent_model'
        return a, details
    loss, onset, coef = min(candidates, key=lambda x: x[0])
    old_loss, _ = _fit(xx, yy, t[a], weights)
    gain = (old_loss-loss)/max(old_loss, 1e-12)
    near = [i for l,i,_ in candidates if l <= loss*1.15+1e-6]
    details.update(onset_fit_start_s=float(xx[0]), onset_fit_end_s=float(xx[-1]),
                   onset_fit_rmse_leg=float(np.sqrt(loss)), onset_fit_relative_gain=float(gain),
                   onset_pretrend_leg_s=float(coef[1]), onset_acceleration_coefficient=float(coef[2]),
                   onset_model_first_frame=min(near)+1, onset_model_last_frame=max(near)+1,
                   onset_model_range_s=float(t[max(near)]-t[min(near)]))
    if np.sqrt(loss) > max(.015, .10*drop):
        details['onset_fit_status'] = 'poor_raw_pose_fit'
        return a, details
    if t[onset]-t[a] <= max(.035, dt) or gain < .05:
        details['onset_fit_status'] = 'original_onset_supported'
        return a, details
    details['onset_fit_status'] = 'raw_body_descent_change_point'
    return onset, details

def refine_front_onsets(data, args, events, frames):
    f = body_features(data, args); t = f['t']; previous_end = 0
    for event in events:
        old = int(event['start_video_frame'])-1
        onset, details = raw_descent_onset(f, event, previous_end)
        event.update(details)
        if onset > old:
            alternatives = [raw_descent_onset(f,event,previous_end,q)[0] for q in (.20,.30)]
            for i in range(old, onset):
                frames[i].update(rep_id=0, squat_label=-1, reason='before_raw_body_descent', movement_phase='')
            event.update(start_video_frame=onset+1, start_video_s=float(t[onset]),
                         duration_s=float(event['end_video_s']-t[onset]),
                         start_earlier_s=float(event['hip_core_start_video_s']-t[onset]),
                         start_driver='raw_shoulders_hips_descent',
                         start_sensitivity_first_frame=min(alternatives+[onset])+1,
                         start_sensitivity_last_frame=max(alternatives+[onset])+1)
        previous_end = int(event['end_video_frame'])
    return events, frames
