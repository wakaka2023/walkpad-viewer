import json

import numpy as np
from scipy.signal import find_peaks, detrend
from scipy.integrate import trapezoid

from action_features import finite_max, interpolate_short, window_stat
from activity_rules import runs, smooth_valid


ACTIVITY_LABELS = {
    "standing": "\u7ad9\u7acb / \u7b49\u5f85",
    "gait_walk": "\u8d70\u8def\u5019\u9009",
    "running": "\u8dd1\u6b65\u5019\u9009",
    "walking": "\u8d70\u8def / \u79fb\u52a8",
    "stepping": "\u8fde\u7eed\u8fc8\u6b65\uff08\u8d70/\u8dd1\u672a\u533a\u5206\uff09",
    "fast_stepping": "\u8fde\u7eed\u8fc8\u6b65\uff08\u5e45\u5ea6\u8f83\u5927\uff09",
    "body_motion": "\u6301\u7eed\u8fd0\u52a8\uff08\u4e0b\u80a2\u663e\u793a\u4e0d\u5168\uff09",
    "arm_activity": "\u51c6\u5907 / \u624b\u90e8\u8c03\u6574",
    "bending": "\u4fef\u8eab / \u8d77\u8eab",
    "between_reps": "\u7ec4\u5185\u56de\u4f4d",
    "squat": "\u4e0b\u8e72 / \u8d77\u7acb",
    "split_squat": "\u4e0b\u80a2\u5c48\u4f38\uff08\u5bbd\u7ad9\u8ddd\uff09",
    "unknown": "\u65e0\u6cd5\u786e\u8ba4 / \u5c40\u90e8\u51fa\u753b"
}


COLORS = {'standing': '#66C2FF', 'walking': '#5DD6A0', 'stepping': '#FFD166',
          'fast_stepping': '#FFAB87', 'body_motion': '#EBC780', 'arm_activity': '#C6A5FF',
          'bending': '#F497C5', 'between_reps': '#A4D8CA', 'squat': '#5CDDB1', 'split_squat': '#72E0E8', 'unknown': '#A6B0BF'}


def clean_mask(t, mask, usable, minimum, gap):
    result = np.asarray(mask, bool).copy() & usable
    dt = float(np.median(np.diff(t)))
    for a, b in runs(~result):
        if a > 0 and b < len(t) and t[b] - t[a] <= gap and usable[a:b].all():
            result[a:b] = True
    for a, b in runs(result):
        if t[b - 1] - t[a] + dt < minimum:
            result[a:b] = False
    return result


def period_features(f, config):
    t = f['time_s']
    dt = float(np.median(np.diff(t)))
    hz, score = np.full(len(t), np.nan), np.full(len(t), np.nan)
    signals = [f['left_foot_relative_x'], f['right_foot_relative_x'],
               f['left_knee_relative_y'], f['right_knee_relative_y']]
    sample_indices = np.arange(0, len(t), max(1, round(.25 / dt)))
    for index in sample_indices:
        a, b = np.searchsorted(t, [t[index] - config['period_window_s'] / 2,
                                   t[index] + config['period_window_s'] / 2])
        a, b = max(0, a), min(len(t), b)
        if b - a < max(12, 1.8 / dt):
            continue
        best = None
        for values in signals:
            sample = values[a:b]
            if np.isfinite(sample).mean() < .8:
                continue
            good = np.isfinite(sample)
            grid = np.arange(t[a], t[b - 1] + dt / 2, dt)
            series = np.interp(grid, t[a:b][good], sample[good])
            if np.percentile(series, 90) - np.percentile(series, 10) < .05:
                continue
            series = detrend(series)
            power = np.dot(series, series)
            if power < 1e-6:
                continue
            corr = np.correlate(series, series, mode='full')[len(series) - 1:] / power
            lo = max(2, int(1 / config['period_max_hz'] / dt))
            hi = min(len(series) // 2, int(1 / config['period_min_hz'] / dt))
            peaks, _ = find_peaks(corr[lo:hi])
            if len(peaks):
                lag = lo + int(peaks[np.argmax(corr[lo + peaks])])
                candidate = (float(corr[lag]), 1 / (lag * dt))
                if best is None or candidate[0] > best[0]:
                    best = candidate
        if best is not None:
            score[index], hz[index] = best
    for values in [score, hz]:
        values[:], _ = interpolate_short(t, values, np.isfinite(values), .5)
    f['period_score'], f['rhythm_hz'] = score, hz


def cycle_events(f, config):
    t = f['time_s']
    signal, good = interpolate_short(t, f['knee_flexion'], np.isfinite(f['knee_flexion']), config['pose_gap_s'])
    signal, blocks = smooth_valid(t, signal, good, .3)
    events, rejected = [], []
    dt = float(np.median(np.diff(t)))
    for lo, hi in blocks:
        if hi - lo < 8:
            continue
        peaks, properties = find_peaks(signal[lo:hi], prominence=config['squat_min_knee_excursion_deg'],
                                       distance=max(1, round(config['cycle_min_separation_s'] / dt)))
        troughs, _ = find_peaks(-signal[lo:hi], prominence=4,
                                distance=max(1, round(.3 / dt)))
        troughs = np.r_[lo, lo + troughs, hi - 1]
        previous_end = lo
        for number, p in enumerate(peaks + lo):
            before = troughs[(troughs < p) & (t[p] - t[troughs] <= config['cycle_max_s'])]
            after = troughs[(troughs > p) & (t[troughs] - t[p] <= config['cycle_max_s'])]
            if not len(before) or not len(after):
                rejected.append(dict(peak_video_s=float(t[p]), reason='no_complete_flexion_recovery'))
                continue
            a, b = int(before[-1]), int(after[0])
            excursion = min(signal[p] - signal[a], signal[p] - signal[b])
            row = dict(peak_video_s=float(t[p]), knee_excursion_deg=float(excursion),
                       peak_knee_flexion_deg=float(signal[p]))
            reason = None
            if excursion < config['squat_min_knee_excursion_deg'] or signal[p] < config['squat_min_knee_flexion_deg']:
                reason = 'insufficient_supported_knee_flexion'
            elif not config['cycle_min_s'] <= t[b] - t[a] <= config['cycle_max_s']:
                reason = 'cycle_duration_out_of_range'
            elif np.isfinite(f['foot_speed'][a:b]).mean() < .5:
                reason = 'missing_camera_compensated_feet'
            else:
                foot_speed = float(np.nanmedian(f['foot_speed'][a:b]))
                row['median_foot_speed_torso_s'] = foot_speed
                if foot_speed > config['cycle_max_foot_speed']:
                    reason = 'moving_feet_not_supported_as_grounded_flexion'
            if reason:
                rejected.append(dict(row, reason=reason))
                continue
            vertical = f['hip_vertical_speed_raw']
            descent = vertical[a:p + 1]
            ascent = vertical[p:b + 1]
            if np.isfinite(descent).mean() < .7 or np.isfinite(ascent).mean() < .7:
                rejected.append(dict(row, reason='missing_supported_hip_descent_recovery'))
                continue
            descent_drop = float(trapezoid(np.nan_to_num(descent), t[a:p + 1]))
            ascent_drop = -float(trapezoid(np.nan_to_num(ascent), t[p:b + 1]))
            row['camera_compensated_hip_drop_torso_ratio'] = min(descent_drop, ascent_drop)
            if min(descent_drop, ascent_drop) < config['cycle_min_hip_drop']:
                rejected.append(dict(row, reason='no_coherent_hip_descent_and_recovery'))
                continue
            fraction = config['cycle_boundary_fraction']
            left_threshold = signal[a] + fraction * (signal[p] - signal[a])
            right_threshold = signal[b] + fraction * (signal[p] - signal[b])
            start_ids = np.flatnonzero(signal[a:p + 1] <= left_threshold)
            end_ids = np.flatnonzero(signal[p:b + 1] <= right_threshold)
            if not len(start_ids) or not len(end_ids):
                rejected.append(dict(row, reason='no_observed_cycle_crossing'))
                continue
            start = int(max(a + int(start_ids[-1]) + 1, previous_end))
            end = int(p + int(end_ids[0]))
            if start >= p or end <= p:
                continue
            around = (t >= t[p] - .25) & (t <= t[p] + .25)
            stance = f['stance_width'][a:b]
            stance = stance[np.isfinite(stance)]
            asymmetry = f['knee_asymmetry'][around]
            asymmetry = asymmetry[np.isfinite(asymmetry)]
            split = len(stance) and np.percentile(stance, 75) >= config['split_min_stance_ratio']
            kind = 'split_squat' if split else 'squat'
            row.update(event_id=0, activity=kind, activity_zh=ACTIVITY_LABELS[kind],
                       start_video_frame=start + 1, end_video_frame=end,
                       end_exclusive_video_frame=end + 1, start_video_s=float(t[start]),
                       end_video_s=float(t[end]), peak_video_frame=int(p + 1),
                       duration_s=float(t[end] - t[start]),
                       projected_stance_ratio=float(np.percentile(stance, 75)) if len(stance) else None,
                       projected_knee_asymmetry_deg=float(np.median(asymmetry)) if len(asymmetry) else None,
                       boundary_method='observed_knee_excursion_crossing',
                       evidence='visible_flexion_recovery_and_camera_compensated_feet',
                       reviewed=0, needs_review=1)
            events.append(row)
            previous_end = end
    for i, row in enumerate(events, 1):
        row['event_id'] = i
    return events, rejected


def bend_events(f, config, return_rejected=False):
    t, ratio = f['time_s'], f['local_torso_ratio']
    signal = 1 - ratio
    good = np.isfinite(signal)
    signal, blocks = smooth_valid(t, signal, good, .3)
    result, partial, rejected = [], [], []
    dt = float(np.median(np.diff(t)))
    for lo, hi in blocks:
        if hi - lo < 8:
            continue
        peaks, _ = find_peaks(signal[lo:hi], prominence=config['bend_min_excursion'],
                              distance=max(1, round(config['cycle_min_separation_s'] / dt)))
        troughs, _ = find_peaks(-signal[lo:hi], prominence=.03,
                                distance=max(1, round(.25 / dt)))
        troughs = np.r_[lo, lo + troughs, hi - 1]
        for p in peaks + lo:
            before, after = troughs[troughs < p], troughs[troughs > p]
            if not len(before) or not len(after):
                continue
            a, b = int(before[-1]), int(after[0])
            if not config['cycle_min_s'] <= t[b] - t[a] <= config['cycle_max_s']:
                continue
            if min(signal[p] - signal[a], signal[p] - signal[b]) < config['bend_min_excursion']:
                continue
            feet, knees = f['foot_speed'][a:b], f['knee_flexion'][a:b]
            foot_coverage, knee_coverage = float(np.isfinite(feet).mean()), float(np.isfinite(knees).mean())
            identity = dict(candidate_type='bending', peak_video_s=float(t[p]),
                            foot_coverage=foot_coverage, knee_coverage=knee_coverage)
            if min(foot_coverage, knee_coverage) < config['bend_partial_min_coverage']:
                rejected.append(dict(identity, reason='insufficient_lower_body_support_for_bend'))
                continue
            if np.nanmedian(feet) > .35 or np.nanmedian(knees) >= 45:
                rejected.append(dict(identity, reason='moving_feet_or_large_knee_flexion_for_bend'))
                continue
            camera = f['camera_speed'][a:b]
            camera = camera[np.isfinite(camera)]
            if len(camera) < .7 * (b - a) or np.percentile(camera, 75) > config['bend_max_camera_speed']:
                rejected.append(dict(identity, reason='unsupported_or_moving_camera_for_bend'))
                continue
            reference = max(signal[a], signal[b])
            start_threshold = signal[a] + config['cycle_boundary_fraction'] * (signal[p] - signal[a])
            end_threshold = reference + config['cycle_boundary_fraction'] * (signal[p] - reference)
            before_cross = np.flatnonzero(signal[a:p + 1] <= start_threshold)
            after_cross = np.flatnonzero(signal[p:b + 1] <= end_threshold)
            if not len(before_cross) or not len(after_cross):
                continue
            start, end = int(a + before_cross[-1] + 1), int(p + after_cross[0])
            if start >= p or end <= p:
                continue
            row = dict(event_id=0, activity='bending', activity_zh=ACTIVITY_LABELS['bending'],
                               start_video_frame=start + 1, end_video_frame=end, end_exclusive_video_frame=end + 1,
                               start_video_s=float(t[start]), end_video_s=float(t[end]), peak_video_s=float(t[p]),
                               peak_video_frame=int(p + 1), duration_s=float(t[end] - t[start]),
                               torso_shortening_excursion=float(min(signal[p] - signal[a], signal[p] - signal[b])),
                               boundary_method='local_torso_shortening_cycle',
                               evidence='projected_torso_shortening_with_quiet_feet_and_limited_knee_flexion',
                               foot_observed_fraction=foot_coverage, knee_observed_fraction=knee_coverage,
                               lower_body_support='direct_cycle_observations',
                               reviewed=0, needs_review=1)
            if min(foot_coverage, knee_coverage) >= .5:
                result.append(row)
            else:
                edges_supported = min(np.isfinite(f['foot_speed'][a:p + 1]).mean(),
                                      np.isfinite(f['foot_speed'][p:b]).mean()) >= .25
                if row['torso_shortening_excursion'] >= config['bend_partial_min_excursion'] and edges_supported:
                    partial.append(row)
                else:
                    rejected.append(dict(identity, reason='partial_bend_missing_strong_torso_or_bilateral_time_support'))
    confirmed = sorted(result, key=lambda row: row['peak_video_s'])
    for candidate in partial:
        before = [row for row in confirmed if row['peak_video_s'] < candidate['peak_video_s']]
        after = [row for row in confirmed if row['peak_video_s'] > candidate['peak_video_s']]
        supported = False
        if before and after:
            previous, following = before[-1], after[0]
            left = candidate['peak_video_s'] - previous['peak_video_s']
            right = following['peak_video_s'] - candidate['peak_video_s']
            tolerance = config['bend_context_period_tolerance']
            supported = (abs(left - right) <= tolerance * min(left, right)
                         and max(left, right) <= min(4., config['cycle_max_s'] / 2)
                         and previous['end_video_s'] <= candidate['start_video_s']
                         and candidate['end_video_s'] <= following['start_video_s'])
        if supported:
            candidate['lower_body_support'] = 'partial_observations_with_two_confirmed_neighbor_cycles'
            candidate['evidence'] = 'strong_observed_torso_cycle;partial_quiet_lower_body;two_confirmed_neighbor_cycles'
            result.append(candidate)
        else:
            rejected.append(dict(candidate_type='bending', peak_video_s=candidate['peak_video_s'],
                                 reason='partial_bend_without_consistent_confirmed_neighbors'))
    result.sort(key=lambda row: row['start_video_s'])
    return (result, rejected) if return_rejected else result


def classify_raw(f, config, *, arm_semantics=True, bend_semantics=True):
    t = f['time_s']
    usable = f['torso_visible']
    labels = np.full(len(t), 'unknown', dtype=object)
    reasons = np.full(len(t), 'missing_or_ambiguous_motion_evidence', dtype=object)
    foot_visible = f['raw_valid'][:, [15, 16]].any(axis=1)
    lower_visible = window_stat(t, foot_visible.astype(float), config['motion_window_s'], 'mean')
    f['foot_visibility_window'] = lower_visible
    net = f['hip_net_speed']
    lower_motion = (f['knee_motion'] > config['knee_motion_speed']) | (f['foot_motion'] > config['foot_motion_speed'])
    quiet = usable & (f['hip_speed'] < config['quiet_hip_speed'])
    quiet &= ((f['knee_motion'] < config['quiet_knee_speed']) |
              (f['foot_speed'] < config['quiet_foot_speed']))
    masks = []
    masks.append(('standing', quiet, 'low_camera_compensated_body_and_leg_motion'))
    if arm_semantics:
        arm_reach = window_stat(t, f['arm_reach'].astype(float), config['motion_window_s'], 'mean')
        arm_pose = (arm_reach >= .45) & (f['arm_posture_change'] > .18)
        arm_gesture = (f['arm_motion'] > config['arm_motion_speed']) & (net < .25)
        arm = usable & (arm_pose | arm_gesture) & ((net < .4) | ~np.isfinite(net))
        arm &= (f['foot_speed'] < .35) | (lower_visible < .35)
        masks.append(('arm_activity', arm, 'sustained_arm_posture_with_limited_locomotion;semantic_operation_unconfirmed'))
        body = usable & (lower_visible < .35)
        body &= ((f['hip_speed'] > .4) & (f['arm_motion'] > .35)) | ((f['knee_motion'] > .3) & (f['arm_motion'] > .35))
        masks.append(('body_motion', body, 'visible_body_motion_with_incomplete_feet;walking_running_unresolved'))
    steps = usable & lower_motion & (lower_visible >= .35) & (f['foot_speed'] > config['walking_foot_speed'])
    masks.append(('stepping', steps, 'repeated_lower_limb_motion;walking_running_unresolved'))
    fast = steps & (f['foot_motion'] > config['fast_foot_motion_speed']) & (f['knee_motion'] > config['fast_knee_motion_speed'])
    masks.append(('fast_stepping', fast, 'larger_projected_limb_motion;not_calibrated_running_speed'))
    walk = steps & (net > config['walking_net_speed']) & (f['foot_speed'] > config['walking_foot_speed'])
    masks.append(('walking', walk, 'net_background_compensated_body_translation_with_leg_motion'))
    bend = usable & (f['local_torso_ratio'] < config['bend_torso_ratio'])
    bend &= (f['foot_speed'] < .4) & (f['knee_flexion'] < 45)
    bend &= f['camera_speed'] <= config['bend_max_camera_speed']
    if bend_semantics:
        masks.append(('bending', bend, 'torso_foreshortening_with_stable_feet;projected_candidate'))
    for name, raw, reason in masks:
        minimum = config['fast_min_bout_s'] if name == 'fast_stepping' else config['minimum_bout_s']
        mask = clean_mask(t, raw, usable, minimum, config['close_gap_s'])
        labels[mask], reasons[mask] = name, reason
    movement_names = ('walking', 'stepping', 'fast_stepping', 'body_motion')
    movement = np.isin(labels, movement_names)
    movement = clean_mask(t, movement, usable, config['minimum_bout_s'], config['close_gap_s'])
    for a, b in runs(movement):
        coverage = float(foot_visible[a:b].mean())
        net_values = net[a:b][np.isfinite(net[a:b])]
        speed_values = f['hip_speed'][a:b][np.isfinite(f['hip_speed'][a:b])]
        net_speed = float(np.median(net_values)) if len(net_values) else 0
        speed = float(np.median(speed_values)) if len(speed_values) else 1
        kind = 'body_motion' if coverage < .35 else 'walking' if net_speed > .25 and net_speed / max(speed, .01) > .45 else 'stepping'
        labels[a:b] = kind
        reasons[a:b] = ('partial_body_motion_bout;walking_running_unresolved' if kind == 'body_motion' else
                        'body_translation_with_visible_steps_bout' if kind == 'walking' else
                        'continuous_limb_motion_bout;walking_running_unresolved')
        if kind == 'stepping':
            fast_raw = (f['foot_motion'][a:b] > config['fast_foot_motion_speed']) & (f['knee_motion'][a:b] > config['fast_knee_motion_speed'])
            fast_mask = clean_mask(t[a:b], fast_raw, usable[a:b], config['fast_min_bout_s'], config['close_gap_s'])
            labels[a:b][fast_mask] = 'fast_stepping'
            reasons[a:b][fast_mask] = 'larger_projected_limb_motion_bout;not_calibrated_running_speed'
    translate = usable & (lower_visible >= .35) & (net > config['walking_net_speed'])
    translate &= net / np.maximum(f['hip_speed'], .01) > .6
    translate &= f['foot_speed'] > config['walking_foot_speed']
    translate = clean_mask(t, translate, usable, 1.5, config['close_gap_s'])
    translate &= np.isin(labels, movement_names)
    labels[translate], reasons[translate] = 'walking', 'sustained_net_body_translation_with_visible_steps'
    events, rejected = cycle_events(f, config)
    if bend_semantics:
        bends, bend_rejected = bend_events(f, config, return_rejected=True)
        rejected.extend(bend_rejected)
        for candidate in bends:
            if not any(candidate['start_video_s'] < row['end_video_s'] and candidate['end_video_s'] > row['start_video_s'] for row in events):
                events.append(candidate)
    events.sort(key=lambda row: row['start_video_s'])
    for number, row in enumerate(events, 1):
        row['event_id'] = number
    event_ids = np.zeros(len(t), int)
    for row in events:
        a, b = row['start_video_frame'] - 1, row['end_video_frame']
        labels[a:b], reasons[a:b] = row['activity'], row['evidence']
        event_ids[a:b] = row['event_id']
    for previous, following in zip(events[:-1], events[1:]):
        if previous['activity'] == following['activity'] and 0 <= following['start_video_s'] - previous['end_video_s'] <= 1.:
            a, b = previous['end_video_frame'], following['start_video_frame'] - 1
            if usable[a:b].all():
                labels[a:b], reasons[a:b] = 'between_reps', 'short_observed_recovery_between_same_type_cycles'
    for a, b in runs(labels == 'unknown'):
        if a > 0 and b < len(t) and t[b] - t[a] <= config['pose_gap_s']:
            if labels[a - 1] == labels[b] and event_ids[a - 1] == event_ids[b] and usable[a:b].all():
                labels[a:b], reasons[a:b] = labels[a - 1], 'brief_supported_same_state_gap'
    for a, b in runs(~usable):
        labels[a:b], reasons[a:b], event_ids[a:b] = 'unknown', 'insufficient_visible_body_for_scale', 0
    ends = np.r_[t[1:], t[-1] + np.median(np.diff(t))]
    segments, frames = [], []
    a = 0
    for b in range(1, len(t) + 1):
        if b < len(t) and labels[a] == labels[b] and event_ids[a] == event_ids[b]:
            continue
        name = str(labels[a])
        reason_set = sorted(set(str(v) for v in reasons[a:b]))
        evidence = 'limited' if name in ('unknown', 'body_motion', 'arm_activity', 'bending') else 'moderate'
        segments.append(dict(segment_id=len(segments) + 1, activity=name, activity_zh=ACTIVITY_LABELS[name],
                             action_event_id=int(event_ids[a]), start_video_frame=a + 1, end_video_frame=b,
                             end_exclusive_video_frame=b + 1, start_video_s=float(t[a]), end_video_s=float(ends[b - 1]),
                             duration_s=float(ends[b - 1] - t[a]), evidence_level=evidence,
                             reason=';'.join(reason_set), visible_body_fraction=float(usable[a:b].mean()),
                             visible_feet_fraction=float(foot_visible[a:b].mean()),
                             camera_compensated_fraction=float(np.isfinite(f['hip_speed_raw'][a:b]).mean()),
                             reviewed=0, needs_review=1))
        for i in range(a, b):
            frames.append(dict(video_frame=i + 1, video_time_s=float(t[i]), frame_end_s=float(ends[i]),
                               segment_id=len(segments), activity=name, activity_zh=ACTIVITY_LABELS[name],
                               action_event_id=int(event_ids[i]), evidence_level=evidence,
                               activity_reason=str(reasons[i]), is_segment_start=int(i == a),
                               is_segment_end=int(i == b - 1), reviewed=0, needs_review=1))
        a = b
    boundaries = []
    for index, row in enumerate(segments):
        boundary = row['start_video_s']
        uncertainty = config['boundary_uncertainty_s']
        boundaries.append(dict(boundary_id=len(boundaries) + 1, video_frame=row['start_video_frame'],
                               video_time_s=boundary, from_activity=segments[index - 1]['activity'] if index else 'VIDEO_START',
                               to_activity=row['activity'], context_start_s=max(float(t[0]), boundary - uncertainty),
                               context_end_s=min(float(ends[-1]), boundary + uncertainty),
                               uncertainty_definition='review_context_not_a_calibrated_error_bound',
                               source='automatic_visual_rules', reviewed=0, needs_review=1))
    boundaries.append(dict(boundary_id=len(boundaries) + 1, video_frame=len(t) + 1,
                           video_time_s=float(ends[-1]), from_activity=segments[-1]['activity'],
                           to_activity='VIDEO_END', context_start_s=float(ends[-1]), context_end_s=float(ends[-1]),
                           uncertainty_definition='video_end_last_frame_duration_estimated',
                           source='source_pts', reviewed=0, needs_review=1))
    return segments, frames, events, boundaries, rejected


def classify(f, config):
    from action_temporal import assemble
    segments, frames, events, boundaries, rejected = classify_raw(f, config)
    f['raw_segments'], f['raw_frame_labels'] = segments, frames
    segments, frames, events, boundaries = assemble(f, segments, frames, events, config, ACTIVITY_LABELS)
    return segments, frames, events, boundaries, rejected
