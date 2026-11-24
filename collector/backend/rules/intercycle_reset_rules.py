import numpy as np


def set_bounds(event, start, end, t):
    start, end = int(start), int(end)
    bottom = int(event['bottom_video_frame']) - 1
    if not 0 <= start <= bottom < end < len(t):
        raise ValueError('A repetition must contain its observed bottom and return.')
    event.update(start_video_frame=start + 1, end_video_frame=end,
                 end_exclusive_video_frame=end + 1,
                 start_video_s=float(t[start]), end_video_s=float(t[end]),
                 duration_s=float(t[end] - t[start]))
    if 'hip_core_start_video_s' in event:
        event['start_earlier_s'] = float(event['hip_core_start_video_s'] - t[start])
        event['end_later_s'] = float(t[end] - event['hip_core_end_video_s'])


def trusted_reset(f, left, right, hold, args, rejected):
    if hold is None:
        return False
    a, b = map(int, hold)
    p = int(left['bottom_video_frame']) - 1
    q = int(right['bottom_video_frame']) - 1
    t = f['t']
    if not p < a < b <= q or not f['valid'][a:b].all():
        return False
    dt = float(np.median(np.diff(t)))
    if np.any(np.diff(t[a:b]) > max(.25, 3 * dt)):
        return False
    for signal in (f['body_top_signal'], f['groups'][2][:, :, 1].mean(axis=1)):
        values = np.r_[signal[a:b], signal[[p, q]]]
        if not np.isfinite(values).all():
            return False
        reference = float(np.median(signal[a:b]))
        if min(float(signal[p]), float(signal[q])) - reference < args.min_hip_drop:
            return False
    ankle_speed = f['base']['ankle_speed'][a:b]
    if not np.isfinite(ankle_speed).all() or np.max(ankle_speed) > .25:
        return False
    for event in rejected:
        if event.get('reason') == 'jump_like_bilateral_lift_and_fast_rise':
            if int(event['start_video_frame']) - 1 < b and int(event['end_video_frame']) > a:
                return False
    return True


def reset_cores(f, events, holds, args, rejected):
    cores = [dict(event) for event in events]
    trusted, barriers = {}, {}
    for j in range(1, len(events)):
        hold = holds[j]
        if not trusted_reset(f, events[j - 1], events[j], hold, args, rejected):
            continue
        a, b = map(int, hold)
        trusted[j] = (a, b)
        if int(events[j - 1]['end_video_frame']) > a or int(events[j]['start_video_frame']) - 1 < b:
            barriers[j] = (a, b)
    t, height = f['t'], f['base']['height']
    for i, core in enumerate(cores):
        start, end = int(core['start_video_frame']) - 1, int(core['end_video_frame'])
        original_start, original_end = start, end
        p = int(core['bottom_video_frame']) - 1
        if i in barriers:
            a, b = barriers[i]
            if start < b:
                anchor = a + int(np.argmin(height[a:p + 1]))
                threshold = height[anchor] + args.boundary_fraction * (height[p] - height[anchor])
                crossing = np.flatnonzero(height[anchor:p + 1] <= threshold)
                start = max(b, anchor + int(crossing[-1]) + 1) if len(crossing) else b
        if i + 1 in barriers:
            a, b = barriers[i + 1]
            if end > a:
                anchor = p + int(np.argmin(height[p:b]))
                threshold = height[anchor] + args.boundary_fraction * (height[p] - height[anchor])
                crossing = np.flatnonzero(height[p:anchor + 1] <= threshold)
                end = min(a, p + int(crossing[0])) if len(crossing) else a
        if (start, end) != (original_start, original_end):
            set_bounds(core, start, end, t)
            core['core_reset_status'] = 'hip_crossing_recomputed_from_observed_upright_reset'
    return cores, trusted, barriers


def constrain_to_resets(start, end, index, barriers):
    if index in barriers:
        start = max(int(start), barriers[index][1])
    if index + 1 in barriers:
        end = min(int(end), barriers[index + 1][0])
    return int(start), int(end)


def resolve_overlaps(events, cores, t, trusted, barriers):
    changed = set()
    for _ in range(len(events) + 1):
        conflicts = [j for j in range(1, len(events))
                     if events[j - 1]['end_video_frame'] >= events[j]['start_video_frame']]
        if not conflicts:
            return changed
        for j in conflicts:
            if j in trusted:
                barriers[j] = trusted[j]
                a, b = trusted[j]
                for i, start, end in (
                    (j - 1, events[j - 1]['start_video_frame'] - 1, min(events[j - 1]['end_video_frame'], a)),
                    (j, max(events[j]['start_video_frame'] - 1, b), events[j]['end_video_frame']),
                ):
                    set_bounds(events[i], start, end, t)
                    events[i]['body_reset_status'] = 'separated_by_observed_upright_hold'
                    changed.add(i)
            else:
                for i in (j - 1, j):
                    set_bounds(events[i], cores[i]['start_video_frame'] - 1, cores[i]['end_video_frame'], t)
                    events[i].update(body_boundary_status='body_core_fallback_neighbor_conflict',
                                     body_boundary_needs_review=1,
                                     body_reset_status='no_trusted_reset_core_fallback',
                                     start_driver='hip_core_conflict_fallback',
                                     end_driver='hip_core_conflict_fallback')
                    changed.add(i)
    raise ValueError('The underlying hip cores themselves are inconsistent.')
