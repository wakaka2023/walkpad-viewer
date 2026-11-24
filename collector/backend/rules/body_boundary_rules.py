import numpy as np
from .activity_rules import features, smooth_valid, derivative, runs, label_hip_poses

GROUP_NAMES = ('shoulders', 'torso', 'hips', 'knees', 'head')

def body_features(data, args):
    f = features(data, args)
    t, k, scale = f['t'], f['k'], f['scale']
    ids = [5, 6, 11, 12, 13, 14, 15, 16]
    valid = np.isfinite(k[:, ids]).all(axis=(1, 2)) & (k[:, ids, 2] >= args.keypoint_conf).all(axis=1)
    feet = k[:, [15, 16], :2].mean(axis=1)
    positions = (k[:, [5, 6, 11, 12, 13, 14], :2] - feet[:, None, :]) / scale
    positions, blocks = smooth_valid(t, positions, valid, args.smooth_seconds)
    head_valid = np.isfinite(k[:,0]).all(axis=1) & (k[:,0,2] >= args.keypoint_conf)
    head, head_blocks = smooth_valid(t, (k[:,0,:2]-feet)/scale, head_valid, args.smooth_seconds)
    shoulders = positions[:, :2].mean(axis=1)
    hips = positions[:, 2:4].mean(axis=1)
    groups = [positions[:, :2], (shoulders-hips)[:, None], positions[:, 2:4], positions[:, 4:6], head[:,None]]
    velocity = derivative(t, positions, blocks)
    speed = np.linalg.norm(velocity, axis=2).max(axis=1)
    head_speed = np.linalg.norm(derivative(t,head,head_blocks),axis=1)
    reference_speed = np.maximum(speed,head_speed*.4)

    top_signal = .35 * head[:,1] + .35 * shoulders[:, 1] + .3 * hips[:, 1]
    return dict(base=f, t=t, valid=valid, head_valid=head_valid, groups=groups, speed=reference_speed,
                top_signal=top_signal, blocks=blocks,
                body_top_signal=.6*shoulders[:,1]+.4*hips[:,1],body_speed=speed,
                upper_valid=np.isfinite(k[:,[5,6]]).all(axis=(1,2)) & (k[:,[5,6],2]>=args.keypoint_conf).all(axis=1))

def neutral_hold(f, lo, hi, hold_seconds=.15):
    t, valid = f['t'], f['valid']
    dt = float(np.median(np.diff(t)))
    variants=[(f['top_signal'],f['speed'],hold_seconds,'body_and_head_quiet'),
              (f['body_top_signal'],f['body_speed'],hold_seconds,'body_quiet_head_not_confirmed'),
              (f['body_top_signal'],f['body_speed'],.10,'short_body_hold_head_not_confirmed')]
    for signal,speed,seconds,quality in variants:
        ok = valid[lo:hi] & np.isfinite(signal[lo:hi]) & np.isfinite(speed[lo:hi])
        if ok.sum() < 3:
            continue
        ceiling = np.quantile(signal[lo:hi][ok], .3)
        candidates = ok & (signal[lo:hi] <= ceiling) & (speed[lo:hi] <= .12)
        holds = [(a+lo, b+lo) for a,b in runs(candidates)
                 if t[lo+b-1]-t[lo+a]+dt >= seconds]
        if holds:
            hold=max(holds,key=lambda ab:t[ab[1]-1]-t[ab[0]])
            f.setdefault('hold_quality',{})[hold]=quality
            return hold
    return None

def clean_activity(t, active, valid, core_a, core_b, minimum=.08, close_gap=.08):
    active = active.copy() & valid
    dt = float(np.median(np.diff(t)))
    for a,b in runs(active):
        if t[b-1]-t[a]+dt < minimum:
            active[a:b] = False

    active[core_a:core_b] |= valid[core_a:core_b]
    for a,b in runs(~active):
        if a>0 and b<len(t) and valid[a:b].all() and t[b]-t[a] <= close_gap:
            active[a:b] = True
    return active

def extend_event(f, event, before, after, fraction):
    t = f['t']
    old_a = event['start_video_frame']-1
    old_b = event['end_video_frame']
    if before is None or after is None:
        return old_a, old_b, None, 'missing_quiet_reference'
    a, b = before[0], after[1]
    if not a <= old_a < old_b <= b or not f['valid'][a:b].all():
        return old_a, old_b, None, 'missing_upper_body_or_time_gap'
    if np.any(np.diff(t[a:b]) > max(.25, 3*np.median(np.diff(t)))):
        return old_a, old_b, None, 'presentation_time_gap'
    left_t, right_t = np.median(t[slice(*before)]), np.median(t[slice(*after)])
    alpha = np.clip((t[a:b]-left_t) / max(right_t-left_t, 1e-9), 0, 1)
    ratios = []
    for group in f['groups']:
        if not np.isfinite(group[slice(*before)]).all() or not np.isfinite(group[slice(*after)]).all():
            ratios.append(np.zeros(b-a))
            continue
        left = np.median(group[slice(*before)], axis=0)
        right = np.median(group[slice(*after)], axis=0)
        baseline = left[None] * (1-alpha[:,None,None]) + right[None] * alpha[:,None,None]
        distance = np.linalg.norm(group[a:b]-baseline, axis=2).max(axis=1)
        reference_noise = np.r_[np.linalg.norm(group[slice(*before)]-left,axis=2).max(axis=1),
                                np.linalg.norm(group[slice(*after)]-right,axis=2).max(axis=1)]
        peak = float(np.nanquantile(distance, .95))
        threshold = max(.008, 1.5*float(np.quantile(reference_noise,.95)), fraction*peak)
        ratios.append(np.nan_to_num(distance / threshold,nan=0.0))
    ratios = np.stack(ratios, axis=1)
    activity = clean_activity(t[a:b], ratios.max(axis=1)>1, f['valid'][a:b], old_a-a, old_b-a)
    connected = next(((x,y) for x,y in runs(activity) if x <= old_a-a < y), None)
    if connected is None:
        return old_a, old_b, None, 'no_connected_body_motion'
    start, end = int(a+connected[0]), int(a+connected[1])
    if end >= len(t):
        return old_a, old_b, None, 'no_observed_return'
    return start, end, (a, b, ratios), 'body_rule_candidate'

def terminal_standing_stop(f, core, before, proposed_end, hold_seconds=.20):
    t = f['t']
    core_end = int(core['end_video_frame'])
    bottom = int(core['bottom_video_frame']) - 1
    if before is None or core_end >= proposed_end:
        return proposed_end, dict(status='no_terminal_extension_to_trim')
    shoulder_y = f['groups'][0][:, :, 1].mean(axis=1)
    hip_y = f['groups'][2][:, :, 1].mean(axis=1)
    upright = f['valid'].copy()
    for y in (shoulder_y, hip_y):
        reference = float(np.median(y[slice(*before)]))
        excursion = float(np.nanmax(y[bottom:core_end+1]) - reference)

        upright &= np.isfinite(y) & (y <= reference + max(.025, .10*excursion))
    speeds = f['body_speed'][bottom:core_end+1]
    speeds = speeds[np.isfinite(speeds)]
    if not len(speeds):
        return proposed_end, dict(status='no_valid_recovery_speed')
    speed_limit = max(.12, .15*float(np.quantile(speeds, .95)))
    quiet = upright & np.isfinite(f['body_speed']) & (f['body_speed'] <= speed_limit)
    quiet[:core_end] = False
    dt = float(np.median(np.diff(t)))
    for a, b in runs(quiet):
        if a >= proposed_end:
            break
        if t[b-1] - t[a] + dt < hold_seconds:
            continue
        confirm = min(b, int(np.searchsorted(t, t[a]+hold_seconds)))
        if np.any(np.diff(t[a:confirm+1]) > max(.25, 3*dt)):
            continue
        return int(a), dict(status='first_upright_body_hold', start=int(a),
                            stop=int(b), confirmation_end=int(confirm),
                            speed_limit=float(speed_limit))
    return proposed_end, dict(status='no_confirmed_terminal_standing_hold',
                             speed_limit=float(speed_limit))

def label_body_poses(data, args):
    events, frames, rejected, _ = label_hip_poses(data, args)
    f = body_features(data, args)
    t = f['t']
    if not events:
        return events, frames, rejected, 'shoulders_torso_hips_knees'
    bottoms = [int(e['bottom_video_frame'])-1 for e in events]
    edges = [0]+bottoms+[len(t)]
    holds = [neutral_hold(f,a,b) for a,b in zip(edges[:-1],edges[1:])]
    original = [dict(e) for e in events]
    from .intercycle_reset_rules import reset_cores, constrain_to_resets, resolve_overlaps
    cores, trusted_resets, reset_barriers = reset_cores(f, original, holds, args, rejected)
    scores = np.full(len(t), np.nan)
    terminal_hold = None
    for number, (event, old) in enumerate(zip(events, cores)):
        start, end, details, status = extend_event(f,old,holds[number],holds[number+1],args.body_fraction)
        start, end = constrain_to_resets(start, end, number, reset_barriers)
        uncapped_end = end
        terminal = dict(status='has_following_cycle')
        if number == len(events)-1:
            end, terminal = terminal_standing_stop(f, old, holds[number], end)
            if terminal['status'] == 'first_upright_body_hold':
                terminal_hold = terminal
        a0, b0 = old['start_video_frame']-1, old['end_video_frame']
        event.update(hip_core_start_video_frame=old['start_video_frame'], hip_core_end_video_frame=old['end_video_frame'],
                     hip_core_start_video_s=old['start_video_s'], hip_core_end_video_s=old['end_video_s'],
                     start_video_frame=start+1, end_video_frame=end, end_exclusive_video_frame=end+1,
                     start_video_s=float(t[start]), end_video_s=float(t[end]), duration_s=float(t[end]-t[start]),
                     start_earlier_s=float(t[a0]-t[start]), end_later_s=float(t[end]-t[b0]),
                     body_boundary_status=status, reviewed=0,
                     head_valid_fraction=float(f['head_valid'][start:end].mean()),
                     before_reference_quality=f.get('hold_quality',{}).get(holds[number],'missing'),
                     after_reference_quality=f.get('hold_quality',{}).get(holds[number+1],'missing'))
        if old.get('core_reset_status'):
            raw = original[number]
            event.update(core_reset_status=old['core_reset_status'],
                         detector_hip_core_start_video_frame=raw['start_video_frame'],
                         detector_hip_core_end_video_frame=raw['end_video_frame'],
                         detector_hip_core_start_video_s=raw['start_video_s'],
                         detector_hip_core_end_video_s=raw['end_video_s'])
        if number in reset_barriers or number + 1 in reset_barriers:
            event['body_reset_status'] = 'separated_by_observed_upright_hold'
        event.update(terminal_stop_status=terminal['status'],
                     terminal_uncapped_end_video_frame=uncapped_end if number==len(events)-1 else '',
                     terminal_uncapped_end_video_s=float(t[uncapped_end]) if number==len(events)-1 else '',
                     terminal_speed_limit_leg_s=terminal.get('speed_limit',''),
                     terminal_confirmation_end_video_frame=terminal.get('confirmation_end',-1)+1 if 'confirmation_end' in terminal else '')
        if details is not None:
            a,b,ratios = details
            scores[a:b] = np.fmax(scores[a:b], ratios.max(axis=1))
            event['start_driver'] = GROUP_NAMES[int(np.argmax(ratios[start-a]))] if start<a0 else 'hip_core'
            event['end_driver'] = GROUP_NAMES[int(np.argmax(ratios[end-1-a]))] if end>b0 else 'hip_core'
            alternatives = [extend_event(f,old,holds[number],holds[number+1],q) for q in (.02,.05)]
            alternatives = [(constrain_to_resets(v[0],v[1],number,reset_barriers)+(v[2],v[3])) for v in alternatives]
            if terminal_hold is not None:
                event['end_driver'] = 'upright_body_hold'
                alternatives = [(v[0], terminal_standing_stop(f,old,holds[number],v[1])[0],v[2],v[3]) for v in alternatives]
            event.update(start_sensitivity_first_frame=min(v[0]+1 for v in alternatives),
                         start_sensitivity_last_frame=max(v[0]+1 for v in alternatives),
                         end_sensitivity_first_frame=min(v[1] for v in alternatives),
                         end_sensitivity_last_frame=max(v[1] for v in alternatives))
        else:
            event.update(start_driver='hip_core_fallback',end_driver='hip_core_fallback')
    resolve_overlaps(events, cores, t, trusted_resets, reset_barriers)

    for index,row in enumerate(frames):
        if row['rep_id']:
            row.update(rep_id=0,squat_label=-1,reason='outside_joint_body_interval')
        row.update(upper_body_valid=int(f['upper_valid'][index]),head_valid=int(f['head_valid'][index]),
                   body_boundary_valid=int(f['valid'][index]),
                   body_activity_score=float(scores[index]),movement_phase='')
    for event in events:
        for index in range(event['start_video_frame']-1,event['end_video_frame']):
            if index+1 < event['hip_core_start_video_frame']:
                phase='BODY_ONSET'
            elif index+1 > event['hip_core_end_video_frame']:
                phase='BODY_RECOVERY'
            else:
                phase='DOWN' if t[index] <= event['bottom_video_s'] else 'UP'
            frames[index].update(squat_label=1,rep_id=event['rep_id'],reason='joint_body_boundary',movement_phase=phase)
    if terminal_hold is not None:
        for index in range(terminal_hold['start'],terminal_hold['stop']):
            frames[index].update(squat_label=0,rep_id=0,reason='confirmed_terminal_upright_hold',movement_phase='')
    return events,frames,rejected,'shoulders_torso_hips_knees'
