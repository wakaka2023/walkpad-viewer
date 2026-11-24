import numpy as np

from .activity_rules import runs


MOTION = {'standing_turn', 'walking', 'gait_walk', 'running', 'running_turn', 'stepping', 'body_motion', 'stairs_up', 'stairs_down'}
REPETITIONS = {'squat', 'split_squat', 'bending'}


def label_runs(labels):
    changes = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    return [(int(a), int(b)) for a, b in zip(changes[:-1], changes[1:])]


def stabilize(t, raw_labels, usable, events, config, *, preserve_walk=False):
    labels = np.asarray(raw_labels, object).copy()
    labels[labels == 'fast_stepping'] = 'stepping'
    direct_motion = {'gait_walk','running','running_turn','stairs_up','stairs_down'}
    if preserve_walk:
        direct_motion.update(('walking', 'standing_turn'))
    inferred = np.zeros(len(t), bool)
    protected = np.zeros(len(t), bool)
    ends = np.r_[t[1:], t[-1] + np.median(np.diff(t))]
    groups = []
    for event in events:
        start, end = event['start_video_frame'] - 1, event['end_video_frame']
        if groups and groups[-1]['kind'] == event['activity'] and event.get('group_can_join_previous', True):
            previous = groups[-1]
            gap = t[start] - ends[previous['end'] - 1]
            gap_labels = labels[previous['end']:start]
            incompatible = np.isin(gap_labels, list(MOTION)).any()
            if gap <= config['event_group_gap_s'] and not incompatible and usable[previous['end']:start].all():
                previous['end'] = end
                previous['events'].append(event['event_id'])
                continue
        groups.append(dict(start=start, end=end, kind=event['activity'], events=[event['event_id']]))
    for group in groups:
        a, b = group['start'], group['end']
        labels[a:b] = group['kind']
        protected[a:b] = True
    for state in ['walking', 'stepping', 'body_motion', 'arm_activity', 'standing']:
        for a, b in runs(labels != state):
            if a == 0 or b == len(t) or protected[a:b].any():
                continue
            duration = ends[b - 1] - t[a]
            allowed = {'unknown', 'arm_activity'} if state in MOTION else {'unknown'}
            maximum = config['motion_bridge_s'] if state in MOTION else config['same_state_bridge_s']
            if state not in MOTION and not usable[a:b].all():
                maximum = min(maximum, config['state_bridge_s'])
            if duration <= maximum and set(labels[a:b]).issubset(allowed):
                inferred[a:b] = True
                labels[a:b] = state
    for a, b in label_runs(labels):
        if a == 0 or b == len(t) or protected[a:b].any():
            continue
        if any(labels[i] in direct_motion for i in (a,a-1,b)):
            continue
        if ends[b - 1] - t[a] >= config['state_min_s']:
            continue
        before, after = labels[a - 1], labels[b]
        if before == after and before not in REPETITIONS and labels[a] not in REPETITIONS:
            labels[a:b], inferred[a:b] = before, True
        elif labels[a] == 'arm_activity' and before in REPETITIONS and after == 'standing':
            labels[a:b], inferred[a:b] = 'standing', True
    for a, b in runs(labels == 'unknown'):
        if a == 0 or b == len(t) or ends[b - 1] - t[a] > config['state_bridge_s']:
            continue
        before, after = labels[a - 1], labels[b]
        if before in REPETITIONS or after in REPETITIONS or before in (direct_motion | {'unknown'}) or after in (direct_motion | {'unknown'}):
            continue
        middle = (t[a] + ends[b - 1]) / 2
        labels[a:b] = np.where(t[a:b] < middle, before, after)
        inferred[a:b] = True
    labels[labels == 'between_reps'] = 'standing'
    return labels, inferred, protected


def assemble(f, raw_segments, raw_frames, events, config, activity_labels, *, merge_walk_labels=False, main_actions=None, review=None, measured=None, walk_observations=None, image_height=None):
    t = np.asarray(f['time_s'])
    raw = np.array([row['activity'] for row in raw_frames], dtype=object)
    labels, inferred, protected = stabilize(t, raw, f['torso_visible'], events, config, preserve_walk=review is not None)
    for i, row in enumerate(raw_frames):
        own_inference = row.get('gait_inferred', False) or row.get('boundary_inferred', False)
        inferred[i] |= bool(own_inference and labels[i] == row['activity'])
    if merge_walk_labels:
        labels[labels == 'gait_walk'] = 'walking'
    if main_actions is not None:
        labels[~np.isin(labels, tuple(main_actions))] = 'unknown'
        protected &= labels != 'unknown'
    ends = np.asarray([row['frame_end_s'] for row in raw_frames])
    event_map = {row['event_id']: row for row in events}
    feet = f['raw_valid'][:, [15, 16]].any(axis=1)
    segments, frames = [], []
    reviews = [review(str(kind),row,inferred[i],protected[i],f,i) for i,(kind,row) in enumerate(zip(labels,raw_frames))] if review else None
    context_bridge = np.full(len(t), '', dtype=object)
    if measured is not None:
        from .bounded_action_bridge import bridge
        context_bridge = bridge(t, ends, labels, raw_frames, inferred, protected, reviews, measured, f['torso_visible'], config)
        if walk_observations is not None:
            from .walk_continuity import bridges
            depth = bridges(t, ends, labels, raw_frames, inferred, protected, reviews, measured, *walk_observations)
            context_bridge[(context_bridge == '') & (depth != '')] = depth[(context_bridge == '') & (depth != '')]
        for a,b in label_runs(context_bridge):
            if not context_bridge[a]:
                continue
            if context_bridge[a].startswith('same_track_walk_with_'):
                labels[a:b] = 'walking'
            else:
                labels[a:b] = labels[a - 1]
            inferred[a:b] = True
            for i in range(a,b):
                reviews[i] = review(str(labels[i]), raw_frames[i], True, False, f, i)
        if walk_observations is not None:
            from .gait_gap_evidence import REASON, bridges
            supported = bridges(t, ends, labels, raw_frames, inferred, protected, reviews, measured,
                                 *walk_observations)
            for i in np.flatnonzero(supported != ''):
                labels[i] = supported[i]
                inferred[i] = True
                context_bridge[i] = REASON
                reviews[i] = review(str(labels[i]), raw_frames[i], True, False, f, i)
        if walk_observations is not None and image_height is not None:
            from .cropped_passage import bridges
            supported, why = bridges(t, ends, labels, raw_frames, inferred, protected, reviews, measured,
                                     *walk_observations, image_height)
            for i in np.flatnonzero(supported != ''):
                labels[i] = supported[i]
                inferred[i] = True
                context_bridge[i] = why[i]
                reviews[i] = review(str(labels[i]), raw_frames[i], True, False, f, i)
            # A tiny inherited walk candidate between a measured run and a
            # matching newly observed run cycle is not a new walk bout.
            from .walk_continuity import _step
            for a,b in label_runs(labels):
                if (a==0 or b==len(t) or labels[a]!='walking' or labels[a-1]!=labels[b] or labels[b]!='running'
                        or not measured[a:b].all() or protected[a:b].any() or not inferred[a:b].all()
                        or not (why[a-1] or why[b]) or (why[a-1] and why[b])):
                    continue
                direction=1 if why[a-1] else -1
                j=b if direction==1 else a-1
                nearby=[]
                while 0<=j<len(t) and labels[j]=='running' and not inferred[j]:
                    step=_step(raw_frames[j])
                    if np.isfinite(step):
                        if abs(t[j]-(t[b] if direction==1 else t[a-1]))<=step/2:
                            nearby.append(step)
                        break
                    j+=direction
                if not nearby or ends[b-1]-t[a]>nearby[0]/2:
                    continue
                if any(row.get('action_event_id') or row.get('stair_sequence_supported') or
                       row.get('gait_evidence') not in ('visible_steps_next_to_confirmed_same_track_walking',
                                                       'gait_type_unresolved', 'moving_partial_gait_unresolved', '')
                       for row in raw_frames[a:b]):
                    continue
                labels[a:b]='running'
                context_bridge[a:b]='brief_inferred_walk_with_observed_running_on_both_sides'
                for i in range(a,b):
                    reviews[i]=review('running',raw_frames[i],True,False,f,i)
    signatures = np.array([f"{k}:{r['annotation']}" for k,r in zip(labels,reviews)],object) if review else labels
    for a, b in label_runs(signatures):
        kind = str(labels[a])
        event_ids = sorted({raw_frames[i]['action_event_id'] for i in range(a, b)} - {0})
        limited = kind in ('unknown', 'body_motion', 'arm_activity', 'bending', 'gait_walk', 'running') or inferred[a:b].any()
        reasons = sorted({raw_frames[i]['activity_reason'] for i in range(a, b)})
        if inferred[a:b].any():
            reasons.append('bounded_temporal_context_inference;not_direct_observation')
        segment = dict(segment_id=len(segments) + 1, activity=kind, activity_zh=activity_labels[kind],
                       action_event_id=0, event_ids=','.join(str(i) for i in event_ids),
                       repetition_count=len(event_ids), start_video_frame=a + 1, end_video_frame=b,
                       end_exclusive_video_frame=b + 1, start_video_s=float(t[a]), end_video_s=float(ends[b - 1]),
                       duration_s=float(ends[b - 1] - t[a]), evidence_level='limited' if limited else 'moderate',
                       reason=';'.join(reasons), visible_body_fraction=float(f['torso_visible'][a:b].mean()),
                       visible_feet_fraction=float(feet[a:b].mean()), context_inferred_fraction=float(inferred[a:b].mean()),
                       camera_compensated_fraction=float(np.isfinite(f['hip_speed_raw'][a:b]).mean()),
                       reviewed=0, needs_review=1)
        if reviews is not None:
            segment.update(reviews[a], activity_zh=reviews[a]['annotation'])
            segment['evidence_level'] = 'limited' if reviews[a]['activity_status'] != 'supported' else segment['evidence_level']
        if (context_bridge[a:b] != '').any():
            segment.update(evidence_level='limited',
                           context_bridge_frames=int(np.count_nonzero(context_bridge[a:b] != '')))
        segments.append(segment)
        for i in range(a, b):
            row = dict(raw_frames[i])
            event = event_map.get(row['action_event_id'])
            phase = ('down' if t[i] <= event['peak_video_s'] else 'up') if event else 'between_reps' if protected[i] else ''
            observed = 'context_inferred' if inferred[i] else 'unobserved' if kind == 'unknown' else 'partial' if not feet[i] else 'observed'
            row.update(raw_activity=row['activity'], activity=kind, activity_zh=activity_labels[kind],
                       segment_id=segment['segment_id'], phase=phase, observation_status=observed,
                       context_inferred=int(inferred[i]), is_segment_start=int(i == a), is_segment_end=int(i == b - 1),
                       evidence_level='limited' if observed != 'observed' or limited else 'moderate')
            if reviews is not None:
                row.pop('action_hypotheses',None)
                row.update(reviews[i], activity_zh=reviews[i]['annotation'])
                if reviews[i]['activity_status'] != 'supported':
                    row['evidence_level'] = 'limited'
            if context_bridge[i]:
                row['context_bridge_reason'] = context_bridge[i]
            frames.append(row)
        for identity in event_ids:
            event_map[identity]['activity_segment_id'] = segment['segment_id']
    boundaries = []
    for index, row in enumerate(segments):
        at = row['start_video_s']
        boundaries.append(dict(boundary_id=index + 1, video_frame=row['start_video_frame'], video_time_s=at,
                               from_activity=segments[index - 1]['activity'] if index else 'VIDEO_START',
                               to_activity=row['activity'], context_start_s=max(float(t[0]), at - config['boundary_uncertainty_s']),
                               context_end_s=min(float(ends[-1]), at + config['boundary_uncertainty_s']),
                               uncertainty_definition='review_context_not_a_calibrated_error_bound',
                               source='automatic_hierarchical_visual_rules', reviewed=0, needs_review=1))
    boundaries.append(dict(boundary_id=len(boundaries) + 1, video_frame=len(t) + 1, video_time_s=float(ends[-1]),
                           from_activity=segments[-1]['activity'], to_activity='VIDEO_END',
                           context_start_s=float(ends[-1]), context_end_s=float(ends[-1]),
                           uncertainty_definition='video_end_last_frame_duration_estimated', source='source_pts',
                           reviewed=0, needs_review=1))
    if reviews is not None:
        for i, boundary in enumerate(boundaries):
            boundary['from_annotation'] = segments[i-1]['annotation'] if i else ''
            boundary['to_annotation'] = segments[i]['annotation'] if i < len(segments) else ''
    return segments, frames, events, boundaries
