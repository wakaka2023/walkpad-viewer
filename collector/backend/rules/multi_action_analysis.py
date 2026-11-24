import numpy as np

from .action_calibration import calibrate
from . import action_features as features
from . import action_rules as rules
from .squat_bouts import adaptive_cycle_recovery, normalize_squat_cycles
from .multi_labels import LABELS, MAIN_ACTIONS, annotation, review
from .stair_evidence import detect as detect_stairs
from .activity_rules import runs

MULTI_LABELS = dict(rules.ACTIVITY_LABELS, **LABELS)
MULTI_LABELS['gait_walk'] = LABELS['walking']

TABLES = {
    'action_calibration': ['track_id','track_fragment','quiet_reference_frames'],
    'gait_regimes': ['track_id','track_fragment','activity','start_video_s','end_video_s','median_step_hz'],
    'activities': ['track_id', 'track_fragment', 'segment_id', 'activity', 'start_video_s', 'end_video_s'],
    'frame_labels': ['track_id', 'track_fragment', 'video_frame', 'video_time_s', 'activity', 'track_state'],
    'action_events': ['track_id', 'track_fragment', 'event_id', 'activity', 'start_video_s', 'peak_video_s', 'end_video_s'],
    'boundary_events': ['track_id', 'track_fragment', 'boundary_id', 'video_frame', 'video_time_s'],
    'raw_activities': ['track_id', 'track_fragment', 'segment_id', 'activity', 'start_video_s', 'end_video_s'],
    'raw_frame_labels': ['track_id', 'track_fragment', 'video_frame', 'video_time_s', 'activity'],
    'other_pose_candidates': ['track_id', 'track_fragment', 'activity', 'start_video_s', 'peak_video_s', 'end_video_s'],
    'rejected_candidates': ['track_id', 'track_fragment', 'peak_video_s', 'reason'],
}


def fragments(indices, times, ends, max_gap, raw_ids=None):
    cut = times[indices[1:]] - ends[indices[:-1]] > max_gap + 1e-9
    if raw_ids is not None:
        cut |= raw_ids[1:] != raw_ids[:-1]
    cuts = np.flatnonzero(cut) + 1
    return np.split(indices, cuts)


def camera_for_fragment(camera, lo, hi, *, bracket=False):
    idx = camera['source_indices']
    start, stop = np.searchsorted(idx, [lo, hi])
    names = {'source_indices', 'matrices', 'valid'}
    if bracket and 'time_s' in camera:
        # Camera transforms describe the background, not another person's
        # poses. Keep the samples bracketing the fragment's first/last frame.
        start = max(0, int(np.searchsorted(idx, lo, side='right'))-1)
        stop = min(len(idx), int(np.searchsorted(idx, hi-1))+1)
        names.add('time_s')
    return {key: (data[start:stop] - lo if key == 'source_indices' else data[start:stop])
            for key, data in camera.items() if key in names}


def bind_rows(rows, *, lo, identity, fragment, event_offset=0, segment_offset=0, boundary_offset=0):
    result = []
    for row in rows:
        out = dict(row, track_id=identity, track_fragment=fragment)
        for key, value in row.items():
            if value is None:
                continue
            if key == 'video_frame' or key.endswith('_video_frame'):
                out[key] = int(value) + lo
            elif key in ('event_id', 'action_event_id') and value:
                out[key] = int(value) + event_offset
            elif key == 'event_ids':
                out[key] = ','.join(str(int(x) + event_offset) for x in value.split(',') if x)
            elif key in ('segment_id', 'activity_segment_id') and value:
                out[key] = int(value) + segment_offset
            elif key == 'boundary_id':
                out[key] = int(value) + boundary_offset
        result.append(out)
    return result


def unknown_frame(i, t, ends, identity, fragment, state, reason):
    return dict(track_id=identity, track_fragment=fragment, video_frame=i + 1,
                video_time_s=float(t[i]), frame_end_s=float(ends[i]), segment_id=0,
                activity='unknown', activity_zh=MULTI_LABELS['unknown'],
                **annotation('unknown',(),False), action_event_id=0, phase='', track_state=state, activity_reason=reason,
                context_inferred=0, observation_status='unobserved', evidence_level='limited', needs_review=1, reviewed=0)


def tiered_candidates(f, pose, frame_rows, rejected, config, height):
    valid = f['raw_valid']
    has_event = np.array([r['action_event_id'] > 0 for r in frame_rows])
    calm = np.isfinite(f['camera_speed']) & (f['camera_speed'] <= .16)
    stairs, strength, supported = detect_stairs(f['time_s'], pose, valid, f['torso_px'], calm, has_event, height)
    for i, tag in enumerate(stairs):
        if not tag:
            continue
        row = frame_rows[i]
        extra = ()
        row.update(activity=tag, activity_reason='directional_stair_traversal_and_foot_landing_evidence',
                   action_hypotheses=extra, ambiguous_gait=0, gait_inferred=0, locomotion_partial=0,
                   stair_sequence_supported=int(supported[i]), stair_step_projection_torso_ratio=float(strength[i]))
    t = f['time_s']
    knee = f['knee_flexion']
    for r in rejected:
        if r['reason'] not in ('missing_supported_hip_descent_recovery','no_observed_cycle_crossing'):
            continue
        peak = min(len(t)-1,int(np.searchsorted(t,r['peak_video_s'])))
        threshold = r['peak_knee_flexion_deg']-.65*r['knee_excursion_deg']
        plausible = np.isfinite(knee) & (knee >= threshold) & (np.abs(t-t[peak]) <= config['cycle_max_s']/2)
        for a,b in runs(plausible):
            if not a <= peak < b or t[b-1]-t[a] < .25:
                continue
            for i in range(a,b):
                row = frame_rows[i]
                if row['activity'] in ('unknown','standing') and not has_event[i] and f['torso_visible'][i]:
                    row.update(activity='squat', activity_reason='grounded_knee_flexion_without_complete_hip_or_boundary_support',
                               action_hypotheses=())


def analyze(data, camera, config, min_seen_s, max_gap_s, *, cropped=None):
    t, ends = data['time_s'], data['frame_end_s']
    output = {key: [] for key in TABLES}
    summaries = []
    for identity in np.unique(data['track_id']):
        identity = int(identity)
        obs = np.flatnonzero(data['track_id'] == identity)
        indices = data['frame_index'][obs]
        if len(indices) and np.any(np.diff(indices) <= 0):
            raise ValueError('Track cache has duplicate or unordered frames for an ID.')
        measured = indices[~data['cropped_position_inferred'][obs]] if 'cropped_position_inferred' in data else indices
        seconds = float(np.sum(ends[measured] - t[measured]))
        accepted = seconds >= min_seen_s and len(measured) >= 3
        summary = dict(track_id=identity, accepted=accepted, observed_frames=len(measured),
                       observed_seconds=seconds, first_video_frame=int(indices[0] + 1),
                       last_video_frame=int(indices[-1] + 1), start_video_s=float(t[indices[0]]),
                       end_video_s=float(ends[indices[-1]]), fragment_count=0, action_event_count=0)
        summaries.append(summary)
        if not accepted:
            continue
        raw_ids = data.get('yolo_track_id', data['track_id'])[obs]
        summary['yolo_track_ids'] = ','.join(str(x) for x in dict.fromkeys(raw_ids))
        parts = fragments(indices, t, ends, max_gap_s, raw_ids)
        totals = dict(event_offset=0, segment_offset=0, boundary_offset=0)
        raw_segment_offset = 0
        from .gait_switch import observations, sample_references, profile, apply as refine_gait_switch
        prepared = {}
        refs = {'walk': [], 'run': []}
        for n, part in enumerate(parts, 1):
            lo, hi = int(part[0]), int(part[-1] + 1)
            active = np.zeros(hi - lo, bool)
            active[part - lo] = True
            if hi - lo < 3:
                continue
            pose = np.full((hi - lo, 17, 3), np.nan, np.float32)
            observed_rows = obs[np.searchsorted(indices, part)]
            pose[active] = data['keypoints'][observed_rows]
            f = features.pose_features(dict(time_s=t[lo:hi], keypoints=pose), config)
            f = features.camera_features(f, camera_for_fragment(camera, lo, hi), config,
                                         int(data['width']), int(data['height']))
            rules.period_features(f, config)
            from .action_temporal import assemble
            from .gait_evidence import judge

            local_config, calibration = calibrate(f, config)
            output['action_calibration'].append(dict(calibration, track_id=identity, track_fragment=n))
            legacy_segments, legacy_frames, _, _, _ = rules.classify_raw(f, config)
            # Unsupported projected bends are audit evidence, not a priority motion seed.
            seed_segments, seed_frames, events, _, rejected = rules.classify_raw(
                f, local_config, arm_semantics=False, bend_semantics=False)
            audit_bends, bend_rejected = rules.bend_events(f, local_config, return_rejected=True)
            rejected.extend(bend_rejected)
            for bend in audit_bends:
                bend['classification_role'] = 'diagnostic_projection_only_not_a_main_action'
            events, displaced = adaptive_cycle_recovery(f, events, seed_frames, local_config)
            assessments, gait_report = judge(f, camera_for_fragment(camera, lo, hi), pose, seed_frames, local_config)
            rows_with_evidence = [{**row, **evidence} for row, evidence in zip(seed_frames, assessments)]
            normalize_squat_cycles(f, active, events, rows_with_evidence, local_config)
            tiered_candidates(f, pose, rows_with_evidence, rejected, local_config, int(data['height']))
            if cropped and identity in cropped:
                from .cropped_gait import apply
                apply(rows_with_evidence,cropped[identity],lo)
            measured_active = active.copy()
            if 'cropped_position_inferred' in data:
                measured_active[active] = ~data['cropped_position_inferred'][observed_rows]
            from .adaptive_cycle_gait import apply as refine_cycle_gait
            valid = np.isfinite(pose).all(axis=2) & (pose[:, :, 2] >= local_config['keypoint_conf'])
            refine_cycle_gait(f['time_s'], pose, valid, rows_with_evidence, measured_active)
            fragment_camera = camera_for_fragment(camera, lo, hi, bracket=True)
            shape, step = observations(f['time_s'], pose, valid, rows_with_evidence, measured_active, fragment_camera)
            anchors = sample_references(f['time_s'], rows_with_evidence, shape, step)
            for k in refs:
                refs[k].extend(anchors[k])
            prepared[n] = (active, observed_rows, pose, f, local_config, legacy_segments, legacy_frames,
                           seed_segments, events, rejected, displaced, audit_bends, gait_report,
                           rows_with_evidence, measured_active, valid, fragment_camera, shape, step)
        profiles = profile(refs)
        previous_stop = int(indices[0])
        for n, part in enumerate(parts, 1):
            lo, hi = int(part[0]), int(part[-1] + 1)
            for i in range(previous_stop, lo):
                row = unknown_frame(i, t, ends, identity, 0, 'not_tracked', 'track_gap_between_fragments')
                output['frame_labels'].append(row)
                output['raw_frame_labels'].append(row.copy())
            previous_stop = hi
            active = np.zeros(hi - lo, bool)
            active[part - lo] = True
            if hi - lo < 3:
                for i in range(lo, hi):
                    state = 'tracked' if active[i - lo] else 'not_tracked'
                    row = unknown_frame(i, t, ends, identity, n, state, 'insufficient_fragment_length')
                    output['frame_labels'].append(row)
                    output['raw_frame_labels'].append(row.copy())
                continue
            (active, observed_rows, pose, f, local_config, legacy_segments, legacy_frames,
             seed_segments, events, rejected, displaced, audit_bends, gait_report,
             rows_with_evidence, measured_active, valid, fragment_camera, shape, step) = prepared.pop(n)
            refine_gait_switch(f['time_s'], rows_with_evidence, shape, step, profiles)
            from .running_turn import apply as refine_running_turn
            refine_running_turn(f['time_s'], pose, valid, rows_with_evidence, measured_active, fragment_camera)
            from .bilateral_gait_continuity import apply as refine_bilateral_gait
            refine_bilateral_gait(f['time_s'], ends[lo:hi], pose, valid, rows_with_evidence, measured_active)
            from .leg_boundary_evidence import recover_clipped_walk, recover_squat_edges, recover_walk_handoffs
            recover_clipped_walk(f['time_s'], ends[lo:hi], pose, valid, rows_with_evidence,
                                 measured_active, fragment_camera, int(data['height']))
            recover_walk_handoffs(f['time_s'], ends[lo:hi], pose, valid, rows_with_evidence,
                                  measured_active, fragment_camera)
            recover_squat_edges(f['time_s'], pose, valid, rows_with_evidence, measured_active, f,
                                events, fragment_camera)
            from .standing_turn import apply as refine_standing_turn
            refine_standing_turn(f['time_s'], pose, valid, rows_with_evidence, measured_active,
                                 fragment_camera, local_config['pose_smooth_s'])
            for origin, row in zip(legacy_frames, rows_with_evidence):
                row['activity_zh'] = MULTI_LABELS[row['activity']]
                row['legacy_raw_activity'] = origin['activity']
            other = displaced + audit_bends + [row for row in events if row['activity'] != 'squat']
            output['other_pose_candidates'].extend(bind_rows(other, lo=lo, identity=identity, fragment=n))
            events = [row for row in events if row['activity'] == 'squat']
            mapping = {row['event_id']: j for j, row in enumerate(events, 1)}
            for row in events:
                row['event_id'] = mapping[row['event_id']]
            for row in rows_with_evidence:
                row['action_event_id'] = mapping.get(row['action_event_id'], 0)
            segs, frames, events, boundaries = assemble(f, seed_segments, rows_with_evidence, events, config,
                                                       MULTI_LABELS, merge_walk_labels=True, main_actions=MAIN_ACTIONS, review=review,
                                                       measured=measured_active,
                      walk_observations=(pose, valid, fragment_camera), image_height=int(data['height']))
            f['raw_segments'], f['raw_frame_labels'] = legacy_segments, legacy_frames
            output['gait_regimes'].extend(dict(row, track_id=identity, track_fragment=n) for row in gait_report)
            for row in segs:
                local = rows_with_evidence[row['start_video_frame']-1:row['end_video_frame']]
                row['partial_locomotion_frame_fraction'] = sum(r['locomotion_partial'] for r in local)/len(local)
            for row in [*segs, *f['raw_segments']]:
                row['end_video_s'] = float(ends[lo + row['end_video_frame'] - 1])
                row['duration_s'] = row['end_video_s'] - row['start_video_s']
            for row in events:
                a, b = row['start_video_frame'] - 1, row['end_video_frame']
                row['track_observed_fraction'] = float(active[a:b].mean())
                row['peak_track_observed'] = int(active[row['peak_video_frame'] - 1])
                row['contains_untracked_frames'] = int(not active[a:b].all())
            boundaries[0].update(from_activity='TRACK_FRAGMENT_START', source='track_fragment_and_visual_rules')
            boundaries[-1].update(video_time_s=float(ends[hi-1]), to_activity='TRACK_FRAGMENT_END',
                                  context_start_s=float(ends[hi-1]), context_end_s=float(ends[hi-1]),
                                  uncertainty_definition='end_of_track_fragment_not_an_action_label', source='source_pts')
            for name, group in [('activities', segs), ('frame_labels', frames), ('action_events', events),
                                ('boundary_events', boundaries), ('rejected_candidates', rejected),
                                ('raw_activities', f['raw_segments']), ('raw_frame_labels', f['raw_frame_labels'])]:
                opts = dict(totals)
                if name.startswith('raw_'):
                    opts['segment_offset'] = raw_segment_offset
                rows = bind_rows(group, lo=lo, identity=identity, fragment=n, **opts)
                if name in ('frame_labels', 'raw_frame_labels'):
                    detection = iter(observed_rows)
                    for i, row in enumerate(rows):
                        row['frame_end_s'] = float(ends[lo+i])
                        row['track_state'] = 'tracked' if active[i] else 'not_tracked'
                        if not active[i]:
                            row['candidate_without_track'] = row['activity']
                            missing = unknown_frame(lo+i, t, ends, identity, n, 'not_tracked', 'no_matched_yolo_track')
                            row.update(missing, segment_id=row['segment_id'])
                            for key in ('gait_step_hz_candidate','gait_rhythm_score','gait_phase_signed','gait_evidence','gait_inferred','locomotion_partial','ambiguous_gait','stair_sequence_supported','stair_step_projection_torso_ratio'):
                                row.pop(key,None)
                        else:
                            j = next(detection)
                            row['yolo_track_id'] = int(data.get('yolo_track_id', data['track_id'])[j])
                            row['detection_confidence'] = (float(data['confidence'][j]) if np.isfinite(data['confidence'][j]) else None)
                            if data.get('cropped_position_inferred',np.empty(0)).size and data['cropped_position_inferred'][j]:
                                row.update(track_state='image_supported_continuation',context_inferred=1,observation_status='context_inferred')
                            row.update(zip(('x1', 'y1', 'x2', 'y2'), map(float, data['boxes'][j])))
                for row in rows:
                    if name not in ('frame_labels', 'raw_frame_labels'):
                        row['yolo_track_id'] = int(data.get('yolo_track_id', data['track_id'])[observed_rows[0]])
                output[name].extend(rows)
            totals['event_offset'] += len(events)
            totals['segment_offset'] += len(segs)
            totals['boundary_offset'] += len(boundaries)
            raw_segment_offset += len(f['raw_segments'])
        summary['fragment_count'] = len(parts)
        summary['action_event_count'] = totals['event_offset']
    return output, summaries

