import numpy as np

from action_features import window_stat
from action_rules import clean_mask
from activity_rules import runs
from gait_phase import classify_windows
from running_rebound import detect as detect_short_running
from personal_short_run import detect as detect_personal_short_run
from gait_ground import advancing_walk, bracketed_walk_gaps, continuity, grounded_walk, motion_above_own_baseline, nearby_transition, walk_edges

def scale_for_motion(f, valid):
    p = f['points']
    shoulder_width = np.linalg.norm(p[:, 5] - p[:, 6], axis=1)
    torso = f['torso_px']
    both = valid[:, [5, 6]].all(axis=1) & np.isfinite(torso) & (shoulder_width > 8)
    if both.sum() >= 8:
        ratio = np.median(torso[both] / shoulder_width[both])
        if .45 <= ratio <= 3.5:
            torso = np.where(np.isfinite(torso), torso, shoulder_width * ratio)
    return torso


def project_track_motion(f, camera, valid):
    t, p = f['time_s'], f['points']
    scale = scale_for_motion(f, valid)
    idx = camera['source_indices']
    velocities = np.full((len(idx), 2), np.nan)
    widths = np.full(len(idx), np.nan)
    camera_motion = np.full(len(idx), np.nan)
    projection_noise = np.full(len(idx), np.nan)
    landmarks = [0, 3, 4, 5, 6, 11, 12]
    for j in range(1, len(idx)):
        a, b = idx[j-1:j+1]
        dt = t[b] - t[a]
        if not camera['valid'][j] or dt > .25 or not np.isfinite(scale[[a, b]]).all():
            continue
        before, after = p[a, landmarks], p[b, landmarks]
        good = valid[a, landmarks] & valid[b, landmarks] & np.isfinite(before).all(axis=1) & np.isfinite(after).all(axis=1)
        if np.count_nonzero(good) < 2 or not good[3:].any():
            continue
        mat = camera['matrices'][j]
        unit = np.mean(scale[[a, b]])
        moves = (after[good] - (before[good] @ mat[:, :2].T + mat[:, 2])) / (dt * unit)
        median = np.median(moves, axis=0)
        residual = np.median(np.linalg.norm(moves - median, axis=1))
        if residual > .48:
            continue
        precision = np.spacing(np.float32(max(1., np.abs(before[good]).max(), np.abs(after[good]).max())))
        projection_noise[j] = residual + 4*precision/(dt*unit)
        velocities[j] = median
        camera_motion[j] = np.median(np.linalg.norm(before[good] @ mat[:, :2].T + mat[:, 2] - before[good], axis=1)) / (dt * unit)
        if valid[[a, b]][:, [5, 6]].all():
            old = np.linalg.norm((p[a, 5] - p[a, 6]) @ mat[:, :2].T)
            new = np.linalg.norm(p[b, 5] - p[b, 6])
            if min(old, new) >= 8:
                widths[j] = np.log(new/old) / dt
    def expand(values):
        result = np.full(len(t), np.nan)
        for a, b in runs(np.isfinite(values)):
            here = (t >= t[idx[a]]) & (t <= t[idx[b-1]])
            result[here] = np.interp(t[here], t[idx[a:b]], values[a:b])
        return result
    vx, vy, width, cam = (expand(x) for x in [velocities[:, 0], velocities[:, 1], widths, camera_motion])
    vx, vy = [window_stat(t, v, .66, 'mean', .63) for v in (vx, vy)]
    path = window_stat(t, np.hypot(expand(velocities[:, 0]), expand(velocities[:, 1])), .66, 'mean', .63)
    speed = np.hypot(vx, vy)
    width = window_stat(t, width, .75, 'mean', .6)
    cam = window_stat(t, cam, .75, 'mean', .6)
    noise = window_stat(t, expand(projection_noise), .66, 'mean', .63) + np.maximum(0.,path-speed)
    return vx, vy, speed, speed/np.maximum(.01, path), width, cam, scale, noise


def judge(f, camera, pose, frame_rows, config):
    t = f['time_s']
    valid = np.isfinite(pose).all(axis=2) & (pose[:, :, 2] >= config['keypoint_conf'])
    labels = np.array([r['activity'] for r in frame_rows], object)
    event = np.array([r['action_event_id'] for r in frame_rows]) > 0
    feet = valid[:, 15:17].any(axis=1)
    foot_ratio = window_stat(t, feet.astype(float), .8, 'mean')
    part = foot_ratio < .38
    vx, vy, speed, _, width, cam, scale, noise = project_track_motion(f, camera, valid)
    quiet_cam = (cam < .24) | (f['camera_speed'] < .20)
    reference = (labels == 'standing') & ~event & quiet_cam & f['torso_visible']
    shifting = motion_above_own_baseline(speed, reference, allow_lower_tail=True)
    lateral = motion_above_own_baseline(abs(vx), reference, allow_lower_tail=True)
    axial = motion_above_own_baseline(abs(vy), reference, allow_lower_tail=True)
    perspective = motion_above_own_baseline(abs(width), reference, allow_lower_tail=True) & (np.sign(vy) == np.sign(width))
    if np.count_nonzero(reference & np.isfinite(speed)) < 8:
        sustained = speed > noise
        shifting |= sustained
        lateral |= abs(vx) > noise
        axial |= abs(vy) > noise
    displacement = shifting & (lateral | (axial & perspective)) & (speed > noise) & ~event & quiet_cam
    stable = clean_mask(t, displacement, displacement, .39, 0.)
    knee_visible = window_stat(t, valid[:, 13:15].any(axis=1).astype(float), .7, 'mean') >= .40
    leg_active = ((motion_above_own_baseline(f['knee_motion'], reference, allow_lower_tail=True) & knee_visible) |
                  (motion_above_own_baseline(f['foot_speed'], reference, allow_lower_tail=True) & ~part))
    eligible = np.isin(labels, ['unknown', 'standing', 'body_motion', 'stepping', 'fast_stepping', 'walking'])
    visible_walk = stable & eligible & leg_active & ~part
    step_anchor = stable & (leg_active | ((labels == 'walking') & ~part))
    step_anchor = clean_mask(t, step_anchor, stable, config['pose_gap_s'], 0.)
    nearby = np.zeros(len(t), bool)
    for start, stop in runs(stable):
        for a, b in runs(step_anchor[start:stop]):
            lo, hi = np.searchsorted(t, [t[start+a]-1.05, t[start+b-1]+1.05])
            nearby[max(start, lo):min(stop, hi)] = True
    partial_walk = stable & eligible & part & (leg_active | nearby)
    cropped_move = stable & eligible & part & ~partial_walk
    leg_seed = (labels == 'walking') & leg_active & ~part & ~event
    local_walk = visible_walk | leg_seed
    grounded_candidate = visible_walk | ((labels == 'walking') & ~part)
    short_window_camera = (cam < .12) | (~np.isfinite(cam) & (f['camera_speed'] < .12))
    gait_window = ~event & f['torso_visible'] & quiet_cam
    regimes, phase, report = classify_windows(t, pose, valid, scale, gait_window, quiet_cam, stable, speed)
    step_batches = []
    new_run, slow_conflict, new_hz, rebound = detect_short_running(
        t, pose, valid, gait_window & ~part, scale, step_batches=step_batches)
    conflicts = np.zeros(len(t),bool)
    for a,b in runs(regimes == 'running'):
        if not new_run[a:b].any() and slow_conflict[a:b].mean() >= .60:
            conflicts[a:b] = True
    if new_run.any() or conflicts.any():
        inherited = report
        regimes[conflicts] = ''
        regimes[new_run] = 'running'
        phase['inferred_step_hz'][new_run] = new_hz[new_run]
        phase['running_support'][new_run | conflicts] = 1
        phase['walking_support'][new_run] = 0
        phase['inferred'][new_run] = False
        report = []
        dt = float(np.median(np.diff(t)))
        for label in ('gait_walk','running'):
            for a,b in runs(regimes == label):
                basis = next((r['basis'] for r in inherited if r['activity']==label and r['start_video_s'] <= t[a] < r['end_video_s']), 'legacy_projected_phase')
                report.append(dict(activity=label,start_video_s=float(t[a]),end_video_s=float(t[b] if b<len(t) else t[b-1]+dt),
                                   median_step_hz=float(np.nanmedian(phase['inferred_step_hz'][a:b])),
                                   signed_phase_median=(float(np.nanmedian(phase['midstance_phase_signed'][a:b])) if np.isfinite(phase['midstance_phase_signed'][a:b]).any() else None),
                                   inferred_frames=int(phase['inferred'][a:b].sum()),
                                   basis='repeated_projected_rebound_and_alternating_steps' if new_run[a:b].any() else basis))
        report.sort(key=lambda r:r['start_video_s'])
    short_grnd = grounded_walk(t, pose, valid, scale, (~event & grounded_candidate),
                               short_window_camera, regimes)
    short_grnd &= ~event & (regimes == '') & (phase['running_support'] != 1)
    transition = nearby_transition(t, regimes)
    walking_anchors = short_grnd | ((regimes == 'gait_walk') & ~phase['inferred'].astype(bool))
    connectable = local_walk & ~event & ~transition
    allow_context = ~event & ~transition & (phase['running_support'] != 1) & (regimes != 'running') & short_window_camera
    useful_context = f['torso_visible'] & (~reference | shifting | leg_active)
    connectable = clean_mask(t, (connectable | walking_anchors) & allow_context, allow_context & useful_context, .0, .23)
    short_fill = continuity(t, walking_anchors, connectable, valid[:,15:17].all(axis=1))
    gap_ok = allow_context & f['torso_visible'] & leg_active & ~part & np.isin(labels, ('unknown','walking','stepping','fast_stepping','body_motion'))
    turn_fill = bracketed_walk_gaps(t, pose, walking_anchors | short_fill, gap_ok,
                                    valid[:,15:17].all(axis=1) & short_window_camera)
    step_states = np.isin(labels, ('walking','stepping','fast_stepping','body_motion','unknown'))
    step_override = advancing_walk(t, pose, valid, scale, allow_context & step_states, short_window_camera)
    supported_walk = walking_anchors | short_fill | turn_fill | step_override
    edge_ok = allow_context & f['torso_visible'] & valid[:,15:17].all(axis=1)
    edge_fill = walk_edges(t, pose, valid, supported_walk, edge_ok & (labels == 'walking'))
    step_edge = walk_edges(t, pose, valid, step_override, edge_ok & np.isin(labels, ('stepping','fast_stepping')))
    edge_fill |= step_edge
    partial_phase = {}
    for kind in ('walking','running'):
        proposed = phase[kind+'_support']==1
        compatible = proposed | (regimes == ('running' if kind=='running' else 'gait_walk'))
        partial_phase[kind] = proposed & clean_mask(t, compatible, compatible, .42, 0.)
    cadence, rhythm = phase['inferred_step_hz'], phase['rhythm_agreement']
    assessments = []
    for i, row in enumerate(frame_rows):
        tag, reason = labels[i], row['activity_reason']
        evidence = ''
        hypotheses = ()
        if not event[i]:
            if new_run[i]:
                tag, reason, evidence = 'running', 'repeated_projected_torso_rebound_synchronized_with_alternating_feet', 'short_running_rebound'
            elif conflicts[i]:
                tag, hypotheses = 'walking', ('walking','running')
                reason, evidence = 'legacy_running_phase_conflicts_with_slow_low_bounce_steps', 'gait_conflict'
            elif regimes[i]:
                tag = regimes[i]
                reason = ('same_track_motion_rate_leg_rhythm_and_hip_oscillation' if phase['profile_mode'][i] else
                          'alternating_ankles_and_same_phase_of_hip_shoulder_vertical_motion')
                evidence = 'running_rhythm_candidate' if tag == 'running' else 'walking_rhythm_candidate'
            elif short_grnd[i]:
                tag = 'gait_walk'
                reason = 'same_track_alternating_swings_with_observed_double_support'
                evidence = 'walking_support_handoff'
            elif step_override[i]:
                tag = 'gait_walk'
                reason = 'consecutive_same_track_support_handoffs_and_advancing_feet'
                evidence = 'walking_advancing_handoffs'
            elif edge_fill[i]:
                tag = 'gait_walk'
                reason = 'visible_steps_next_to_confirmed_same_track_walking'
                evidence = 'walking_bout_edge'
            elif turn_fill[i]:
                tag = 'gait_walk'
                reason = 'same_track_walk_bracket_with_observed_foot_switch_through_translation_gap'
                evidence = 'walking_turn_context'
            elif short_fill[i]:
                tag = 'gait_walk'
                reason = 'short_continuity_of_supported_same_track_walking'
                evidence = 'walking_short_context'
            elif transition[i] and not stable[i]:
                tag, hypotheses = 'walking', ('walking', 'running')
                reason = 'short_transition_between_supported_walking_and_running'
                evidence = 'gait_transition'
            elif partial_phase['walking'][i] or partial_phase['running'][i]:
                hypotheses = tuple(k for k in ('walking','running') if partial_phase[k][i])
                tag = 'walking' if hypotheses[0] == 'walking' else 'running'
                reason = 'ankle_rhythm_with_partial_or_conflicting_torso_phase_support'
                evidence = 'rhythm_candidate'
            elif visible_walk[i] or partial_walk[i] or leg_seed[i]:
                tag = 'walking'
                reason = ('camera_compensated_torso_displacement_with_visible_step_evidence' if visible_walk[i] else
                          'visible_leg_activity_without_reliable_translation' if leg_seed[i] else
                          'partial_view_camera_compensated_translation_with_leg_or_nearby_walk_support')
                hypotheses = ('walking',)
                evidence = 'translation_with_step_motion_but_unresolved_gait'
            elif cropped_move[i]:
                tag, reason, evidence = 'body_motion', 'camera_compensated_translation_with_incomplete_gait_evidence', 'moving_partial_gait_unresolved'
            elif tag in ('walking','stepping','fast_stepping'):
                tag = 'body_motion'
                reason, evidence = 'motion_without_supported_gait_type', 'gait_type_unresolved'
        assessments.append(dict(activity=tag, activity_reason=reason, gait_evidence=evidence,
                                action_hypotheses=hypotheses, ambiguous_gait=int(bool(hypotheses)),
                                visible_foot_fraction_window=float(foot_ratio[i]) if np.isfinite(foot_ratio[i]) else None,
                                gait_rebound_peak_torso_ratio=float(rebound[i]) if new_run[i] else None,
                                gait_step_hz_candidate=float(cadence[i]) if np.isfinite(cadence[i]) else None,
                                gait_rhythm_score=float(rhythm[i]) if np.isfinite(rhythm[i]) else None,
                                gait_phase_signed=float(phase['midstance_phase_signed'][i]) if np.isfinite(phase['midstance_phase_signed'][i]) else None,
                                gait_hip_oscillation_torso_ratio=float(phase['hip_oscillation'][i]) if np.isfinite(phase['hip_oscillation'][i]) else None,
                                gait_inferred=int(bool(turn_fill[i] or edge_fill[i] or short_fill[i] or (phase['inferred'][i] and regimes[i]))),
                                locomotion_partial=int(bool(part[i] and tag in ('walking', 'body_motion')))))
    unresolved = np.array([r['gait_evidence'] == 'gait_type_unresolved' for r in assessments])
    short_candidate = detect_personal_short_run(t, pose, valid, scale, step_batches,
                                                walking_anchors | step_override, new_run,
                                                unresolved & gait_window & ~part)
    for i in np.flatnonzero(short_candidate):
        assessments[i].update(activity='running',
                              activity_reason='same_fragment_walking_running_profiles_favor_short_run',
                              gait_evidence='personal_short_running_candidate',
                              action_hypotheses=('running',), ambiguous_gait=1)
    return assessments, report
