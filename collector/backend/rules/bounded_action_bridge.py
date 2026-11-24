import numpy as np


CONTINUOUS = frozenset({'standing', 'walking', 'running', 'stairs_up', 'stairs_down'})
WALK_CANDIDATE_EVIDENCE = 'translation_with_step_motion_but_unresolved_gait'


def bridge(t, ends, labels, rows, inferred, protected, reviews, measured, torso_visible, config):
    measured = np.asarray(measured, bool)
    visible = np.asarray(torso_visible, bool) & measured
    result = np.full(len(t), '', dtype=object)
    changes = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(t)]
    available = visible & ~inferred & ~protected

    def support(edge, step, kind):
        strong, weak = 0., 0.
        i = edge
        while 0 <= i < len(t) and labels[i] == kind and available[i]:
            row, rv = rows[i], reviews[i]
            raw = row['activity']
            if ('walking' if raw == 'gait_walk' else raw) != kind or row['action_event_id']:
                break
            duration = ends[i] - t[i]
            if rv['activity_status'] == 'supported':
                strong += duration
                weak += duration
            elif (kind == 'walking' and rv['action_candidates'] == 'walking'
                  and row.get('gait_evidence') == WALK_CANDIDATE_EVIDENCE):
                weak += duration
                i += step
                while (0 <= i < len(t) and labels[i] == kind and available[i]
                       and not rows[i]['action_event_id']
                       and rows[i]['activity'] in ('gait_walk', 'walking')
                       and reviews[i]['action_candidates'] == 'walking'
                       and rows[i].get('gait_evidence') == WALK_CANDIDATE_EVIDENCE):
                    weak += ends[i] - t[i]
                    i += step
                break
            else:
                break
            i += step
        return strong, weak

    def gait_period(i):
        row = rows[i]
        value = row.get('personal_gait_step_period_s')
        if value is None:
            hz = row.get('gait_step_hz_candidate')
            value = 1./hz if hz is not None and np.isfinite(hz) and hz>0 else np.nan
        return float(value) if np.isfinite(value) and value>0 else np.nan

    eps = 1e-9
    anchor = config['continuity_anchor_s']
    for a,b in zip(changes[:-1],changes[1:]):
        if a == 0 or b == len(t) or labels[a] != 'unknown':
            continue
        kind = labels[a-1]
        if kind not in CONTINUOUS or labels[b] != kind or not visible[a:b].all() or protected[a:b].any():
            continue
        dur = ends[b-1] - t[a]
        if any(rows[i]['action_event_id'] or reviews[i]['activity_status'] != 'other'
               or rows[i].get('action_hypotheses') for i in range(a,b)):
            continue
        left, lw = support(a-1,-1,kind)
        right, rw = support(b,1,kind)
        if kind in ('walking','running'):
            p0,p1 = gait_period(a-1),gait_period(b)
            step = np.sqrt(p0*p1)
            moving_only = all(rows[i]['activity']=='body_motion' and rows[i].get('gait_evidence')=='gait_type_unresolved'
                              for i in range(a,b))
            anchors = min(left, right) + eps >= min(p0,p1)/2
            brief_moving = (moving_only and min(left,right)+eps>=dur and max(left,right)+eps>=step/2
                            and dur<=step/2+eps)
            if np.isfinite(step) and (anchors or brief_moving):
                unresolved = all(rows[i]['activity']=='unknown' or
                                 (rows[i]['activity']=='body_motion' and rows[i].get('gait_evidence') in
                                  ('gait_type_unresolved','moving_partial_gait_unresolved')) for i in range(a,b))
                if unresolved and dur <= p0+p1+eps:
                    result[a:b] = 'same_gait_motion_gap_within_observed_stride'
                    continue
        if dur > config['continuity_gap_s'] + eps:
            continue
        if min(left,right) + eps >= anchor and all(rows[i]['activity'] == 'unknown' for i in range(a,b)):
            result[a:b] = 'short_other_same_direct_action'
        elif (kind == 'walking' and dur <= config['continuity_walk_motion_gap_s'] + eps
              and max(left,right) + eps >= anchor
              and min(lw,rw) + eps >= config['continuity_walk_candidate_anchor_s']
              and all(rows[i]['activity'] == 'body_motion' and rows[i].get('gait_evidence') == 'gait_type_unresolved'
                      for i in range(a,b))):
            result[a:b] = 'short_unresolved_motion_with_walking_context'
    return result
