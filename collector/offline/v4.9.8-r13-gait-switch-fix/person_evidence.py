import numpy as np

from activity_rules import runs


OBSERVATION_FIELDS = ('frame_index', 'track_id', 'boxes', 'confidence', 'keypoints')
AUDIT_FIELDS = ('yolo_track_id', 'status', 'reason', 'observed_frames', 'confirmed_body_frames',
                'edge_lower_body_frames', 'weak_or_better_frames', 'removed_observations',
                'start_video_s', 'end_video_s')


def pose_evidence(box, pose, image_height, conf):
    box = np.asarray(box, float)
    pose = np.asarray(pose, float)
    size = box[:, 2:] - box[:, :2]
    unit = np.maximum(size[:, 0], size[:, 1])
    unit = np.where(np.isfinite(unit) & (size > 0).all(axis=1), unit, np.nan)
    xy = pose[:, :, :2]
    margin = .10 * unit[:, None, None]
    valid = (np.isfinite(pose).all(axis=2) & (pose[:, :, 2] >= conf)
             & (xy >= box[:, None, :2] - margin).all(axis=2)
             & (xy <= box[:, None, 2:] + margin).all(axis=2))
    count = valid.sum(axis=1)

    def link(a, b, lower=.040, upper=.81):
        d = np.linalg.norm(xy[:, a] - xy[:, b], axis=1)
        return valid[:, a] & valid[:, b] & (d >= lower*unit) & (d <= upper*unit)

    trunk = link(5, 11, .09, .82) | link(6, 12, .09, .82)
    arms = link(5, 7) | link(6, 8)
    thighs = link(11, 13) | link(12, 14)
    shins = link(13, 15) | link(14, 16)
    head = np.any([link(h, shoulder, .04, .58) for h in (0, 1, 2, 3, 4) for shoulder in (5, 6)], axis=0)
    shoulders = link(5, 6, .055, .72)
    upper_body = shoulders & head & arms & (count >= 5)
    body = (trunk & (arms | thighs | (head & shoulders)) & (count >= 5)) | upper_body
    edge = ((size[:, 1] >= .44*image_height) & (box[:, 1] <= .18*image_height)
            & (valid[:, :7].sum(axis=1) <= 1))
    two_legs = (link(13, 14, .042, .79) | (shins & (valid[:, 13:17].sum(axis=1) >= 3)))
    lower = edge & two_legs
    weak = (count >= 3) & (trunk | arms | thighs | shins | head | two_legs)
    return body, lower, weak | body | lower


def sustained(t, evidence, minimum_span, max_between):
    positive = t[evidence]
    for block in np.split(positive, np.flatnonzero(np.diff(positive) > max_between + 1e-9) + 1):
        if len(block) >= 3 and block[-1] - block[0] >= minimum_span - 1e-9:
            return True
    return False


def filter_person_tracks(data, config):
    ids = data['track_id']
    if not len(ids):
        return data, []
    t, ends = data['time_s'], data['frame_end_s']
    body, lower, weak = pose_evidence(data['boxes'], data['keypoints'], float(data['height']), config['keypoint_conf'])
    keep = np.ones(len(ids), bool)
    audit = []
    for identity in np.unique(ids):
        obs = np.flatnonzero(ids == identity)
        idx = data['frame_index'][obs]
        own_time = t[idx]
        regular = bool(sustained(own_time, body[obs] | lower[obs], config['person_pose_confirm_s'],
                                 config['person_pose_max_sample_gap_s']))
        if not regular:
            keep[obs] = False
        else:
            for a, b in runs(~weak[obs]):
                if b-a >= 3 and ends[idx[b-1]] - t[idx[a]] >= config['person_unsupported_span_s'] - 1e-9:
                    keep[obs[a:b]] = False
        removed = int(np.count_nonzero(~keep[obs]))
        audit.append(dict(yolo_track_id=int(identity), status='person_supported' if regular else 'rejected_person_evidence',
                          reason=('insufficient_repeated_human_pose' if not regular else
                                  'unsupported_observations_removed' if removed else 'repeated_human_pose'),
                          observed_frames=len(obs), confirmed_body_frames=int(np.count_nonzero(body[obs])),
                          edge_lower_body_frames=int(np.count_nonzero(lower[obs])),
                          weak_or_better_frames=int(np.count_nonzero(weak[obs])), removed_observations=removed,
                          start_video_s=float(t[idx[0]]), end_video_s=float(ends[idx[-1]])))
    if keep.all():
        return data, audit
    return dict(data, **{key:data[key][keep] for key in OBSERVATION_FIELDS}), audit
