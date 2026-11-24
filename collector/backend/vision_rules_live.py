from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .rules import action_features as features
from .rules import action_rules as rules
from .rules.action_calibration import calibrate
from .rules.gait_evidence import judge
from .rules.stair_evidence import detect as detect_stairs
from .vision_rules_r10 import CONFIG, Camera

WINDOW_S = 6.
UPDATE_S = .40
GAP_S = .30
UNDECIDED = ('未判定', 'unknown')
CORE = [5, 6, 11, 12, 13, 14, 15, 16]
GAIT = {'walking': '步行', 'gait_walk': '步行', 'running': '跑步',
        'stairs_up': '上楼梯', 'stairs_down': '下楼梯'}


@dataclass
class Track:
    seen: float
    samples: deque = field(default_factory=lambda: deque(maxlen=240))
    posture: deque = field(default_factory=lambda: deque(maxlen=80))
    count: int = 0
    reference: np.ndarray | None = None
    started: float | None = None
    bottom: bool = False
    returned: float | None = None
    last_valid: float = -np.inf
    completed: float = -np.inf
    analysed: float = -np.inf
    gait: tuple = UNDECIDED

    def clear_motion(self):
        self.posture.clear()
        self.reference = None
        self.started = self.returned = None
        self.bottom = False

    def reset(self):
        self.clear_motion()
        self.samples.clear()
        self.analysed = self.last_valid = self.completed = -np.inf
        self.gait = UNDECIDED


def pose_reading(pose):
    joints = pose[CORE]
    if not np.isfinite(joints).all() or (joints[:, 2] < CONFIG['keypoint_conf']).any():
        return None
    shoulder, hip, knee, ankle = joints[:, :2].reshape(4, 2, 2)
    a, b = hip - knee, ankle - knee
    product = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    torso = hip.mean(axis=0) - shoulder.mean(axis=0)
    scale = np.linalg.norm(torso)
    if product.min() < 16 or scale < 12:
        return None
    flexion = np.degrees(np.arccos(np.clip(-(a * b).sum(axis=1) / product, -1., 1.)))
    if np.any(np.linalg.norm(hip - shoulder, axis=1) < 8.):
        return None
    lean = np.degrees(np.arctan2(abs(torso[0]), torso[1]))
    hip_y = hip[:, 1].mean()
    return np.r_[flexion, hip_y, scale, lean, ankle.ravel(), hip[:, 0].mean()]


def squat_step(track, now, reading):
    if reading is None:
        if now - track.last_valid > GAP_S:
            track.clear_motion()
        return False
    if now - track.last_valid > GAP_S:
        track.clear_motion()
    track.last_valid = now
    track.posture.append((now, reading))
    while now - track.posture[0][0] > .67:
        track.posture.popleft()
    short = [v for t, v in track.posture if now - t <= .15]
    current = np.median(short, axis=0)
    upright = current[:2].max() <= 28 and current[4] <= 32
    ref = track.reference
    if ref is not None:
        drift = np.linalg.norm((current[5:9] - ref[5:9]).reshape(2, 2), axis=1).max() / ref[3]
        drop = ((current[2] - current[[6, 8]].mean()) - (ref[2] - ref[[6, 8]].mean())) / ref[3]
        flex = current[:2] - ref[:2]
        knee_motion = float(np.max(flex))
        if drift > .24 or (track.started is not None and now - track.started > 10.):
            track.clear_motion()
            return False
        if track.started is None and drop >= .10 and (flex.mean() >= 15 or (drop >= .30 and knee_motion >= 16)):
            track.started = now
        if track.started is not None:
            enough_bend = flex.mean() >= 32 and flex.min() >= 10
            # A foreshortened leg can look straight in 2D; require more planted-hip excursion in that case.
            projected_bend = drift <= .17 and drop >= .48 and knee_motion >= 25
            track.bottom |= drop >= .25 and (enough_bend or projected_bend)
            if upright and drop <= .13:
                if track.returned is None:
                    track.returned = now
                if now - track.returned >= .075:
                    if track.bottom and now - track.started >= .40:
                        track.count += 1
                        track.completed = now
                    track.started = track.returned = None
                    track.bottom = False
                    track.reference = current
            else:
                track.returned = None
            return track.started is not None
    base = [(t, v) for t, v in track.posture if now - t <= .26]
    if (upright and len(base) >= 3 and now - base[0][0] >= .16
            and all(v[:2].max() <= 28 and v[4] <= 32 for _, v in base)):
        values = np.stack([v for _, v in base])
        if np.ptp(values[:, [2, 5, 6, 7, 8, 9]], axis=0).max() <= .12 * current[3]:
            track.reference = np.median(values, axis=0)
    return False


def quiet_pose(track, now):
    if len(track.posture) < 4 or now - track.posture[0][0] < .55:
        return UNDECIDED
    values = np.stack([v for _, v in track.posture])
    scale = np.median(values[:, 3])
    if np.ptp(values[:, [2, 5, 6, 7, 8, 9]], axis=0).max() > .13 * scale:
        return UNDECIDED
    if np.quantile(values[:, :2].max(axis=1), .85) <= 25 and np.quantile(values[:, 4], .85) <= 27:
        return ('站立', 'confirmed')
    if np.quantile(values[:, :2].min(axis=1), .15) >= 55 or np.quantile(values[:, 4], .15) >= 52:
        return ('其他', 'confirmed')
    return UNDECIDED


def summarize(rows, stairs_supported, available):
    labels = [row['activity'] for row in rows]
    for kind in ('stairs_up', 'stairs_down'):
        valid = sum(a and s and label == kind for label, s, a in zip(labels, stairs_supported, available))
        if valid >= max(2, .60 * len(labels)):
            return GAIT[kind], 'confirmed'
    if any(label.startswith('stairs_') for label in labels):
        return UNDECIDED
    for kind in ('walking', 'running'):
        votes = [row for row, valid in zip(rows, available) if valid
                 and (row['activity'] == kind or kind == 'walking' and row['activity'] == 'gait_walk')
                 and not any(row.get(flag) for flag in ('action_hypotheses', 'ambiguous_gait',
                                                        'locomotion_partial', 'gait_inferred'))]
        if len(votes) >= max(2, .60 * len(labels)):
            return GAIT[kind], 'confirmed'
    if sum(a and label == 'bending' for label, a in zip(labels, available)) >= max(2, .60 * len(labels)):
        return '其他', 'confirmed'
    return UNDECIDED


class LiveRules:
    def __init__(self, cv2, retention=3.):
        self.camera = Camera(cv2)
        self.tracks = {}
        self.index = 0
        self.retention = retention

    def reset(self, identities):
        for identity in identities:
            if identity in self.tracks:
                self.tracks[identity].reset()

    def frame(self, now, image, detections):
        self.index += 1
        height, width = image.shape[:2]
        self.camera.update(self.index, image, [np.asarray(box) * [width, height, width, height]
                                              for _, box, _ in detections])
        self.tracks = {i: track for i, track in self.tracks.items() if now - track.seen <= self.retention}
        output = []
        for identity, box, pose in detections:
            track = self.tracks.get(identity)
            if track is None:
                track = self.tracks[identity] = Track(now)
            if now - track.seen > GAP_S:
                track.reset()
            track.seen = now
            pixel = pose.copy()
            pixel[:, :2] *= max(width, height)
            reading = pose_reading(pixel)
            if reading is not None and now - track.last_valid > GAP_S:
                track.reset()
            track.samples.append((now, pixel, self.index))
            while now - track.samples[0][0] > WINDOW_S:
                track.samples.popleft()
            active = squat_step(track, now, reading)
            action, status = UNDECIDED
            if reading is not None:
                if now - track.completed <= .35 and not active:
                    action, status = '深蹲完成', 'confirmed'
                elif active:
                    action, status = '深蹲中', 'candidate'
                else:
                    action, status = quiet_pose(track, now)
            if action == UNDECIDED[0] and reading is not None:
                if now - track.analysed >= UPDATE_S:
                    track.analysed = now
                    track.gait = self._gait(track, now, width, height)
                action, status = track.gait
            else:
                track.gait = UNDECIDED
                track.analysed = -np.inf
            output.append(dict(id=identity, box=box, action=action, status=status, count=track.count))
        return output

    def _gait(self, track, now, width, height):
        if len(track.samples) < 16 or now - track.samples[0][0] < 1.6:
            return UNDECIDED
        times, poses, indices = zip(*track.samples)
        t, pose = np.array(times), np.stack(poses)
        camera = self.camera.slice(indices)
        if camera is None:
            return UNDECIDED
        f = features.pose_features(dict(time_s=t, keypoints=pose), CONFIG)
        f = features.camera_features(f, camera, CONFIG, width, height)
        rules.period_features(f, CONFIG)
        local, _ = calibrate(f, CONFIG)
        _, rows, _, _, _ = rules.classify_raw(f, local, arm_semantics=False)
        gait, _ = judge(f, camera, pose, rows, local)
        blocked = np.array([row['action_event_id'] > 0 for row in rows])
        calm = np.isfinite(f['camera_speed']) & (f['camera_speed'] <= .16)
        stairs, _, supported = detect_stairs(t, pose, f['raw_valid'], f['torso_px'], calm, blocked, height)
        recent = np.flatnonzero((t >= now - .95) & (t <= now - .35))
        votes = [dict(gait[i], activity=stairs[i]) if stairs[i] and not blocked[i] else gait[i] for i in recent]
        available = f['raw_valid'][recent][:, CORE].all(axis=1)
        return summarize(votes, supported[recent], available)
