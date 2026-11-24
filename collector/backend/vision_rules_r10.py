"""窗口级实时适配：把 rules 包的分析跑在最近一段帧上。"""

import json
from collections import deque
from pathlib import Path

import numpy as np

from .rules import multi_action_analysis as analysis


CONFIG = json.loads((Path(__file__).resolve().parent / 'rules' / 'action_config.json')
                    .read_text(encoding='utf-8'))
WINDOW_S = 12.
INTERVAL_S = 1.
LAG_S = .7
HOLD_S = .9
MIN_SPAN_S = 5.
MIN_FRAMES = 24
MAX_GAP_S = .6
CAMERA_WIDTH = 480
COVERAGE_FLOOR = .5
SQUAT = ('squat', 'split_squat')
UNKNOWN = ('其他', 'unknown')
IDENTITY = np.array([[1., 0., 0.], [0., 1., 0.]])


class Camera:
    """逐帧估计镜头运动，供规则做位移补偿。"""

    def __init__(self, cv2):
        self.cv2 = cv2
        self.previous = None
        self.records = deque(maxlen=1200)

    def update(self, index, image, boxes):
        cv2 = self.cv2
        shrink = min(1., CAMERA_WIDTH / image.shape[1])
        small = (cv2.resize(image, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
                 if shrink < 1 else image)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mask = np.full(gray.shape, 255, np.uint8)
        for x0, y0, x1, y1 in boxes:
            pad = max(12., .1 * max(x1 - x0, y1 - y0)) * shrink
            lo = np.maximum(0, np.floor([x0 * shrink - pad, y0 * shrink - pad])).astype(int)
            hi = np.minimum([gray.shape[1] - 1, gray.shape[0] - 1],
                            np.ceil([x1 * shrink + pad, y1 * shrink + pad])).astype(int)
            cv2.rectangle(mask, tuple(lo), tuple(hi), 0, -1)
        matrix, valid = IDENTITY.copy(), False
        if self.previous is not None:
            matrix, valid = self._match(self.previous[0], self.previous[1], gray, mask, shrink)
        self.previous = (gray, mask)
        self.records.append((index, matrix, valid))

    def _match(self, before, before_mask, after, after_mask, shrink):
        cv2 = self.cv2
        corners = cv2.goodFeaturesToTrack(before, maxCorners=350, qualityLevel=.01,
                                         minDistance=8, mask=before_mask, blockSize=7)
        if corners is None or len(corners) < 16:
            return IDENTITY.copy(), False
        moved, status, _ = cv2.calcOpticalFlowPyrLK(before, after, corners, None,
                                                    winSize=(21, 21), maxLevel=3)
        if moved is None or status is None:
            return IDENTITY.copy(), False
        back, backward_status, _ = cv2.calcOpticalFlowPyrLK(after, before, moved, None,
                                                            winSize=(21, 21), maxLevel=3)
        if back is None or backward_status is None:
            return IDENTITY.copy(), False
        p, q = corners[:, 0], moved[:, 0]
        ok = (status[:, 0] > 0) & (backward_status[:, 0] > 0)
        ok &= np.linalg.norm(back[:, 0] - p, axis=1) < 1.5
        ok &= (q[:, 0] >= 0) & (q[:, 0] < after.shape[1]) & (q[:, 1] >= 0) & (q[:, 1] < after.shape[0])
        qint = np.clip(np.round(q).astype(int), [0, 0], [after.shape[1] - 1, after.shape[0] - 1])
        ok &= after_mask[qint[:, 1], qint[:, 0]] > 0
        if ok.sum() < 16:
            return IDENTITY.copy(), False
        fit, flags = cv2.estimateAffinePartial2D(p[ok], q[ok], method=cv2.RANSAC,
                                                 ransacReprojThreshold=2., maxIters=1500,
                                                 confidence=.99, refineIters=10)
        if fit is None or flags is None:
            return IDENTITY.copy(), False
        selected = flags[:, 0] > 0
        scale = float(np.linalg.norm(fit[:, 0]))
        if float(selected.mean()) < CONFIG['camera_min_inliers'] or not .8 <= scale <= 1.25 or selected.sum() < 12:
            return IDENTITY.copy(), False
        matrix = fit.copy()
        matrix[:, 2] /= shrink
        return matrix, True

    def slice(self, indices):
        position = {index: n for n, index in enumerate(indices)}
        picked = [(position[i], m, v) for i, m, v in self.records if i in position]
        if len(picked) < 2:
            return None
        source = np.array([n for n, _, _ in picked])
        matrices = np.array([m for _, m, _ in picked])
        valid = np.array([v for _, _, v in picked])
        if valid.mean() < COVERAGE_FLOOR:
            # 无纹理背景常出现在固定机位；此时按镜头静止补偿，只影响位移口径。
            matrices = np.tile(IDENTITY, (len(source), 1, 1))
            valid = np.ones(len(source), bool)
        return dict(source_indices=source, matrices=matrices, valid=valid)


class Track:
    def __init__(self):
        self.samples = deque(maxlen=600)
        self.counted = deque(maxlen=40)
        self.seen = -1e9
        self.count = 0
        self.label = UNKNOWN
        self.held = (-1e9, UNKNOWN)
        self.completed = -1e9
        self.analysed = -1e9
        self.failed = 0


class R10Actions:
    """按 R10 规则分析最近的窗口，并只回报滞后一帧的结论。"""

    def __init__(self, cv2, retention=3., window=WINDOW_S, interval=INTERVAL_S):
        self.camera = Camera(cv2)
        self.tracks = {}
        self.retention = retention
        self.window = window
        self.interval = interval
        self.size = (0, 0)
        self.index = 0

    def frame(self, now, image, detections):
        self.index += 1
        height, width = image.shape[:2]
        self.size = (width, height)
        self.camera.update(self.index, image, [np.asarray(box) * [width, height, width, height]
                                              for _, box, _ in detections])
        self.tracks = {i: p for i, p in self.tracks.items() if now - p.seen <= self.retention}
        output = []
        for identity, box, pose in detections:
            track = self.tracks.setdefault(identity, Track())
            track.seen = now
            pixel = pose.copy()
            pixel[:, :2] *= max(width, height)
            track.samples.append((now, pixel, self.index, np.asarray(box, float), float(pose[:, 2].mean())))
            while now - track.samples[0][0] > self.window:
                track.samples.popleft()
            if now - track.analysed >= self.interval:
                track.analysed = now
                self._analyze(track, now)
            text, status = self._describe(track, now)
            output.append(dict(id=identity, box=box, action=text, status=status, count=track.count))
        return output

    def reset(self, identities):
        for identity in identities:
            track = self.tracks.get(identity)
            if track is not None:
                # 计数与已计数周期保留，避免同一周期在新窗口里被重复计数。
                track.samples.clear()
                track.held = (-1e9, UNKNOWN)
                track.label = UNKNOWN
                track.analysed = -1e9

    def _analyze(self, track, now):
        samples = list(track.samples)
        if len(samples) < MIN_FRAMES or samples[-1][0] - samples[0][0] < MIN_SPAN_S:
            return
        times = np.array([s[0] for s in samples], float)
        pose = np.stack([s[1] for s in samples]).astype(np.float32)
        camera = self.camera.slice([s[2] for s in samples])
        if camera is None:
            return
        width, height = self.size
        boxes = np.array([s[3] for s in samples], float)
        confidence = np.array([s[4] for s in samples], float)
        try:
            self._run(track, now, times, pose, camera, width, height, boxes, confidence)
        except (ValueError, IndexError, KeyError, TypeError, np.linalg.LinAlgError):
            track.failed += 1
            if track.failed == 1:
                track.label = UNKNOWN

    def _run(self, track, now, times, pose, camera, width, height, boxes, confidence):
        count = len(times)
        step = float(np.median(np.diff(times)))
        data = dict(time_s=times, frame_end_s=times + step, frame_index=np.arange(count),
                    track_id=np.zeros(count, np.int64), keypoints=pose,
                    boxes=boxes, confidence=confidence, width=float(width), height=float(height))
        result, _ = analysis.analyze(data, camera, CONFIG, MIN_SPAN_S, MAX_GAP_S)
        self._count(track, result['action_events'], now)
        rows = {int(row['video_frame']): row for row in result['frame_labels']}
        if not rows:
            return
        frame = int(np.searchsorted(times, now - LAG_S, side='right'))
        track.label = self._describe_row(rows.get(max(1, min(frame, count))))
        if track.label[1] != 'unknown':
            track.held = (now, track.label)

    def _count(self, track, events, now):
        for event in events:
            if event.get('activity') != 'squat' or event['end_video_s'] > now - .15:
                continue
            start, stop = event['start_video_s'], event['end_video_s']
            if stop - start > CONFIG['cycle_max_s']:
                continue
            if any(start < old[1] - .2 and stop > old[0] + .2 for old in track.counted):
                continue
            track.counted.append((start, stop))
            track.count += 1
            track.completed = now

    @staticmethod
    def _describe_row(row):
        if str(row['activity']) in SQUAT:
            return '深蹲中', 'candidate'
        status = row.get('activity_status')
        if status not in ('supported', 'candidate', 'transition'):
            return UNKNOWN
        text = str(row.get('annotation') or '其他')
        return (text, 'confirmed') if status == 'supported' else (text, 'candidate')

    @staticmethod
    def _describe(track, now):
        if now - track.completed <= 1.2:
            return '深蹲完成', 'confirmed'
        text, status = track.label
        if status == 'unknown' and now - track.held[0] <= HOLD_S:
            # 动作之间的空档不立刻改回未判定，沿用上一结论但降为候选。
            return track.held[1][0], 'candidate'
        return text, status
