from collections import deque
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .rules.person_evidence import pose_evidence
from .vision_identity import BREAK_GAP, clear

POSE_CONF = .50
POSE_SPAN = .08
POSE_GAP = .19
MIN_SAMPLES = 5
SAMPLE_SPAN = .40
MAX_FEATURES = 96
MAX_TRACKLETS = 256
MAX_PEOPLE = 32
LIVE_SIM = .70
REVIEW_SIM = .52
MARGIN = .12


def overlaps(a, b):
    return a.start <= b.end and b.start <= a.end


@dataclass
class Evidence:
    seen: float
    positive: deque = field(default_factory=lambda: deque(maxlen=12))
    confirmed: bool = False


class PoseGate:
    def __init__(self):
        self.states = {}

    def frame(self, now, image, detections):
        self.states = {key: state for key, state in self.states.items() if now - state.seen <= BREAK_GAP}
        if not detections:
            return [], [], 0
        h, w = image.shape[:2]
        boxes = np.asarray([box for _, box, _ in detections]) * [w, h, w, h]
        poses = np.stack([pose for _, _, pose in detections]).copy()
        poses[:, :, :2] *= max(w, h)
        body, lower, weak = pose_evidence(boxes, poses, h, POSE_CONF)
        accepted, sampleable, pending = [], [], 0
        for row, item in enumerate(detections):
            state = self.states.setdefault(item[0], Evidence(now))
            state.seen = now
            if body[row] or lower[row]:
                if state.positive and now - state.positive[-1] > POSE_GAP + 1e-9:
                    state.positive.clear()
                state.positive.append(now)
                state.confirmed |= len(state.positive) >= 3 and now - state.positive[0] >= POSE_SPAN - 1e-9
            if state.confirmed and weak[row]:
                accepted.append(item)
                sampleable.append(bool(body[row]))
            else:
                pending += 1
        return accepted, sampleable, pending


@dataclass
class Tracklet:
    key: int
    start: float
    end: float
    person: int = 0
    first_sample: float = -1.
    last_sample: float = -1.
    last_attempt: float = -1.
    samples: list = field(default_factory=list)
    pick: int = 0
    votes: int = 0

    def observe(self, now, value):
        if value is None or now - self.last_sample < .095 or len(self.samples) == MAX_FEATURES:
            return
        vector = np.asarray(value, np.float32).reshape(-1)
        norm = np.linalg.norm(vector)
        if np.isfinite(norm) and norm > 0:
            if not self.samples:
                self.first_sample = now
            self.samples.append(vector / norm)
            self.last_sample = now

    def ready(self):
        return len(self.samples) >= MIN_SAMPLES and self.last_sample - self.first_sample >= SAMPLE_SPAN - 1e-9


def affinity(query, reference):
    similar = np.asarray(query, dtype=np.float32) @ np.asarray(reference, dtype=np.float32).T
    def directional(matrix):
        return float(np.median(np.sort(matrix, axis=1)[:, -3:].mean(axis=1)))
    return min(directional(similar), directional(similar.T))


class CollectionIdentity:
    """Tracks may join the roster only before the first enrolled person leaves."""
    ttl = float('inf')

    def __init__(self):
        self.pose = PoseGate()
        self.tracks = {}
        self.anchors = {}
        self.history = []
        self.seen = {}
        self.locked = False
        self.overflow = False

    def _rank(self, track):
        if not track.ready():
            return []
        return sorted(((affinity(track.samples, anchor.samples), pid) for pid, anchor in self.anchors.items()
                       if anchor.ready()), reverse=True)

    @staticmethod
    def _pick(scores, threshold):
        if not scores or scores[0][0] < threshold:
            return 0
        if len(scores) > 1 and scores[0][0] - scores[1][0] < MARGIN:
            return 0
        return scores[0][1]

    def resolve(self, now, image, detections, features):
        valid, sampleable, pending = self.pose.frame(now, image, detections)
        self.tracks = {key: tr for key, tr in self.tracks.items() if now - tr.end <= BREAK_GAP}
        present = {}
        boxes = [box for _, box, _ in detections]
        positions = {key: i for i, (key, _, _) in enumerate(detections)}
        h, w = image.shape[:2]
        for (key, box, _), good in zip(valid, sampleable):
            tr = self.tracks.get(key)
            if tr is None:
                if len(self.history) == MAX_TRACKLETS:
                    self.overflow = True
                    pending += 1
                    continue
                tr = self.tracks[key] = Tracklet(len(self.history) + 1, now, now)
                self.history.append(tr)
            tr.end = now
            present[key] = tr
            i = positions[key]
            if (good and (box[3] - box[1]) >= .30 and clear(h, w, box, boxes[:i] + boxes[i+1:])):
                tr.observe(now, features.get(key))
        active = {tr.person for tr in present.values() if tr.person}
        self.locked |= any(pid not in active and now - time > BREAK_GAP for pid, time in self.seen.items())
        if not self.locked and set(self.anchors) == active:
            for tr in present.values():
                if not tr.person and len(self.anchors) < MAX_PEOPLE:
                    tr.person = len(self.anchors) + 1
                    self.anchors[tr.person] = tr
                    active.add(tr.person)
        proposals = {}
        for tr in present.values():
            if tr.person or tr.last_sample <= tr.last_attempt:
                continue
            tr.last_attempt = tr.last_sample
            scores = self._rank(tr)
            pick = self._pick(scores, LIVE_SIM)
            if pick and (pick in active or any(old.person == pick and overlaps(tr, old) for old in self.history)):
                pick = 0
            tr.votes = tr.votes + 1 if pick and pick == tr.pick else (1 if pick else 0)
            tr.pick = pick
        # A competitor with less accumulated evidence still blocks an automatic link.
        for tr in present.values():
            if not tr.person and tr.pick:
                proposals.setdefault(tr.pick, []).append(tr)
        broken = set()
        for pid, matches in proposals.items():
            tr = matches[0]
            if (len(matches) == 1 and tr.votes >= 3 and pid not in active
                    and not any(old.person == pid and overlaps(tr, old) for old in self.history)):
                tr.person = pid
                broken.add(pid)
        accepted = []
        for key, box, pose in valid:
            tr = present.get(key)
            if tr is None:
                continue
            if not tr.person:
                pending += 1
                continue
            if now - self.seen.get(tr.person, now) > BREAK_GAP:
                broken.add(tr.person)
            self.seen[tr.person] = now
            accepted.append((tr.person, box, pose))
        return accepted, broken, pending

    def review(self):
        proposals = {}
        for tr in self.history:
            if tr.person:
                continue
            scores = self._rank(tr)
            pick = self._pick(scores, REVIEW_SIM)
            if pick and not any(old.person == pick and overlaps(tr, old) for old in self.history):
                proposals[tr.key] = (tr, pick, scores[0][0])
        collisions = {item.key for (first, a, _), (second, b, _) in combinations(proposals.values(), 2)
                      if a == b and overlaps(first, second) for item in (first, second)}
        return {
            'type': 'review', 'overflow': self.overflow,
            'tracks': [dict(track_id=tr.key, start_s=tr.start, last_seen_s=tr.end, person_id=tr.person or None,
                            suggested_id=proposals[tr.key][1] if tr.key in proposals and tr.key not in collisions else None,
                            status='confirmed_tracklet' if tr is self.anchors.get(tr.person) else
                                   'live_reconnected' if tr.person else 'needs_review')
                       for tr in self.history],
        }
