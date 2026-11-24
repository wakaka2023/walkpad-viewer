import numpy as np
import pytest

from backend.vision_session import CollectionIdentity, PoseGate, Tracklet, affinity
from backend.vision_worker import COLLECTION_ID_MODE
from backend.vision_workstation import PROTOCOLS, selection

IMAGE = np.zeros((540, 960, 3), dtype=np.uint8)
F1 = np.array([1., 0., 0.], np.float32)
F2 = np.array([0., 1., 0.], np.float32)


def detection(key, x, valid=True, lower=False):
    box = [x, .12, x + .20, .82]
    pose = np.zeros((17, 3), np.float32)
    points = {0: (.50, .10), 5: (.32, .23), 6: (.68, .23), 7: (.22, .40), 8: (.78, .40),
              11: (.38, .54), 12: (.62, .54), 13: (.34, .73), 14: (.66, .73), 15: (.33, .93), 16: (.67, .93)}
    if lower:
        points = {i: point for i, point in points.items() if i >= 13}
    if valid:
        for index, (dx, dy) in points.items():
            pose[index] = x + dx * .2, (.12 + dy * .70) * 540 / 960, .9
    return key, box, pose


def feed(identity, indices, *items):
    results = []
    for i in indices:
        rows = [item for item, _ in items]
        vectors = {item[0]: feature for item, feature in items}
        people, broken, pending = identity.resolve(i * .10, IMAGE, rows, vectors)
        results.append(([pid for pid, _, _ in people], broken, pending))
    return results


def test_r5_rejects_missing_collapsed_and_ankle_only_but_allows_cropped_legs():
    gate = PoseGate()
    bad = detection(1, .12, valid=False)
    collapsed = detection(2, .37, valid=False)
    collapsed[2][:] = [.46, .25, .95]
    ankles = detection(3, .62, lower=True)
    ankles[2][13:15] = 0.
    healthy = detection(4, .12)
    legs = detection(5, .40, lower=True)
    for i in range(8):
        output, sampleable, pending = gate.frame(i * .0667, IMAGE, [bad, collapsed, ankles, healthy, legs])
        if i < 2:
            assert not output
        else:
            assert [key for key, _, _ in output] == [4, 5]
            assert sampleable == [True, False] and pending == 3


def test_reentry_reuses_id_only_after_multiple_new_samples_and_breaks_action_window():
    identity = CollectionIdentity()
    p1, p2, p2_return = detection(11, .12), detection(12, .55), detection(66, .55)
    assert feed(identity, range(16), (p1, F1), (p2, F2))[-1][0] == [1, 2]
    feed(identity, range(16, 26), (p1, F1))
    resumed = feed(identity, range(26, 38), (p1, F1), (p2_return, F2))
    assert resumed[5][0] == [1]
    assert [r for r in resumed if r[1]] == [([1, 2], {2}, 0)]
    assert identity.review()['tracks'][-1]['status'] == 'live_reconnected'


def test_reentry_uncertain_and_newcomer_not_allocated_after_departure():
    identity = CollectionIdentity()
    p1, p2, p3 = detection(11, .12), detection(12, .55), detection(13, .55)
    feed(identity, range(15), (p1, F1), (p2, F2))
    feed(identity, range(15, 25), (p1, F1))
    medium = np.array([0., .61, np.sqrt(1 - .61**2)], np.float32)
    r = feed(identity, range(25, 41), (p1, F1), (p3, medium))
    assert all(people == [1] for people, _, _ in r)
    feed(identity, range(41, 50), (p1, F1))
    assert feed(identity, range(50, 65), (p1, F1), (detection(77, .55), F1))[-1][0] == [1]
    review = identity.review()
    assert review['tracks'][-2]['person_id'] is None and review['tracks'][-2]['suggested_id'] == 2
    assert review['tracks'][-1]['person_id'] is None and review['tracks'][-1]['suggested_id'] is None


def test_offline_competitors_and_ties_never_get_arbitrary_links():
    ident = CollectionIdentity()
    def track(key, start, end, vector, person=0):
        return Tracklet(key, start, end, person, start, end, samples=[vector]*5)
    ident.anchors[1] = track(1, 0., 2., F1, 1)
    a = track(2, 3., 4., F1)
    b = track(3, 3.5, 5., F1)
    ident.history = [ident.anchors[1], a, b]
    assert [r['suggested_id'] for r in ident.review()['tracks']] == [None, None, None]
    ident.anchors[2] = track(4, 0., 2., F1, 2)
    ident.history = [ident.anchors[1], ident.anchors[2], a]
    assert ident.review()['tracks'][-1]['suggested_id'] is None


def test_affinity_is_symmetric_and_protocol_is_explicit():
    src = [F1] * 5
    query = [F1, F1, F2, F2, F2]
    assert affinity(src, query) == pytest.approx(affinity(query, src))
    assert selection(PROTOCOLS[COLLECTION_ID_MODE, 'r13']) == (COLLECTION_ID_MODE, 'r13')
    assert (COLLECTION_ID_MODE, 'r4') not in PROTOCOLS
