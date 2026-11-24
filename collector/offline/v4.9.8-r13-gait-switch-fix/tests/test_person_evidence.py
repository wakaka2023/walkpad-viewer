import unittest

import numpy as np

from action_pipeline import load_config
from display_ids import audit_id_map, number_outputs
from person_evidence import filter_person_tracks


BOX = [100, 120, 280, 470]
EDGE_BOX = [90, 78, 400, 540]
PERSON = [[190, 142], [185, 139], [195, 139], [179, 145], [201, 145], [151, 181],
          [228, 181], [140, 239], [245, 248], [138, 295], [251, 301], [165, 282],
          [214, 281], [160, 355], [219, 355], [158, 446], [222, 446]]


def pose(points, indexes=None):
    result = np.zeros((17, 3))
    indexes = list(range(17)) if indexes is None else indexes
    result[indexes, :2] = points
    result[indexes, 2] = .92
    return result


def video(rows, fps=30):
    rows = sorted(rows, key=lambda row: row[0])
    n = max(r[0] for r in rows)+1
    t = np.arange(n+1)/fps
    frame, ids, boxes, keypoints = zip(*rows)
    return dict(frame_index=np.array(frame), track_id=np.array(ids), boxes=np.array(boxes),
                time_s=t[:-1], frame_end_s=t[1:], confidence=np.full(len(rows),.95),
                height=598, keypoints=np.asarray(keypoints))


def observations(identity, times, box, joints):
    return [(t, identity, box, joints) for t in times]


class PersonEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.empty = np.zeros((17,3))

    def test_persistent_high_confidence_boxes_without_pose_do_not_publish_id(self):
        data = video(observations(10, range(75), BOX, self.empty))
        out, audit = filter_person_tracks(data, self.cfg)
        render, mapping = number_outputs(out, {}, [])
        self.assertFalse(render['track_id'].size)
        self.assertEqual(mapping, {})
        self.assertEqual(audit[0]['removed_observations'],75)
        self.assertEqual(audit_id_map(data['track_id'], {}, mapping, set(), set(), non_person={10}),
                         [dict(yolo_track_id=10, track_id=None, internal_track_id=None,
                               status='rejected_person_evidence')])
        self.assertIs(data['time_s'], out['time_s'])
        self.assertEqual(len(data['track_id']),75)
        self.assertEqual(out['boxes'].shape,(0,4))
        self.assertEqual(out['keypoints'].shape,(0,17,3))

    def test_collapsed_joints_and_points_outside_box_are_rejected(self):
        phantom = pose([[180, 280]]*17)
        outside = pose(np.asarray(PERSON)+[300, 0])
        out, audit = filter_person_tracks(video(observations(1, range(40), BOX, phantom) +
                                               observations(2, range(40), BOX, outside)), self.cfg)
        self.assertEqual(out['frame_index'].size,0)
        self.assertEqual([a['confirmed_body_frames'] for a in audit],[0,0])

    def test_sporadic_pose_is_not_enough_for_a_persistent_box(self):
        data = video([(i, 1, BOX, pose(PERSON) if i in (1, 27, 55) else self.empty) for i in range(75)])
        out, audit = filter_person_tracks(data, self.cfg)
        self.assertFalse(out['track_id'].size)
        self.assertEqual(audit[0]['confirmed_body_frames'],3)

    def test_still_person_and_order_after_rejection(self):
        data = video(observations(1, range(55), BOX, self.empty) +
                     observations(8, range(8,55), BOX, pose(PERSON)))
        out, audit = filter_person_tracks(data, self.cfg)
        render, ids = number_outputs(out, {}, [dict(track_id=8, accepted=True)])
        self.assertEqual(ids,{8:1})
        self.assertEqual(np.unique(render['track_id']).tolist(),[1])
        self.assertEqual(audit[1]['removed_observations'],0)

    def test_cropped_legs_but_not_shoe_points_alone(self):
        legs = pose([[134,260], [325,257], [125,486], [350,495]], [13,14,15,16])
        shoes = pose([[130,486], [350,500]], [15,16])
        data = video(observations(4, range(45), EDGE_BOX, legs) +
                     observations(9, range(45), EDGE_BOX, shoes))
        out, audit = filter_person_tracks(data, self.cfg)
        self.assertEqual(np.unique(out['track_id']).tolist(),[4])
        self.assertEqual([a['edge_lower_body_frames'] for a in audit],[45,0])
        self.assertEqual([a['confirmed_body_frames'] for a in audit],[0,0])

    def test_weak_pose_and_brief_occlusion_remain_but_unsupported_tail_is_removed(self):
        torso = pose(np.asarray(PERSON)[[5,6,11]], [5,6,11])
        data = video([(i, 9, BOX, self.empty if 14 <= i < 20 or i >= 55 else torso if i >= 24 else pose(PERSON))
                      for i in range(90)])
        out, audit = filter_person_tracks(data, self.cfg)
        self.assertEqual(out['frame_index'].tolist(),list(range(55)))
        self.assertEqual(audit[0]['removed_observations'],35)

    def test_portion_of_real_track_can_start_with_a_brief_weak_pose(self):
        data = video([(i, 14, BOX, self.empty if i < 5 else pose(PERSON)) for i in range(40)])
        out, audit = filter_person_tracks(data, self.cfg)
        self.assertIs(data,out)
        self.assertEqual(audit[0]['status'],'person_supported')
