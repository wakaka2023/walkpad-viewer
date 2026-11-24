import unittest

import numpy as np

from display_ids import audit_id_map, number_outputs


def sample(groups, frames, x):
    x = np.asarray(x)
    boxes = np.column_stack([x, np.zeros(len(x)), x+20, np.full(len(x),100)])
    return dict(track_id=np.asarray(groups), frame_index=np.asarray(frames), boxes=boxes)


class DisplayIDsTests(unittest.TestCase):
    def test_first_valid_appearance_controls_order_even_if_internal_ids_do_not(self):
        data = sample([1,17,3,17,24], [0,1,3,8,11], [40,80,15,100,32])
        tables = {name: [dict(track_id=3, activity='walking', video_frame=4),
                         dict(track_id=17, activity='running', video_frame=9)] for name in ('frames','activities','events')}
        summaries = [dict(track_id=j, accepted=j in (3,17)) for j in (1,3,17,24)]
        render, public = number_outputs(data, tables, summaries)
        self.assertEqual(public,{17:1,3:2})
        self.assertEqual(render['track_id'].tolist(),[0,1,2,1,0])
        self.assertEqual(data['track_id'].tolist(),[1,17,3,17,24])
        for rows in tables.values():
            self.assertEqual([(r['track_id'],r['activity'],r['video_frame']) for r in rows],[(1,'running',9),(2,'walking',4)])
        self.assertEqual([(r['track_id'],r['internal_track_id']) for r in summaries],[(1,17),(2,3),(None,1),(None,24)])

    def test_simultaneous_entry_is_ordered_from_left_to_right(self):
        d = sample([5,6,9,8], [0,0,0,5], [23,15,63,0])
        _, ids = number_outputs(d,{},[dict(track_id=i,accepted=True) for i in (5,6,9,8)])
        self.assertEqual(ids,{6:1,5:2,9:3,8:4})

    def test_reentry_raw_fragments_share_display_id_and_invalid_rows_are_blank(self):
        rows = audit_id_map([1,2,3,5,9,10], {2:2,3:3,5:2,9:9,10:10}, {2:1,3:2}, {1}, {9})
        self.assertEqual([(r['yolo_track_id'],r['track_id'],r['status']) for r in rows],
                         [(1,None,'ignored_static_roi'), (2,1,'accepted'),(3,2,'accepted'),(5,1,'accepted'),
                          (9,None,'cropped_duplicate'),(10,None,'rejected_short_or_insufficient_track')])

    def test_empty_or_all_rejected_video_does_not_allocate_an_id(self):
        for data,summary in [(sample([],[],[]),[]),(sample([9],[2],[42]),[dict(track_id=9,accepted=False)])]:
            out,mapping = number_outputs(data,{},summary)
            self.assertEqual(mapping,{})
            self.assertFalse(out['track_id'].any())
