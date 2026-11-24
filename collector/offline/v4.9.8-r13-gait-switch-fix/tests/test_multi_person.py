import unittest
from unittest.mock import patch

import numpy as np

from action_pipeline import load_config
from multi_action_analysis import analyze, fragments
from test_action_boundaries import pose_data
from track_reconnect import appearance_distance, choose_links, reconnect


def two_people():
    data = pose_data()
    t = data['time_s']
    standing = data['keypoints']
    moving = standing.copy()
    depth = np.exp(-((t-4.)/.82)**2)
    moving[:, [5, 6, 7, 8, 9, 10, 11, 12], 1] += 75*depth[:, None]
    moving[:, [13, 14], 0] += 64*depth[:, None]
    moving[:, :, 0] += 150
    n = len(t)
    data.update(frame_end_s=np.r_[t[1:], t[-1]+.05], frame_index=np.repeat(np.arange(n), 2),
                track_id=np.tile([4,9],n), keypoints=np.stack([standing,moving],axis=1).reshape(-1,17,3),
                confidence=np.full(2*n,.98), boxes=np.tile([[140,40,260,345],[290,40,430,345]],(n,1)))
    camera = dict(source_indices=np.arange(n), matrices=np.tile([[1.,0,0],[0,1.,0]],(n,1,1)),
                  valid=np.ones(n,bool))
    return data,camera


class MultiPersonTests(unittest.TestCase):
    def test_standing_and_squat_on_same_original_frames(self):
        data,cam = two_people()
        tables, summaries = analyze(data, cam, load_config(), .6, .35)
        events = tables['action_events']
        self.assertEqual({r['track_id'] for r in events}, {9})
        self.assertEqual(len(events),1)
        frame = events[0]['peak_video_frame']
        observed = {r['track_id']:r['activity'] for r in tables['frame_labels'] if r['video_frame']==frame}
        self.assertEqual(observed,{4:'standing',9:'squat'})
        self.assertTrue(all(r['accepted'] for r in summaries))

    def test_companion_trajectory_cannot_change_a_persons_labels_or_boundaries(self):
        data, cam = two_people()
        config = load_config()
        both, _ = analyze(data, cam, config, .6, .35)
        for track in (4, 9):
            own = data['track_id'] == track
            single = dict(data)
            for key in ('frame_index', 'track_id', 'keypoints', 'confidence', 'boxes'):
                single[key] = data[key][own]
            alone, _ = analyze(single, cam, config, .6, .35)
            for table, columns in (
                ('frame_labels', ('video_frame', 'activity', 'annotation', 'phase', 'action_event_id')),
                ('action_events', ('event_id', 'start_video_frame', 'peak_video_frame', 'end_video_frame')),
            ):
                def signature(rows):
                    return [tuple(r[k] for k in columns) for r in rows if r['track_id'] == track]
                self.assertEqual(signature(both[table]), signature(alone[table]))

    def test_absent_id_has_no_action_phase_during_gap(self):
        data,cam = two_people()
        missing = (data['track_id']==9)&(data['frame_index']>=60)&(data['frame_index']<100)
        for k in ('frame_index','track_id','keypoints','confidence','boxes'):
            data[k]=data[k][~missing]
        tables,summaries = analyze(data,cam,load_config(),.6,.35)
        absent = [r for r in tables['frame_labels'] if r['track_state']=='not_tracked']
        self.assertEqual(len(absent),40)
        self.assertTrue(all(r['track_id']==9 and r['activity']=='unknown' and not r['phase'] for r in absent))
        self.assertEqual(summaries[1]['fragment_count'],2)
        self.assertFalse(any(r['start_video_frame']<60 <100 < r['end_video_frame'] for r in tables['action_events']))

    def test_new_raw_id_is_a_new_analysis_fragment_even_after_reconnection(self):
        t = np.arange(8)*.1
        parts = fragments(np.arange(8),t,t+.1,.35,np.array([2]*4+[5]*4))
        self.assertEqual([p.tolist() for p in parts],[[0,1,2,3],[4,5,6,7]])

    def test_mutual_link_refuses_ambiguous_similar_tracks(self):
        a={'from_track_id':1,'to_track_id':4,'cost':.28}
        b={'from_track_id':2,'to_track_id':4,'cost':.31}
        self.assertEqual(choose_links([a,b]),[])
        self.assertEqual(choose_links([a,dict(b,cost=.43)]),[a])

    def test_reconnect_will_not_merge_simultaneous_tracks(self):
        data,cam=two_people()
        with patch('track_reconnect.read_appearances') as read:
            ids,links,_=reconnect(None,data,cam,.6,2.)
        read.assert_not_called()
        self.assertEqual(links,[])
        np.testing.assert_equal(ids,data['track_id'])

    def test_appearance_distance_distinguishes_distinct_color_distributions(self):
        a=np.tile([[1.,0],[1.,0]],(3,1,1)); b=a[:,:,::-1]
        self.assertAlmostEqual(appearance_distance(a,a),0.)
        self.assertAlmostEqual(appearance_distance(a,b),1.)


if __name__ == '__main__':
    unittest.main()

