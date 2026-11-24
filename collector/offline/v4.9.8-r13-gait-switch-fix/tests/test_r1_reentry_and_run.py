import unittest
import numpy as np
from appearance_reid import join_returns
from running_rebound import detect
from scene_ignore import filter_tracks, parse_roi


def record(spec):
    rows=sorted((f,u,b) for u,a,z,b in spec for f in range(a,z))
    f=np.array([x[0] for x in rows]); ids=np.array([x[1] for x in rows]); boxes=np.asarray([x[2] for x in rows],float)
    t=np.arange(int(f.max()+1))*.05
    return dict(frame_index=f,track_id=ids,boxes=boxes,confidence=np.ones(len(f)),keypoints=np.zeros((len(f),17,3)),
                width=1000,height=1000,time_s=t,frame_end_s=t+.05)


def fit(lower):
    return np.tile([[.12,.1,0.,0.,0.],list(lower)],(8,1,1))


class ReturnTests(unittest.TestCase):
    def test_distinct_clothing_persists_across_nonoverlapping_appearances(self):
        b=(12,20,100,220)
        data=record([(2,0,20,b),(3,12,70,b),(6,90,120,b),(8,150,190,b),(11,200,240,b)])
        blue=fit((.5,.36,.75,.06,.06)); white=fit((.82,.04,0.,.92,0.))
        samples={2:blue,3:white,6:blue-.002,8:blue+.002,11:white+.002}
        ids,links,_=join_returns(data,data['track_id'],samples,60)
        self.assertEqual(set(zip(data['track_id'],ids)),{(2,2),(3,3),(6,2),(8,2),(11,3)})
        self.assertEqual(len(links),3)

    def test_covisible_identical_clothing_blocks_ambiguous_reentry(self):
        b=(12,20,100,220)
        data=record([(1,0,30,b),(2,10,65,b),(3,75,110,b)])
        blue=fit((.5,.36,.73,.06,.06))
        ids,links,_=join_returns(data,data['track_id'],{1:blue,2:blue,3:blue},60)
        self.assertFalse(links)
        np.testing.assert_array_equal(ids,data['track_id'])

    def test_short_gap_children_do_not_poison_clear_prototype(self):
        b=(12,20,100,220)
        data=record([(1,0,80,b),(2,90,99,b),(4,150,220,b)])
        blue=fit((.5,.36,.73,.06,.06)); white=fit((.82,.04,0.,.92,0.))
        ids,_,_=join_returns(data,np.where(data['track_id']==2,1,data['track_id']),{1:blue,2:white,4:blue},60)
        self.assertEqual(set(ids),{1})

    def test_configured_glass_region_removes_only_static_contained_track(self):
        b=(110,205,180,420)
        data=record([(1,0,20,b),(2,0,20,b),(3,0,20,(100,80,400,500))])
        k=data['track_id']==2
        data['boxes'][k] += np.linspace(0,55,np.count_nonzero(k))[:,None]*np.array([1,0,1,0])
        cleaned,removed=filter_tracks(data,[parse_roi('0.10,0.18,0.26,0.46')])
        self.assertEqual([r['yolo_track_id'] for r in removed],[1])
        self.assertEqual(set(cleaned['track_id']),{2,3})
        np.testing.assert_array_equal(cleaned['time_s'],data['time_s'])


def pose_series(rebound=.09, alternates=True, advance=True, freq=2.8):
    t=np.arange(0,4,1/30.)
    poses=np.zeros((len(t),17,3)); poses[:,:,2]=1.
    moves=70*t if advance else 0*t
    poses[:,:,0]=160+moves[:,None]
    s=np.cos(np.pi*freq*t); y=100*rebound*np.cos(2*np.pi*freq*t)
    poses[:,[5,6],1]=100+y[:,None]; poses[:,[11,12],1]=200+y[:,None]
    poses[:,15,1]=310+20*s; poses[:,16,1]=310+20*s*(-1 if alternates else 1)
    poses[:,[7,8],1]=140; poses[:,[9,10],1]=180
    return t,poses,poses[:,:,2]>.5,np.ones(len(t),bool),np.full(len(t),100.)


class RunTests(unittest.TestCase):
    def test_short_running_recognized_from_coincident_rebounds_and_alternations(self):
        t,pose,v,can,s=pose_series()
        label,_,hz,_=detect(t,pose,v,can,s)
        self.assertGreater(label.sum()/30,2.)
        self.assertAlmostEqual(np.nanmedian(hz),2.8,delta=.27)

    def test_fast_stride_or_bouncing_alone_does_not_establish_running(self):
        for args in (dict(rebound=.02),dict(alternates=False),dict(advance=False),dict(freq=1.7)):
            self.assertFalse(np.any(detect(*pose_series(**args))[0]),args)

    def test_missing_pose_is_not_relabelled_as_running(self):
        t,pose,v,can,s=pose_series()
        can[(t>=1.35)&(t<=2.05)]=False
        label,_,_,_=detect(t,pose,v,can,s)
        self.assertFalse(label[~can].any())


if __name__=='__main__':
    unittest.main()
