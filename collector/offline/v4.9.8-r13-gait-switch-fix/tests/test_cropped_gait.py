from pathlib import Path
import unittest

import numpy as np

from cropped_gait import classify, deduplicate


class CroppedGaitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(Path(__file__).parent/'fixtures/cropped_gait_flow_regression.npz') as data:
            cls.t=data['time']
            cls.signals=data['signals']

    def test_close_walking_is_supported_without_a_torso_or_person_speed_threshold(self):
        labels,direct,_,_=classify(self.t,self.signals)
        walk=self.t[labels=='gait_walk']
        self.assertGreater(len(walk),315)
        self.assertLess(abs(walk[0]-.4333),.15)
        self.assertLess(abs(walk[-1]+1/30-11.2333),.15)
        self.assertGreater(np.count_nonzero(direct & (labels=='gait_walk')),2)
        self.assertEqual(labels[0],labels[-1]); self.assertEqual(labels[-1],'standing')
        rescaled=self.signals.copy(); rescaled[:,:7]*=4.25
        np.testing.assert_array_equal(labels,classify(self.t,rescaled)[0])

    def test_prolonged_camera_motion_alone_does_not_establish_a_walk(self):
        x=self.signals.copy(); x[:,3:7]=x[:,2:3]*1.05
        labels=classify(self.t,x)[0]
        self.assertFalse(np.isin(labels,('walking','gait_walk')).any())

    def test_no_formal_walk_when_moving_feet_have_no_ground_support(self):
        x=self.signals.copy(); x[:,5:7]=x[:,2:3]*9.
        labels=classify(self.t,x)[0]
        self.assertFalse((labels=='gait_walk').any())
        self.assertTrue((labels=='walking').any())

    def test_shared_box_with_distinct_visible_legs_keeps_both_people(self):
        f=np.r_[np.arange(48),np.arange(4,19)]
        ident=np.r_[np.ones(48,int),np.full(15,2)]
        boxes=np.array([[10,0,310,350]]*48+[[78,0,378,350]]*15,dtype=float)
        k=np.zeros((len(f),17,3))
        k[:,13:17,:]=[100,240,.9]
        k[48:,13:17,0]=260
        data=dict(height=480,frame_index=f,track_id=ident,boxes=boxes,keypoints=k,confidence=np.ones(len(f)))
        _,modes,removed=deduplicate(data,.35)
        self.assertEqual(modes,{1,2})
        self.assertEqual(removed,[])
        k[48:,13:17,0]=102
        _,_,removed=deduplicate(data,.35)
        self.assertEqual(removed,[2])


if __name__ == '__main__':
    unittest.main()
