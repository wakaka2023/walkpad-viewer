import unittest

import cv2
import numpy as np

from display_ids import audit_id_map
from flat_depiction import _evidence


class FlatDepictionTests(unittest.TestCase):
    def fixture(self, sheet=True, move=False, camera=True, patterned_wall=True):
        rng = np.random.default_rng(794)
        box = np.array([105.,75.,177.,188.])
        wall = np.full((263,285),73,np.uint8)
        if patterned_wall:
            for x,y in rng.integers([10,10],[275,253],(75,2)):
                cv2.rectangle(wall,(x-2,y-2),(x+3,y+2),int(rng.integers(30,136)), -1)
        if sheet:
            polygon = np.array([[100,70],[185,76],[181,203],[99,197]],np.int32)
            cv2.fillConvexPoly(wall,polygon,210)
        result = []
        offsets = [-3,-2,-1,0,0,1,2,3,4] if camera else [0]*9
        for frame,dx in enumerate(offsets):
            img = wall.copy()
            cv2.circle(img,(139,96),14,110,-1)
            cv2.rectangle(img,(124,113),(154,154),127,-1)
            cv2.line(img,(130,150),(126,185),99,8)
            cv2.line(img,(149,150),(152,185),99,8)
            bend=11 if move and frame >= 4 else 0
            cv2.line(img,(128,115),(109+bend,153-bend),35,8)
            cv2.line(img,(150,116),(168-bend,150-bend),35,8)
            shift = np.float32([[1,0,dx],[0,1,dx//2]])
            result.append(cv2.warpAffine(img,shift,img.shape[::-1],borderMode=cv2.BORDER_REPLICATE))
        return result,box

    def test_stationary_framed_depiction_remains_registered_during_camera_shift(self):
        self.assertTrue(_evidence(*self.fixture())[-1])
        self.assertTrue(_evidence(*self.fixture(camera=False))[-1])

    def test_live_standing_subject_not_rejected_on_plain_background(self):
        self.assertFalse(_evidence(*self.fixture(sheet=False))[-1])

    def test_articulation_inside_sheet_outline_vetoes_rejection(self):
        self.assertFalse(_evidence(*self.fixture(move=True))[-1])

    def test_insufficient_local_image_registration_does_not_remove_subject(self):
        self.assertFalse(_evidence(*self.fixture(patterned_wall=False))[-1])

    def test_rejected_depiction_has_no_public_id(self):
        self.assertEqual(audit_id_map(np.array([7,12]),{12:12},{12:1},set(),set(),flat_depictions={7}),
                         [dict(yolo_track_id=7,track_id=None,internal_track_id=None,status='rejected_flat_depiction'),
                          dict(yolo_track_id=12,track_id=1,internal_track_id=12,status='accepted')])
