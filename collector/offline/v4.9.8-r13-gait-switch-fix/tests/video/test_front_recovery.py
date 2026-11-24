import unittest
from unittest.mock import patch
import numpy as np
from front_recovery_rules import cap_front_recoveries
from test_terminal_recovery import fixture

class FrontRecoveryTests(unittest.TestCase):
    def test_successful_middle_event_excludes_standing_dwell(self):
        f,core,before,old_end=fixture()
        e=dict(core,rep_id=2,body_boundary_status='body_rule_candidate',hip_core_end_video_frame=core['end_video_frame'],
               start_video_s=1.0,hip_core_end_video_s=f['t'][core['end_video_frame']],end_video_frame=old_end)
        next_event=dict(rep_id=3,bottom_video_frame=349,body_boundary_status='front_adaptive_body_candidate')
        rows=[dict(rep_id=2,squat_label=1) for _ in f['t']]
        with patch('front_recovery_rules.neutral_hold',return_value=before):
            events,frames=cap_front_recoveries(f,[e,next_event],rows)
        end=events[0]['end_video_frame']
        self.assertLess(end,old_end)
        self.assertGreater(f['t'][end],5.15)
        self.assertTrue(all(r['squat_label']==0 and r['rep_id']==0 for r in frames[end:old_end]))
        self.assertGreater(events[0]['recovery_confirmation_end_video_frame'],end)
        self.assertEqual(next_event,events[1])

    def test_no_upright_hold_retains_middle_recovery(self):
        f,core,before,old_end=fixture();f['body_speed'][:]=.5
        e=dict(core,rep_id=1,body_boundary_status='body_rule_candidate',hip_core_end_video_frame=core['end_video_frame'],end_video_frame=old_end)
        rows=[dict(rep_id=1,squat_label=1) for _ in f['t']]
        with patch('front_recovery_rules.neutral_hold',return_value=before):
            events,frames=cap_front_recoveries(f,[e],rows)
        self.assertEqual(events[0]['end_video_frame'],old_end)
        self.assertTrue(all(r['squat_label']==1 for r in frames))

if __name__=='__main__':unittest.main()
