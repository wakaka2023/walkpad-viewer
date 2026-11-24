import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import multiview_fusion as f
import multiview_pipeline as p
from test_individual_quality import TemporaryDirectory

def grade(view, **levels):
    keys = set(f.KEYS) - (f.SIDE_ONLY if view == 'front' else f.FRONT_ONLY)
    cs = {key: dict(status=('PASS', 'WARN', 'FAIL')[levels.get(key, 0)], level=levels.get(key, 0), value=0.1, unit='test units', warn=0.2, severe=0.4) for key in keys}
    return dict(video_view_hint=view + '_or_oblique', video_view_stable=1, video_pose_coverage=1.0, side_leg_pose_coverage=1.0, bilateral_pose_coverage=1.0, video_largest_gap_s=0.033, observable_criteria_json=json.dumps(cs))

def event(rid, start, end, view, **levels):
    return dict(rep_id=rid, start_video_s=start, end_video_s=end, bottom_video_s=(start + end) / 2, _grade=grade(view, **levels))

class FusionTests(unittest.TestCase):

    def test_both_views_have_eight_unique_observed_items(self):
        result = f.combine(grade('front'), grade('side'))
        self.assertEqual((result['grade'], result['observed_items']), ('GOOD', 8))

    def test_single_visible_view_does_not_get_missing_observation_penalty(self):
        result = f.combine(grade('front'), None)
        self.assertEqual((result['grade'], result['observed_items']), ('GOOD', 6))
        self.assertEqual(result['criteria']['depth']['status'], 'UNAVAILABLE')

    def test_shared_criterion_is_counted_once(self):
        result = f.combine(grade('front', foot_motion=1), grade('side', foot_motion=1))
        self.assertEqual(result['grade'], 'MID')
        self.assertEqual(result['observed_items'], 8)
        self.assertEqual(len(result['criteria']['foot_motion']['evidence']), 2)

    def test_supported_major_issue_is_not_averaged_away(self):
        self.assertEqual(f.combine(grade('front'), grade('side', depth=2))['grade'], 'POOR')

    def test_pass_in_one_view_cannot_cancel_supported_warning_in_the_other(self):
        self.assertEqual(f.combine(grade('front', completion=1), grade('side'))['grade'], 'MID')

    def test_many_minor_issues_do_not_accumulate_to_poor(self):
        self.assertEqual(f.combine(grade('front', completion=1, lateral_control=1), grade('side', depth=1, hip_knee_coordination=1))['grade'], 'MID')

    def test_indirect_cue_alone_is_never_poor(self):
        self.assertEqual(f.combine(grade('front', indirect_control=2), grade('side', indirect_control=2))['grade'], 'MID')

    def test_unstable_view_cannot_supply_a_major_penalty(self):
        bad = grade('side', depth=2)
        bad['video_view_stable'] = 0
        result = f.combine(grade('front'), bad)
        self.assertEqual((result['grade'], result['observed_items']), ('GOOD', 6))

    def test_view_specific_evidence_is_not_taken_from_wrong_camera(self):
        front = grade('front')
        cs = json.loads(front['observable_criteria_json'])
        cs['depth'] = dict(level=2, status='FAIL')
        front['observable_criteria_json'] = json.dumps(cs)
        self.assertEqual(f.combine(front, grade('side'))['grade'], 'GOOD')

    def test_zero_observed_items_is_never_good(self):
        self.assertIsNone(f.combine(None, None)['grade_code'])
        e = event(1, 0, 2, 'front')
        e['_grade']['video_view_stable'] = 0
        (rows, diagnostics) = f.fuse_events([e], [])
        self.assertEqual(rows, [])
        self.assertEqual(diagnostics[0]['kind'], 'zero_observed_items')

    def test_missing_repetition_does_not_shift_later_matches(self):
        front = [event(i + 1, 3 * i, 3 * i + 2, 'front') for i in range(3)]
        side = [event(1, 0, 2, 'side'), event(2, 6, 8, 'side')]
        (rows, _) = f.fuse_events(front, side)
        self.assertEqual([r['view_match'] for r in rows], ['both', 'front_only', 'both'])
        self.assertEqual(rows[-1]['side_rep_id'], 2)

    def test_separate_actions_are_not_forced_into_a_pair(self):
        (rows, _) = f.fuse_events([event(1, 0, 2, 'front')], [event(1, 5, 7, 'side')])
        self.assertEqual(len(rows), 2)
        self.assertTrue(all((r['view_match'] != 'both' for r in rows)))

    def test_coverage_selects_boundary_and_original_evidence_is_retained(self):
        (front, side) = (event(1, 0, 2, 'front'), event(1, 0.1, 2.1, 'side'))
        side['_grade']['side_leg_pose_coverage'] = 0.95
        (rows, _) = f.fuse_events([front], [side])
        self.assertEqual(rows[0]['boundary_source'], 'front')
        self.assertEqual(rows[0]['side_start_video_s'], 0.1)

    def test_side_boundary_wins_a_quality_tie(self):
        (rows, _) = f.fuse_events([event(1, 0, 2, 'front')], [event(1, 0.1, 2.1, 'side')])
        self.assertEqual(rows[0]['boundary_source'], 'side')

    def test_fusion_does_not_mutate_source_detections(self):
        (front, side) = ([event(1, 0, 2, 'front')], [event(1, 0, 2, 'side')])
        saved = copy.deepcopy((front, side))
        f.fuse_events(front, side)
        self.assertEqual((front, side), saved)

    def test_adjacent_rows_have_one_continuous_rep_id_and_phase(self):
        (events, _) = f.fuse_events([event(1, 0.5, 2.5, 'front')], [event(1, 0.5, 2.5, 'side')])
        times = np.arange(0, 3.0, 0.01)
        rows = list(p.frame_records(times, times + 0.01, events))
        inside = [r for r in rows if r['squat_label']]
        self.assertEqual(len(inside), 200)
        self.assertTrue(all((r['squat_rep_id'] == 1 and r['overall_grade'] == 'GOOD' for r in inside)))
        self.assertEqual(set((r['squat_phase'] for r in inside)), {'DOWN', 'UP'})
        self.assertEqual(rows[250]['squat_label'], 0)

    def test_overlap_is_partitioned_without_losing_bottoms(self):
        (rows, _) = f.fuse_events([event(1, 0, 2.2, 'front'), event(2, 2, 4, 'front')], [])
        self.assertEqual(rows[0]['end_video_s'], rows[1]['start_video_s'])
        self.assertTrue(all((e['start_video_s'] <= e['bottom_video_s'] < e['end_video_s'] for e in rows)))

    def test_config_rejects_turning_off_quality_gates(self):
        with TemporaryDirectory() as d:
            path = Path(d) / 'bad.json'
            path.write_text(json.dumps({'require_stable_view_for_fusion': False}))
            with self.assertRaises(ValueError):
                f.load_config(path)

    def test_both_panels_receive_text_and_skeleton(self):
        (rows, _) = f.fuse_events([event(1, 0, 2, 'front')], [event(1, 0, 2, 'side', depth=1)])
        image = np.zeros((360, 960, 3), np.uint8)
        keypoints = np.zeros((1, 17, 3))
        keypoints[:, :, :2] = 200
        keypoints[:, :, 2] = 1
        data = {view: {'keypoints': keypoints} for view in ('front', 'side')}
        import cv2
        with patch('cv2.putText', wraps=cv2.putText) as text:
            p.draw_frame(image, 1.0, 0, 1, rows, [0], data, {'front': (0, 0, 480, 360), 'side': (480, 0, 960, 360)})
        titles = [call.args[1] for call in text.call_args_list if call.args[1].startswith('OVERALL:')]
        self.assertEqual(len(titles), 2)
        self.assertEqual(titles[0], titles[1])
        self.assertIn('MID', titles[0])
        self.assertGreater(image[10:140, 10:470].sum(), 0)
        self.assertGreater(image[10:140, 490:950].sum(), 0)

    def test_ambiguous_overlap_cannot_silently_discard_a_major_fault(self):
        with self.assertRaisesRegex(ValueError, 'Ambiguous overlapping'):
            f.fuse_events([event(1, 0, 4, 'front')], [event(1, 0.5, 4.5, 'side', depth=2)])

    def test_gross_time_mismatch_is_rejected(self):
        front = [event(i, i * 3, i * 3 + 2, 'front') for i in range(3)]
        side = [event(i, i * 3 + 1, i * 3 + 3, 'side') for i in range(3)]
        with self.assertRaisesRegex(ValueError, 'synchron'):
            f.fuse_events(front, side)
    def test_cut_during_ascent_is_not_scored_as_failed_completion(self):
        t = np.arange(0, 2, 1 / 30)
        k = np.zeros((len(t), 17, 3)); k[:, :, 2] = 1
        for ids in ([5, 11, 13, 15], [6, 12, 14, 16]):
            for i, point in enumerate(ids):
                k[:, point, 1] = 50 + 50 * i
            k[:, ids[0], 1] -= 50 * t
            k[:, ids[1], 1] -= 50 * t
        rows, skipped = p.filter_edge_truncations([event(1, .3, t[-1], 'front')],
                                                 {'time_s': t, 'keypoints': k}, f.load_config(), 'front')
        self.assertEqual(rows, [])
        self.assertEqual(skipped[0]['reason'], 'video_ends_during_ascent')

    def test_visible_stable_finish_at_video_edge_is_kept(self):
        t = np.arange(0, 2, 1 / 30)
        k = np.zeros((len(t), 17, 3)); k[:, :, 2] = 1
        for i, point in enumerate([5, 11, 13, 15]): k[:, point, 1] = 50 + 50 * i
        rows, skipped = p.filter_edge_truncations([event(1, .3, t[-1], 'front')],
                                                 {'time_s': t, 'keypoints': k}, f.load_config(), 'front')
        self.assertEqual(len(rows), 1)
        self.assertEqual(skipped, [])

if __name__ == '__main__':
    unittest.main()
