import json
import unittest
from pathlib import Path
from uuid import uuid4

from recognize_multi_actions import completed_result
from video_to_intervals import sha256_file


class ResultReuseTests(unittest.TestCase):
    def test_only_matching_complete_and_intact_outputs_are_reused(self):
        dest = Path(__file__).resolve().parent / ('result_test_' + uuid4().hex)
        dest.mkdir()
        request = dict(source='sample', no_preview=False)
        stamp = dest / 'completed_signature.json'
        items = ['metadata.json', 'review.mp4', 'frame_labels.csv']
        try:
            metadata = dict(version='4.9.8-multi-person')
            (dest / items[0]).write_text(json.dumps(metadata), encoding='utf-8')
            (dest / items[1]).write_bytes(b'video')
            (dest / items[2]).write_bytes(b'track,frame,label')
            files = {item: sha256_file(dest / item) for item in items}
            stamp.write_text(json.dumps(dict(request=request, files=files)), encoding='utf-8')
            self.assertEqual(completed_result(dest, stamp, request, items), metadata)
            self.assertIsNone(completed_result(dest, stamp, dict(request, source='other'), items))
            self.assertIsNone(completed_result(dest, stamp, request, items[:-1]))
            (dest / items[2]).write_bytes(b'track,frame,LABEL')
            self.assertIsNone(completed_result(dest, stamp, request, items))
            (dest / items[1]).unlink()
            self.assertIsNone(completed_result(dest, stamp, request, items))
        finally:
            stamp.unlink(missing_ok=True)
            for item in items:
                (dest / item).unlink(missing_ok=True)
            dest.rmdir()
