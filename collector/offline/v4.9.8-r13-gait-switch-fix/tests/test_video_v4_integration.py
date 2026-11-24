import ast
import hashlib
import io
import json
from pathlib import Path
import re
import tokenize
import unittest
from unittest.mock import patch

import run
import video_to_intervals as video
from video_pipeline_adapter import make_video_args, video_argv

HERE = Path(__file__).resolve().parents[1]

class VideoIntegrationTests(unittest.TestCase):
    def test_every_shipped_parameter_reaches_video_entrypoint(self):
        config = json.loads((HERE/'final_parameters.json').read_text(encoding='utf-8'))
        args = run.parser().parse_args(['--video','source.mp4','--data','feet.txt','--output','result'])
        actual = video.parser().parse_args(video_argv(make_video_args(args,'cpu'),Path('stage')))
        for key,expected in config['parameters'].items():
            with self.subTest(parameter=key):
                self.assertEqual(getattr(actual,key),expected)
        self.assertEqual(actual.boundary_mode,'body')
        self.assertEqual(actual.device,'cpu')

    def test_explicit_view_and_rule_overrides_reach_video(self):
        args = run.parser().parse_args(['--video','source.mp4','--data','feet.txt','--output','result',
                                       '--view','side','--body-fraction','.04','--side-onset-pretrend-max','.09'])
        actual = make_video_args(args,'0')
        self.assertEqual((actual.view,actual.body_fraction,actual.side_onset_pretrend_max),('side',.04,.09))

    def test_video_option_is_not_misread_as_video_config(self):
        with patch.object(run,'execute',side_effect=lambda args:args):
            args = run.main(['--video','source.mp4','--data','feet.txt'])
        self.assertEqual(args.video,Path('source.mp4'))
        self.assertEqual(args.video_config,HERE/'final_parameters.json')

    def test_bundled_weights_and_parameters_match_manifest(self):
        reference = json.loads((HERE/'reference_parameters.json').read_text(encoding='utf-8'))
        for name,expected in reference.items():
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((HERE/name).read_bytes()).hexdigest(),expected)

    def test_source_is_english_without_comments_or_docstrings(self):
        for path in HERE.rglob('*.py'):
            text = path.read_text(encoding='utf-8')
            with self.subTest(file=str(path.relative_to(HERE))):
                self.assertIsNone(re.search('[\u3400-\u9fff]',text))
                self.assertFalse(any(t.type==tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(text).readline)))
                for node in ast.walk(ast.parse(text)):
                    if isinstance(node,(ast.Module,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
                        self.assertIsNone(ast.get_docstring(node))

if __name__=='__main__':
    unittest.main()
