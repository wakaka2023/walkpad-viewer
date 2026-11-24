import json
from pathlib import Path
import uuid

import video_to_intervals as video

HERE = Path(__file__).resolve().parent

def read_config(path=None):
    path = Path(path or HERE/'final_parameters.json').expanduser().resolve()
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    actions = {a.dest: a for a in video.parser()._actions}
    parameters = config['parameters']
    for name, value in parameters.items():
        if name not in actions or name in {'video','output','preview','keypoints','help','model','device'}:
            raise ValueError('Unsupported video parameter: '+name)
        action = actions[name]
        if action.choices and value not in action.choices:
            raise ValueError('Invalid video parameter: '+name)
        if action.type:
            parameters[name] = action.type(value)
    return path, config

def add_video_arguments(parser, config):
    keys = set(config['parameters']) | {'track_id'}
    for action in video.parser()._actions:
        if action.dest in keys:
            parser.add_argument(*action.option_strings, type=action.type,
                                choices=action.choices, help=action.help,
                                default=config['parameters'].get(action.dest, action.default))

def make_video_args(args, device):
    target = video.parser().parse_args([str(args.video),'--output',str(args.output/'video')])
    for action in video.parser()._actions:
        name = action.dest
        if name not in {'help','video','output','preview','keypoints','device'} and hasattr(args,name):
            setattr(target, name, getattr(args, name))
    target.model = str(args.model)
    target.device = device
    target.preview = False
    return target

def video_argv(args, output, keypoints=None):
    result = [str(args.video)] if args.video is not None else []
    result += ['--output',str(output)]
    if keypoints is not None:
        result += ['--keypoints',str(keypoints)]
    for action in video.parser()._actions:
        name = action.dest
        if name in {'help','video','output','preview','keypoints'}:
            continue
        value = getattr(args, name)
        if value is not None:
            result += [action.option_strings[0],str(value)]
    return result

def process_video(args, reuse_cached_poses=False):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = output/'keypoints.npz' if reuse_cached_poses else None
    if cache is not None and not cache.is_file():
        raise ValueError('Verified pose cache is missing.')

    staging = output.parent/('.video_v4_'+uuid.uuid4().hex)
    staging.mkdir()
    generated = staging/'annotation'
    video.main(video_argv(args, generated, cache))
    metadata = json.loads((generated/'metadata.json').read_text(encoding='utf-8'))

    metadata['parameters']['output'] = str(output)
    metadata['parameters']['keypoints'] = str(output/'keypoints.npz') if cache else None
    metadata['pipeline_adapter'] = 'video_v4_2_main; observable_scoring_v4'
    (generated/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    output.mkdir(exist_ok=True)
    for path in generated.iterdir():
        path.replace(output/path.name)
    generated.rmdir()
    staging.rmdir()
    return metadata
