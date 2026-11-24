import csv
import html
import json
from pathlib import Path
import shutil

import numpy as np

import action_features as features
import action_rules as rules
import video_to_intervals as annotation


ROOT = Path(__file__).resolve().parent
VERSION = '4.7-hierarchical-automatic-activities'


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def save_csv(path, rows, defaults):
    fields = list(dict.fromkeys(list(defaults) + [key for row in rows for key in row]))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def load_config(path=None):
    value = json.loads((ROOT / 'action_config.json').read_text(encoding='utf-8'))
    if path:
        override = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if not isinstance(override, dict) or set(override) - set(value):
            raise ValueError('Action configuration has unsupported keys.')
        value.update(override)
    for key, number in value.items():
        if key != 'version' and (isinstance(number, bool) or not isinstance(number, (int, float)) or not np.isfinite(number) or number <= 0):
            raise ValueError('Action parameter must be a finite positive number: ' + key)
    if value['keypoint_conf'] >= 1 or value['camera_min_inliers'] > 1 or value['cycle_boundary_fraction'] >= .5:
        raise ValueError('Confidence or crossing fraction is outside its valid range.')
    if value['cycle_min_s'] >= value['cycle_max_s'] or value['period_min_hz'] >= value['period_max_hz']:
        raise ValueError('Minimum must be smaller than maximum in action configuration.')
    return value


def load_npz(path):
    with np.load(path, allow_pickle=False) as packed:
        return {key: packed[key] for key in packed.files}


def cache_matches(cache, stamp, signature):
    if not cache.is_file() or not stamp.is_file():
        return False
    try:
        return json.loads(stamp.read_text(encoding='utf-8')) == signature
    except (ValueError, OSError):
        return False


def check_pose(data, source_hash, model_hash):
    if str(data.get('source_sha256', '')) != source_hash:
        raise ValueError('Imported pose cache source hash differs from the video.')
    if str(data.get('model_sha256', '')) != model_hash:
        raise ValueError('Imported pose cache model hash differs from the requested weights.')
    t = np.asarray(data.get('time_s', []), float)
    if len(t) < 3 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Pose cache needs at least three increasing source timestamps.')
    if np.asarray(data.get('keypoints', [])).shape != (len(t), 17, 3):
        raise ValueError('Invalid pose-cache keypoint shape.')
    return data


def write_html(path, segments, events, has_preview):
    rows = []
    for row in segments:
        cells = [row['segment_id'], row['activity_zh'], f"{row['start_video_s']:.3f}",
                 f"{row['end_video_s']:.3f}", row.get('repetition_count', 0) or '', row['evidence_level'],
                 f"{row['visible_feet_fraction']:.0%}"]
        rows.append(f"<tr onclick='seek({row['start_video_s']},{row['end_video_s']})'>" +
                    ''.join('<td>' + html.escape(str(value)) + '</td>' for value in cells) + '</tr>')
    title = '\u81ea\u52a8\u52a8\u4f5c\u533a\u95f4\u5019\u9009'
    note = '\u7ed3\u679c\u7531\u89c6\u9891\u7279\u5f81\u81ea\u52a8\u751f\u6210\uff0c\u672a\u7ecf\u4eba\u5de5\u590d\u6838\u3002\u8d70/\u8dd1\u3001\u8bbe\u5907\u64cd\u4f5c\u8bed\u4e49\u4e0d\u786e\u5b9a\u65f6\u4fdd\u7559\u5019\u9009\u6807\u8bb0\u3002'
    headings = ['\u6bb5\u53f7', '\u52a8\u4f5c', '\u5f00\u59cb\u79d2', '\u7ed3\u675f\u79d2', '\u5355\u6b21\u52a8\u4f5c\u7f16\u53f7', '\u8bc1\u636e', '\u8db3\u90e8\u53ef\u89c1']
    headings[4] = '\u91cd\u590d\u6b21\u6570'
    player = '<video id="v" src="review.mp4" controls preload="metadata"></video>' if has_preview else ''
    text = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>' + title + '</title>'
            '<style>body{font:15px system-ui;color:#203040;margin:24px}video{max-width:100%;max-height:65vh}'
            'table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:7px}tr{cursor:pointer}</style>'
            '<h1>' + title + '</h1><p>' + note + '</p>' + player + '<table><thead><tr>' +
            ''.join('<th>' + value + '</th>' for value in headings) + '</tr></thead><tbody>' + ''.join(rows) +
            '</tbody></table><script>const v=document.getElementById("v");let stop=null;'
            'function seek(a,b){if(v){stop=b;v.currentTime=a;v.play()}}'
            'if(v)v.addEventListener("timeupdate",()=>{if(stop!==null&&v.currentTime>=stop){v.pause();stop=null}});'
            '</script></html>')
    Path(path).write_text(text, encoding='utf-8')


def run(source, dest, command, *, config_path=None, no_preview=False, resume=False, font=None):
    source, dest = Path(source), Path(dest)
    args = annotation.parser().parse_args(command)
    if args.device in ('', 'auto'):
        import torch
        args.device = '0' if torch.cuda.is_available() else 'cpu'
    config = load_config(config_path)
    if dest.exists() and not resume:
        raise ValueError('Activity output exists; choose a new directory or use --resume.')
    dest.mkdir(parents=True, exist_ok=True)
    source_hash, model_hash = annotation.sha256_file(source), annotation.sha256_file(args.model)
    pose_signature = dict(source_sha256=source_hash, model_sha256=model_hash, imgsz=args.imgsz,
                          box_conf=args.box_conf, keypoint_conf=args.keypoint_conf, track_id=args.track_id,
                          extractor='v4.5-preserved-single-person-PTS')
    cache, stamp = dest / 'keypoints.npz', dest / 'pose_signature.json'
    if args.keypoints:
        data = check_pose(load_npz(args.keypoints), source_hash, model_hash)
        if args.keypoints.resolve() != cache.resolve():
            shutil.copy2(args.keypoints, cache)
        save_json(stamp, pose_signature)
        save_json(dest / 'pose_cache_origin.json', dict(imported_cache=str(args.keypoints.resolve()),
                  cache_sha256=annotation.sha256_file(cache), source_and_model_hash_checked=True,
                  inference_settings_from_imported_cache_unverified=True))
    elif resume and cache_matches(cache, stamp, pose_signature):
        data = check_pose(load_npz(cache), source_hash, model_hash)
    else:
        temporary = dest / 'keypoints.building.npz'
        args.video = source
        annotation.extract_pose(args, temporary)
        data = check_pose(load_npz(temporary), source_hash, model_hash)
        temporary.replace(cache)
        save_json(stamp, pose_signature)
    f = features.pose_features(data, config)
    signature = dict(source_sha256=source_hash, pose_sha256=annotation.sha256_file(cache),
                     feature_code_sha256=annotation.sha256_file(ROOT / 'action_features.py'),
                     camera_hz=config['camera_hz'], camera_width=config['camera_width'],
                     camera_min_inliers=config['camera_min_inliers'])
    camera_cache, camera_stamp = dest / 'camera_motion.npz', dest / 'camera_signature.json'
    if resume and cache_matches(camera_cache, camera_stamp, signature):
        camera = load_npz(camera_cache)
    else:
        temporary = dest / 'camera_motion.building.npz'
        camera = features.extract_camera(source, data, f, config, temporary)
        temporary.replace(camera_cache)
        save_json(camera_stamp, signature)
    f = features.camera_features(f, camera, config, int(data['width']), int(data['height']))
    rules.period_features(f, config)
    segments, frames, events, boundaries, rejected = rules.classify(f, config)
    flat = {key: value for key, value in f.items() if isinstance(value, np.ndarray) and value.shape == f['time_s'].shape}
    np.savez_compressed(dest / 'action_features.npz', **flat)
    save_csv(dest / 'activities.csv', segments, ['segment_id', 'activity', 'activity_zh', 'start_video_s', 'end_video_s'])
    save_csv(dest / 'raw_activities.csv', f['raw_segments'], ['segment_id', 'activity', 'start_video_s', 'end_video_s'])
    save_csv(dest / 'raw_frame_labels.csv', f['raw_frame_labels'], ['video_frame', 'video_time_s', 'activity'])
    save_csv(dest / 'frame_labels.csv', frames, ['video_frame', 'video_time_s', 'segment_id', 'activity'])
    save_csv(dest / 'action_events.csv', events, ['event_id', 'activity', 'start_video_s', 'peak_video_s', 'end_video_s'])
    save_csv(dest / 'boundary_events.csv', boundaries, ['boundary_id', 'video_frame', 'video_time_s', 'from_activity', 'to_activity'])
    save_csv(dest / 'rejected_candidates.csv', rejected, ['peak_video_s', 'reason'])
    save_json(dest / 'activities.json', dict(segments=segments, action_events=events, boundaries=boundaries, rejected=rejected))
    save_json(dest / 'action_config_used.json', config)
    preview_audit = None
    if not no_preview:
        from action_preview import render, verify
        temporary = dest / 'review.building.mp4'
        render(source, temporary, data, frames, segments, events, font_path=font)
        preview_audit = verify(source, temporary, data)
        temporary.replace(dest / 'review.mp4')
        save_json(dest / 'preview_audit.json', preview_audit)
    write_html(dest / 'review.html', segments, events, not no_preview)
    seconds = {}
    for row in segments:
        seconds[row['activity']] = seconds.get(row['activity'], 0.) + row['duration_s']
    metadata = dict(version=VERSION, source=str(source.resolve()), source_sha256=source_hash,
                    model_sha256=model_hash, frame_count=len(frames), segment_count=len(segments),
                    action_event_count=len(events), activity_seconds=seconds,
                    raw_segment_count=len(f['raw_segments']),
                    context_inferred_fraction=float(np.mean([row['context_inferred'] for row in frames])),
                    output_layers='activity_bouts;individual_repetitions;phase;observation_status;raw_rule_labels',
                    unknown_fraction=seconds.get('unknown', 0.) / sum(seconds.values()),
                    camera_fit_fraction=float(np.mean(camera['valid'])),
                    interval_convention='[start_video_s,end_video_s); frame numbers are 1-based and inclusive',
                    timing='Original relative source PTS; last frame end estimated from median source spacing',
                    status='automatic_unreviewed_candidates; development recordings are not an independent accuracy set',
                    semantics='Stepping speed is projected motion, not running confirmation; arm activity does not prove device use',
                    boundary_context='Review windows are not calibrated error bounds',
                    legacy_squat_grading_modified=False, preview=preview_audit,
                    code_hashes={p.name: annotation.sha256_file(p) for p in [ROOT / 'action_features.py', ROOT / 'action_rules.py', ROOT / 'action_temporal.py', ROOT / 'action_preview.py', ROOT / 'action_pipeline.py']})
    save_json(dest / 'metadata.json', metadata)
    print('ACTIVITY_READY ' + json.dumps(dict(output=str(dest), segments=len(segments), events=len(events), unknown_fraction=metadata['unknown_fraction']), ensure_ascii=True), flush=True)
    return metadata
