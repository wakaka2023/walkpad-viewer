import argparse
import json
import os
import sys
from importlib import metadata
from time import perf_counter
from functools import partial
from pathlib import Path

import numpy as np

import action_features
from action_pipeline import cache_matches, load_config, load_npz, save_csv, save_json
from multi_action_analysis import analyze, TABLES
from multi_track import extract, oriented_image
from scene_ignore import filter_tracks, parse_roi
from person_evidence import AUDIT_FIELDS, filter_person_tracks
from flat_depiction import FIELDS as FLAT_FIELDS, filter_flat_depictions
from display_ids import audit_id_map, number_outputs
from video_to_intervals import sha256_file

ROOT = Path(__file__).resolve().parent
TRACKER = ROOT / 'bytetrack_actions.yaml'
MODEL = ROOT / 'models' / 'yolo26s-pose.pt'
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.qt', '.mkv', '.avi', '.m4v'}
RESULT_FILES = ['metadata.json', 'track_links.csv', 'reentry_links.csv', 'ignored_tracks.csv', 'track_id_map.csv', 'frame_coverage.csv',
                'tracks.csv', 'action_config_used.json', 'person_evidence.csv', 'flat_depiction.csv',
                *(name + '.csv' for name in TABLES)]


def completed_signature(source, signature, config, args):
    from action_preview import find_font
    versions = {}
    for name in ('ultralytics', 'av', 'torch', 'torchvision', 'numpy', 'scipy', 'lap', 'Pillow',
                 'opencv-python', 'opencv-python-headless', 'opencv-contrib-python'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    font = None if args.no_preview else find_font(args.font)
    return dict(source=str(source.resolve()), tracking=signature, config=config, tracker=args.tracker,
                min_track_s=args.min_track_s, max_track_gap_s=args.max_track_gap_s,
                reconnect_gap_s=args.reconnect_gap_s, reentry_gap_s=args.reentry_gap_s,
                ignore_static_roi=[list(r) for r in args.ignore_static_roi], no_preview=args.no_preview,
                font_sha256=sha256_file(font) if font else None,
                python=sys.version, dependencies=versions, cuda_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
                code={p.name: sha256_file(p) for p in sorted(ROOT.glob('*.py'))})


def completed_result(dest, stamp, signature, outputs):
    try:
        saved = json.loads(stamp.read_text(encoding='utf-8'))
        if (saved['request'] == signature and set(saved['files']) == set(outputs)
                and all(sha256_file(dest/name) == saved['files'][name] for name in outputs)):
            return json.loads((dest/'metadata.json').read_text(encoding='utf-8'))
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def validated_cache(data):
    t, e = data['time_s'], data['frame_end_s']
    i, uid = data['frame_index'], data['track_id']
    n = len(i)
    if (len(t) < 3 or t.shape != e.shape or not np.isfinite(t).all() or not np.isfinite(e).all()
            or not np.all(e > t) or np.any(np.diff(t) <= 0) or not np.array_equal(e[:-1], t[1:])
            or np.any(i < 0) or np.any(i >= len(t)) or np.any(np.diff(i) < 0)
            or uid.shape != (n,) or np.any(uid <= 0) or data['keypoints'].shape != (n, 17, 3)
            or data['boxes'].shape != (n, 4) or data['confidence'].shape != (n,)):
        raise ValueError('Multitrack cache has invalid timeline, track IDs or pose data.')
    offsets = data['mask_offsets']
    if (len(offsets) != len(t) + 1 or offsets[0] != 0 or np.any(np.diff(offsets) < 0)
            or offsets[-1] != len(data['mask_boxes']) or data['mask_boxes'].shape != (int(offsets[-1]), 4)):
        raise ValueError('Multitrack cache has invalid foreground masks.')
    return data


def run(source, dest, args):
    started = perf_counter()
    elapsed = {}
    boundary = started

    def stage(name):
        nonlocal boundary
        now = perf_counter()
        elapsed[name] = round(now - boundary, 4)
        boundary = now

    def report(result_cache):
        print('TIMING ' + json.dumps(dict(elapsed, total_s=round(perf_counter()-started, 4),
                                          completed_cache=result_cache)), flush=True)

    import ultralytics

    if dest.exists() and not args.resume:
        raise ValueError('Output already exists. Use a new output folder or --resume.')
    dest.mkdir(parents=True, exist_ok=True)
    config = load_config(args.action_config)
    source_hash, model_hash = sha256_file(source), sha256_file(args.model)
    tracker_sha256 = sha256_file(args.tracker) if Path(args.tracker).is_file() else None
    if tracker_sha256 is None:
        bundled = Path(ultralytics.__file__).resolve().parent / 'cfg' / 'trackers' / args.tracker
        if not bundled.is_file():
            raise ValueError('Tracker YAML not found: ' + args.tracker)
        tracker_sha256 = sha256_file(bundled)
    signature = dict(source_sha256=source_hash, model_sha256=model_hash, tracker_sha256=tracker_sha256,
                     pose_code_sha256=sha256_file(ROOT/'multi_track.py'),
                     ultralytics_version=ultralytics.__version__, rotate=args.rotate, imgsz=args.imgsz,
                     box_conf=args.box_conf, device=args.device)
    done = dest/'completed_signature.json'
    outputs = RESULT_FILES + ([] if args.no_preview else ['review.mp4', 'preview_audit.json'])
    request = (completed_signature(source, signature, config, args)
               if tracker_sha256 == sha256_file(TRACKER) else None)
    if args.resume and not args.recompute_results and request is not None:
        result = completed_result(dest, done, request, outputs)
        if result is not None:
            stage('signatures_and_completed_check_s')
            report(True)
            print(f'READY {dest} | verified completed result reused', flush=True)
            return result
    done.unlink(missing_ok=True)
    stage('signatures_s')
    cache, stamp = dest/'tracks.npz', dest/'tracks_signature.json'
    tracks_cached = args.resume and cache_matches(cache, stamp, signature)
    if tracks_cached:
        data = validated_cache(load_npz(cache))
    else:
        temp = dest/'tracks.building.npz'
        data = validated_cache(extract(source, temp, args.model, imgsz=args.imgsz, conf=args.box_conf,
                                       device=args.device, tracker=args.tracker, rotate=args.rotate))
        temp.replace(cache)
        save_json(stamp, signature)
    stage('tracks_cached_s' if tracks_cached else 'tracking_s')
    camera_file, camera_stamp = dest/'camera_motion.npz', dest/'camera_signature.json'
    camera_sig = dict(source_sha256=source_hash, tracks_sha256=sha256_file(cache),
                      code_sha256=sha256_file(ROOT/'action_features.py'),
                      parameters={key: config[key] for key in ('camera_hz', 'camera_width', 'camera_min_inliers')})
    camera_cached = args.resume and cache_matches(camera_file, camera_stamp, camera_sig)
    if camera_cached:
        camera = load_npz(camera_file)
    else:
        offsets = data['mask_offsets']
        def regions(index):
            return data['mask_boxes'][offsets[index]:offsets[index+1]]
        temp = dest/'camera_motion.building.npz'
        camera = action_features.extract_camera(source, data, {'time_s': data['time_s']}, config, temp,
                                                frame_image=partial(oriented_image, rotate=args.rotate),
                                                person_regions=regions)
        temp.replace(camera_file)
        save_json(camera_stamp, camera_sig)
    stage('camera_cached_s' if camera_cached else 'camera_s')
    from track_reconnect import reconnect
    raw_all = data['track_id']
    data, ignored = filter_tracks(data, args.ignore_static_roi)
    save_csv(dest/'ignored_tracks.csv', ignored, ['yolo_track_id','start_video_s','end_video_s','removed_observations','reason','roi'])
    data, person_audit = filter_person_tracks(data, config)
    save_csv(dest/'person_evidence.csv', person_audit, AUDIT_FIELDS)
    non_person = {r['yolo_track_id'] for r in person_audit if r['status'] == 'rejected_person_evidence'}
    stage('person_evidence_s')
    data, flat_audit = filter_flat_depictions(source, data, partial(oriented_image, rotate=args.rotate))
    save_csv(dest/'flat_depiction.csv', flat_audit, FLAT_FIELDS)
    flat_depictions = {r['yolo_track_id'] for r in flat_audit if r['status'] == 'rejected_flat_depiction'}
    behind_surface = {r['yolo_track_id'] for r in flat_audit if r['status'] == 'rejected_behind_surface'}
    stage('flat_depiction_s')
    people, links, considered = reconnect(source, data, camera, args.min_track_s, args.reconnect_gap_s)
    returns, reentry_candidates = [], []
    if args.reentry_gap_s and len(np.unique(people)) > 1:
        from appearance_reid import clothing_samples, join_returns
        profiles = clothing_samples(source, data, args.min_track_s)
        people, returns, reentry_candidates = join_returns(data, people, profiles, args.reentry_gap_s)
    save_csv(dest/'reentry_links.csv', reentry_candidates, ['from_track_id','to_track_id','gap_s','appearance_distance','clothing_pants_distance','accepted','method','reason'])
    raw_ids = data['track_id']
    save_csv(dest/'track_links.csv', considered, ['from_track_id', 'to_track_id', 'gap_s', 'accepted'])
    internal_by_raw = {int(raw): int(group) for raw, group in zip(raw_ids, people)}
    data = dict(data, track_id=people, yolo_track_id=raw_ids)
    stage('reconnect_s')
    from cropped_gait import prepare
    data, cropped, suppressed = prepare(source,data,config['keypoint_conf'])
    stage('cropped_gait_s')
    tables, summaries = analyze(data, camera, config, args.min_track_s, args.max_track_gap_s, cropped=cropped)
    stage('analysis_s')
    data, display_ids = number_outputs(data, tables, summaries)
    id_rows = audit_id_map(raw_all, internal_by_raw, display_ids, {r['yolo_track_id'] for r in ignored}, set(suppressed), non_person=non_person, flat_depictions=flat_depictions, behind_surface=behind_surface)
    save_csv(dest/'track_id_map.csv', id_rows, ['yolo_track_id','track_id','internal_track_id','status'])
    stage('display_ids_s')
    for name, fields in TABLES.items():
        save_csv(dest/f'{name}.csv', tables[name], fields)
    save_csv(dest/'tracks.csv', summaries, ['track_id', 'accepted', 'observed_frames', 'observed_seconds'])
    save_json(dest/'action_config_used.json', config)
    simultaneous = np.zeros(len(data['time_s']), dtype=int)
    for row in tables['frame_labels']:
        if row['track_state'] in ('tracked','image_supported_continuation'):
            simultaneous[row['video_frame']-1] += 1
    save_csv(dest/'frame_coverage.csv', [dict(video_frame=i+1, video_time_s=float(t), frame_end_s=float(data['frame_end_s'][i]),
                  visible_track_count=int(simultaneous[i])) for i, t in enumerate(data['time_s'])],
             ['video_frame', 'video_time_s', 'frame_end_s', 'visible_track_count'])
    meta = dict(version='4.9.8-r13-gait-switch-fix-multi-person',
                context_bridge_frames=sum(bool(row.get('context_bridge_reason')) for row in tables['frame_labels']), source=str(source.resolve()), signature=signature, tracker=args.tracker,
                frame_count=len(data['time_s']), track_count=len(summaries), accepted_track_count=sum(row['accepted'] for row in summaries),
                simultaneous_track_frames=int((simultaneous >= 2).sum()), max_simultaneous_tracks=int(simultaneous.max()),
                activity_segment_count=len(tables['activities']), action_event_count=len(tables['action_events']),
                min_track_s=args.min_track_s, max_track_gap_s=args.max_track_gap_s,
                yolo_track_count=int(len(np.unique(raw_all))), short_gap_link_count=len(links),
                reconnect_gap_s=args.reconnect_gap_s,
                clothing_reentry_link_count=len(returns), reentry_gap_s=args.reentry_gap_s, ignored_static_tracks=ignored,
                identity_scope='per_video_track_id_with_conservative_clothing_reentry_not_biometric_identity',
                action_rules='v4.9.8-r13-gait-switch-fix_person_local_walk_to_run_with_staggered_turn_and_camera_fragment_edges',
                display_id_policy='contiguous_one_based_by_first_observed_frame_then_left_to_right',
                cropped_content_top_px=int(data.get('cropped_content_top_px',0)),
                cropped_banner_tracks=sorted(display_ids[j] for j in data.get('cropped_banner_track_ids',[]) if j in display_ids),
                cropped_gait_tracks=sorted(display_ids[j] for j in cropped if j in display_ids),
                cropped_duplicate_internal_track_ids=suppressed,
                cropped_position_inferred_frames=int(np.count_nonzero(data.get('cropped_position_inferred',[]))),
                rejected_person_evidence_tracks=sorted(non_person),
                person_evidence_removed_observations=sum(r['removed_observations'] for r in person_audit),
                rejected_flat_depiction_tracks=sorted(flat_depictions),
                flat_depiction_removed_observations=sum(r['removed_observations'] for r in flat_audit if r['yolo_track_id'] in flat_depictions),
                rejected_behind_surface_tracks=sorted(behind_surface),
                behind_surface_removed_observations=sum(r['removed_observations'] for r in flat_audit if r['yolo_track_id'] in behind_surface),
                interval_convention='[start_video_s,end_video_s)', preview=None)
    stage('export_tables_s')
    if not args.no_preview:
        from multi_action_preview import render, verify
        temp = dest/'review.building.mp4'
        render(source, temp, data, tables['frame_labels'], keypoint_conf=config['keypoint_conf'], font_path=args.font)
        stage('render_s')
        meta['preview'] = verify(source, temp, data)
        stage('verify_s')
        temp.replace(dest/'review.mp4')
        save_json(dest/'preview_audit.json', meta['preview'])
    save_json(dest/'metadata.json', meta)
    if request is not None:
        temp = dest/'completed_signature.building.json'
        save_json(temp, dict(request=request, files={name: sha256_file(dest/name) for name in outputs}))
        temp.replace(done)
    stage('finalize_s')
    report(False)
    print(f"READY {dest} | {meta['accepted_track_count']} track IDs, {meta['simultaneous_track_frames']} frames with simultaneous tracks", flush=True)
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--tracker', default=str(TRACKER), help='Ultralytics tracker YAML path or bundled tracker name.')
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--box-conf', type=float, default=.10)
    parser.add_argument('--rotate', type=int, choices=[0, 90, 180, 270], default=0,
                        help='Additional clockwise correction after metadata rotation.')
    parser.add_argument('--min-track-s', type=float, default=.6, help='Minimum cumulative observed seconds per ID.')
    parser.add_argument('--max-track-gap-s', type=float, default=.35, help='Split temporal analysis at longer track gaps.')
    parser.add_argument('--reconnect-gap-s', type=float, default=2.0, help='Maximum seconds for conservative offline ID reconnection; 0 disables it.')
    parser.add_argument('--reentry-gap-s', type=float, default=60., help='Maximum offline clothing re-entry gap; 0 disables.')
    parser.add_argument('--ignore-static-roi', type=parse_roi, action='append', default=[], help='Repeatable normalized x1,y1,x2,y2 fixed-camera ROI; removes only wholly-contained stationary tracks.')
    parser.add_argument('--action-config', type=Path)
    parser.add_argument('--font', type=Path)
    parser.add_argument('--no-preview', action='store_true')
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Reuse verified final results or compatible intermediate caches.')
    parser.add_argument('--recompute-results', action='store_true', help='Recompute tables and preview even when a verified completed result exists; --resume still reuses compatible tracking and camera caches.')
    args = parser.parse_args(argv)
    if args.recompute_results and not args.resume:
        parser.error('--recompute-results requires --resume.')
    if not 0 < args.box_conf < 1 or args.imgsz < 32 or not (np.isfinite(args.min_track_s) and args.min_track_s >= 0) or not (np.isfinite(args.max_track_gap_s) and args.max_track_gap_s >= 0) or not (np.isfinite(args.reconnect_gap_s) and args.reconnect_gap_s >= 0):
        parser.error('Check --box-conf, --imgsz and track-duration options.')
    if not np.isfinite(args.reentry_gap_s) or args.reentry_gap_s < 0:
        parser.error('--reentry-gap-s must be finite and nonnegative.')
    if not args.model.is_file():
        parser.error('Missing pose weights: ' + str(args.model))
    if args.device == 'auto':
        import torch
        args.device = '0' if torch.cuda.is_available() else 'cpu'
    paths = []
    for value in args.inputs:
        if value.is_file() and value.suffix.lower() in VIDEO_EXTENSIONS:
            paths.append(value.resolve())
        elif value.is_dir():
            it = value.rglob('*') if args.recursive else value.iterdir()
            paths.extend(sorted(p.resolve() for p in it if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS))
        else:
            parser.error('Missing or unsupported video: ' + str(value))
    sources = list(dict.fromkeys(paths))
    if not sources or len({source.stem.casefold() for source in sources}) != len(sources):
        parser.error('No videos found or output names collide.')
    for source in sources:
        run(source, args.output/source.stem, args)


if __name__ == '__main__':
    main()

