from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
from importlib import metadata as package_metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
SCHEMA = 'squat-video-v4_2-observable-grade-v4'

def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b''):
            result.update(block)
    return result.hexdigest()

def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    temporary.replace(path)

def parser(video_config=None):
    from video_pipeline_adapter import read_config, add_video_arguments
    config_path, video_settings = read_config(video_config)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--video', required=True, type=Path, help='Source video')
    p.add_argument('--data', required=True, type=Path, help='Raw device TXT/TSV; selected feet only')
    p.add_argument('--output', type=Path, help='New output directory; default: results/video_timestamp')
    p.add_argument('--model', type=Path, default=(config_path.parent/video_settings['model']).resolve())
    p.add_argument('--device', default='auto', help='auto, 0, 1 or cpu')
    p.add_argument('--left-device', default='F8')
    p.add_argument('--right-device', default='F9')
    p.add_argument('--sync-mode', choices=['common-start','unmapped'], default='common-start',
                   help='Assume a common start for candidate labels; unmapped exports video labels and unlabeled data')
    p.add_argument('--resume', action='store_true', help='Resume this pipeline after validating inputs and pose cache')
    p.add_argument('--no-preview', action='store_true', help='Skip preview encoding; still export data and labels')
    p.add_argument('--video-review', type=Path, help='Optional per-repetition video review CSV')
    p.add_argument('--grading-config', type=Path, help='Optional grading settings; default: grading_config.json')
    p.add_argument('--preview-height', type=int, default=720)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--video-config', type=Path, default=config_path, help='Video v4 settings; default: final_parameters.json')
    add_video_arguments(p, video_settings)
    return p

def execute(args):
    started = time.perf_counter()
    for key in ('video','data','model'):
        path = getattr(args,key).expanduser().resolve()
        if not path.is_file():
            raise ValueError('{} file not found: {}'.format(key,path))
        setattr(args,key,path)
    if args.data.suffix.lower() not in {'.txt','.tsv'}:
        raise ValueError('--data requires a tab-separated device TXT/TSV file')
    if args.cpu_threads < 1 or args.preview_height < 200:
        raise ValueError('cpu-threads must be >=1; preview-height must be >=200')
    if not 0 < args.boundary_fraction < .5:
        raise ValueError('boundary-fraction must be between 0 and 0.5')
    if args.min_duration <= 0 or args.max_duration < args.min_duration:
        raise ValueError('Invalid repetition duration settings')
    if args.imgsz < 32 or args.smooth_seconds <= 0 or args.min_peak_separation <= 0:
        raise ValueError('imgsz, smoothing duration and peak separation must be positive')
    out = (args.output or HERE/'results'/(args.video.stem+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).expanduser().resolve()
    args.output = out
    if any(out == p or out in p.parents for p in (args.video,args.data,args.model)):
        raise ValueError('Output directory must not contain the input video, raw data or model')
    manifest_path = out/'pipeline_manifest.json'
    previous = None
    if out.exists():
        if not args.resume:
            raise ValueError('Output directory exists; use a new directory or --resume: '+str(out))
        if not manifest_path.is_file():
            raise ValueError('--resume requires an output directory created by this pipeline')
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous.get('schema') != SCHEMA:
            raise ValueError('Output pipeline version mismatch')
    elif args.resume:
        raise ValueError('--resume output directory does not exist')

    from grade_repetitions import load_config, run as grade_reps, make_video_overlay
    grading_config=load_config(args.grading_config)
    print('[1/5] Checking inputs and foot acceleration/gyroscope fields...',flush=True)

    from prepare_and_label import run as prepare_feet
    from foot_io import read_native_feet
    native, _ = read_native_feet(args.data,args.left_device,args.right_device)
    print('Native left/right samples: {} / {}'.format(len(native['left'][0]),len(native['right'][0])),flush=True)
    del native
    hashes = {name:sha256(getattr(args,name)) for name in ('video','data','model')}
    if previous and previous['input_sha256']['video'] != hashes['video']:
        raise ValueError('Video content changed; choose a new --output directory')
    os.environ.setdefault('YOLO_CONFIG_DIR',str(out/'.runtime/yolo'))
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    import torch
    import cv2
    import video_to_intervals as video
    from video_pipeline_adapter import make_video_args, process_video
    from pipeline_preview import render_preview
    from activity_rules import classify_activities
    torch.set_num_threads(args.cpu_threads)
    cv2.setNumThreads(min(2,args.cpu_threads))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = ('0' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    pose_config = {'video_sha256':hashes['video'],'model_sha256':hashes['model'],
                   'imgsz':args.imgsz,'box_conf':args.box_conf,'track_id':args.track_id,'keypoint_conf':args.keypoint_conf,'device':device,
                   'ultralytics':package_metadata.version('ultralytics'),'torch':torch.__version__,
                   'extractor_function_sha256':hashlib.sha256((inspect.getsource(video.extract_pose)+inspect.getsource(video.display_image)).encode('utf-8')).hexdigest()}
    cache = out/'video/keypoints.npz'
    use_cache = bool(previous and previous.get('pose_config')==pose_config and cache.is_file()
                     and previous.get('pose_cache_sha256')==sha256(cache))
    out.mkdir(parents=True,exist_ok=args.resume)
    state = {'schema':SCHEMA,'status':'running','stage':'video','started_utc':datetime.now(timezone.utc).isoformat(),
             'inputs':{key:str(getattr(args,key)) for key in ('video','data','model')},
             'input_sha256':hashes,'parameters':vars(args),'resolved_device':device,'pose_config':pose_config,
             'pose_cache_reused':use_cache,'wrist_used':False,'sync_confirmed':False,'stage_seconds':{}}
    state['video_algorithm_version']='4.2.0'
    state['video_config_sha256']=sha256(args.video_config)
    state['video_source_sha256']={p.name:sha256(p) for p in HERE.glob('*rules.py')}
    state['video_source_sha256']['video_to_intervals.py']=sha256(HERE/'video_to_intervals.py')
    state['grading_pose_features_sha256']=sha256(HERE/'grading_pose_features.py')
    state['pipeline_source_sha256']=sha256(HERE/'run.py')
    state['grading_source_sha256']=sha256(HERE/'grade_repetitions.py')
    state['individual_quality_source_sha256']=sha256(HERE/'individual_quality.py')
    state['motion_fusion_source_sha256']=sha256(HERE/'motion_fusion.py')
    state['video_review_sha256']=sha256(args.video_review) if args.video_review else None
    state['grading_config']=grading_config
    state['device_euler_angles_used']=False

    def save_state(stage):
        state['stage']=stage
        state['elapsed_seconds']=round(time.perf_counter()-started,3)
        write_json(manifest_path,state)

    save_state('video')
    try:
        if use_cache:
            print('[2/5] Reusing verified poses and recalculating repetition boundaries...',flush=True)
        else:
            print('[2/5] Running per-frame pose inference and repetition detection; device={}...'.format(device),flush=True)
        begin=time.perf_counter()
        va=make_video_args(args, device)
        vm=process_video(va, reuse_cached_poses=use_cache)
        state['stage_seconds']['video_inference_and_boundaries']=round(time.perf_counter()-begin,3)
        state['pose_cache_sha256']=sha256(cache)
        save_state('preprocess_and_map')
        print('[3/5] Native outlier repair, 8 Hz zero-phase low-pass, 50 Hz interpolation{}...'.format(' and candidate label mapping' if args.sync_mode=='common-start' else ' (no label mapping)'),flush=True)
        begin=time.perf_counter()
        dm=prepare_feet(args.data,out/'video',out/'data',args.sync_mode=='common-start',args.left_device,args.right_device)
        if args.sync_mode=='unmapped':
            for stale in ('feet_50hz_candidate_labels.csv','rep_boundaries_candidate.csv'):
                path=out/'data'/stale
                if path.is_file():path.unlink()
        state['stage_seconds']['feet_preprocess_and_mapping']=round(time.perf_counter()-begin,3)
        save_state('metrics_and_grading')
        print('[4/5] Calculating four foot IMU metrics and joint video/IMU grades...',flush=True)
        begin=time.perf_counter()
        gm,grades=grade_reps(out/'video',out/'data',out/'grading',mapped=args.sync_mode=='common-start',config_path=args.grading_config,review_path=args.video_review)
        state['stage_seconds']['metrics_and_grading']=round(time.perf_counter()-begin,3)
        save_state('preview')
        begin=time.perf_counter()
        preview=out/'video/review.mp4'
        if not args.no_preview:
            print('[5/5] Exporting full video with per-repetition grades and four metrics...',flush=True)
            import numpy as np
            with np.load(cache,allow_pickle=False) as kp:
                events,frames,rejected,_=video.label_poses(kp,va)
                activities=classify_activities(kp,frames,events,rejected,va)
                render_preview(args.video,preview,kp,frames,events,activities,max_height=args.preview_height,preset='ultrafast',frame_overlay=make_video_overlay(grades))
        else:
            if preview.is_file():preview.unlink()
            print('[5/5] Preview encoding skipped.',flush=True)
        state['stage_seconds']['preview']=round(time.perf_counter()-begin,3)
        for name in ('video','data','model'):
            if sha256(getattr(args,name)) != hashes[name]:
                raise RuntimeError('Input file changed during processing: '+name)
        summary={'schema':SCHEMA,'status':'complete','video_algorithm_version':'4.2.0','grading_algorithm_version':'observable-video-foot-v4.4','video':str(args.video),'data':str(args.data),
                 'video_frames':vm['total_frames'],'valid_pose_frames':vm['valid_pose_frames'],
                 'video_duration_s':vm['video_end_s'],'video_squat_candidates':vm['detected_complete_squats'],
                 'native_feet_samples':dm['native_samples'],'grid50_rows':dm['grid50_rows'],
                 'wrist_used':False,'raw_foot_cells_preserved':dm['raw_cells_preserved'],'source_files_unchanged':True,
                 'device_euler_angles_used':False,'candidate_grade_counts':gm['candidate_grade_counts'],
                 'grade_scope':gm['reference_scope'],'grades_expert_validated':False,
                 'sync_mode':args.sync_mode,'sync_confirmed':False,'unknown_packet_losses_recovered':False,
                 'candidate_only':args.sync_mode=='common-start','pose_cache_reused':use_cache,
                 'stage_seconds':state['stage_seconds'],'elapsed_seconds':round(time.perf_counter()-started,3),
                 'outputs':{'video_intervals':'video/intervals.csv','video_frame_labels':'video/frame_labels.csv',
                            'review_video':'video/review.mp4' if not args.no_preview else None,
                            'processed_50hz':'data/feet_50hz_unlabeled.csv','native_feet':'data/native_feet_preserved.tsv',
                            'labeled_50hz':'data/feet_50hz_candidate_labels.csv' if args.sync_mode=='common-start' else None,
                            'mapped_repetitions':'data/rep_boundaries_candidate.csv' if args.sync_mode=='common-start' else None,
                            'rep_metrics_and_grades':'grading/rep_metrics_and_grades.csv',
                            'grading_review':'grading/review.html',
                            'graded_50hz':'data/feet_50hz_candidate_grades.csv' if args.sync_mode=='common-start' else None,
                            'graded_native':'data/native_feet_candidate_grades.tsv' if args.sync_mode=='common-start' else None}}
        write_json(out/'pipeline_summary.json',summary)
        (out/'results.md').write_text(
            'Squat candidates: {}\n50 Hz rows: {}\nGrades: {}\nSynchronization: {} (unconfirmed)\n\n'
            'See video/review.mp4, video/intervals.csv, grading/review.html, '
            'grading/rep_metrics_and_grades.csv and data/feet_50hz_candidate_grades.csv.\n'.format(
                vm['detected_complete_squats'],dm['grid50_rows'],gm['candidate_grade_counts'],args.sync_mode),encoding='utf-8')
        state['status']='complete';save_state('complete')
        if (out/'error.txt').is_file():
            (out/'error.txt').unlink()
        print('Completed: {} squat candidates; 50 Hz rows: {}; grades: {}.\nOutput: {}'.format(vm['detected_complete_squats'],dm['grid50_rows'],gm['candidate_grade_counts'],out),flush=True)
        return summary
    except Exception as exc:
        state['status']='failed';state['error']=str(exc)
        save_state(state['stage'])
        (out/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise

def main(argv=None):
    config_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    config_parser.add_argument('--video-config', type=Path)
    config_args, _ = config_parser.parse_known_args(argv)
    return execute(parser(config_args.video_config).parse_args(argv))

if __name__=='__main__':
    try:
        main()
    except Exception as error:
        print('Processing failed: '+str(error),file=sys.stderr,flush=True)
        raise SystemExit(1)
