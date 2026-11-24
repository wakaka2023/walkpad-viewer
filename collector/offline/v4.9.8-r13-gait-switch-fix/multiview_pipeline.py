from __future__ import annotations
import bisect
import csv
from fractions import Fraction
import html
import json
from pathlib import Path
import numpy as np
import multiview_fusion as fusion
import video_to_intervals as annotation
VERSION = '4.5-dual-view'
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def save_csv(path, rows, fields):
    fields = list(dict.fromkeys(list(fields) + [k for row in rows for k in row if not k.startswith('_')]))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

def display_geometry(video, layout, split_ratio):
    import av
    if layout not in ('front-side', 'side-front'):
        raise ValueError('Dual layout must be front-side or side-front.')
    if not 0.2 <= split_ratio <= 0.8:
        raise ValueError('Split ratio must be in [0.2, 0.8].')
    with av.open(str(video)) as source:
        first = next(source.decode(video=0), None)
        if first is None:
            raise ValueError('Video has no decoded frames.')
        image = annotation.display_image(first)
    (h, w) = image.shape[:2]
    cut = int(round(w * split_ratio))
    if min(cut, w - cut, h) < 64:
        raise ValueError('Each panel must be at least 64 pixels wide/high.')
    (left, right) = ((0, 0, cut, h), (cut, 0, w, h))
    return ((h, w), dict(zip(layout.split('-'), (left, right))))

def prepare_view(source, dest, view, roi, command, resume=False):
    from activity_rules import classify_activities
    a = annotation.parser().parse_args(command + ['--view', view])
    if a.keypoints is not None:
        raise ValueError('Use --resume for independently validated per-panel pose caches; --keypoints is single-view only.')
    (a.video, a.output, a.preview, a._crop_roi) = (source, dest, False, tuple(roi))
    dest.mkdir(parents=True, exist_ok=True)
    signature = dict(source_sha256=annotation.sha256_file(source), model_sha256=annotation.sha256_file(a.model), crop_roi=list(roi), box_conf=a.box_conf, imgsz=a.imgsz, track_id=a.track_id, extractor='v4.4-tracking-with-v4.5-panel-crop')
    (cache, stamp) = (dest / 'keypoints.npz', dest / 'pose_signature.json')
    valid_cache = resume and cache.is_file() and stamp.is_file() and (json.loads(stamp.read_text(encoding='utf-8')) == signature)
    if not valid_cache:
        temporary = dest / 'keypoints.building.npz'
        annotation.extract_pose(a, temporary)
        temporary.replace(cache)
        save_json(stamp, signature)
    with np.load(cache, allow_pickle=False) as packed:
        data = {key: packed[key] for key in packed.files}
    if str(data.get('source_sha256', '')) != signature['source_sha256']:
        raise ValueError('Pose-cache source hash mismatch.')
    if str(data.get('model_sha256', '')) != signature['model_sha256']:
        raise ValueError('Pose-cache model hash mismatch.')
    if (int(data['width']), int(data['height'])) != (roi[2] - roi[0], roi[3] - roi[1]):
        raise ValueError('Cached panel geometry differs from declared layout.')
    (k, times) = (data['keypoints'], np.asarray(data['time_s'], float))
    visible = any(((np.isfinite(k[:, ids]).all(axis=(1, 2)) & (k[:, ids, 2] >= a.keypoint_conf).all(axis=1)).any() for ids in ([5, 11, 13, 15], [6, 12, 14, 16])))
    if visible:
        (intervals, frames, rejected, chosen) = annotation.label_poses(data, a)
        activities = classify_activities(data, frames, intervals, rejected, a)
    else:
        dt = float(np.median(np.diff(times)))
        ends = np.r_[times[1:], times[-1] + dt]
        frames = [dict(video_frame=i + 1, video_time_s=float(t), frame_end_s=float(ends[i]), pose_valid=0, rep_id=0, squat_label=-1, activity='unknown', reason='no_visible_shoulder_hip_knee_ankle_chain') for (i, t) in enumerate(times)]
        activities = [dict(segment_id=1, activity='unknown', start_video_s=float(times[0]), end_video_s=float(ends[-1]))]
        (intervals, rejected, chosen) = ([], [], None)
    save_csv(dest / 'intervals.csv', intervals, ['rep_id', 'start_video_frame', 'end_video_frame', 'start_video_s', 'end_video_s', 'bottom_video_s', 'duration_s', 'hip_drop_leg_ratio', 'ankle_motion_leg_ratio'])
    save_csv(dest / 'frame_labels.csv', frames, ['video_frame', 'video_time_s', 'rep_id', 'squat_label'])
    save_csv(dest / 'activities.csv', activities, ['segment_id', 'activity', 'start_video_s', 'end_video_s'])
    save_csv(dest / 'rejected_candidates.csv', rejected, ['candidate_id', 'reason'])
    metadata = dict(algorithm_version='4.2.0', package_version=VERSION, parameters={k: str(v) if isinstance(v, Path) else v for (k, v) in vars(a).items()}, view_information=getattr(a, '_view_info', {}), crop_roi=list(roi), chosen_side=chosen, source_sha256=signature['source_sha256'], model_sha256=signature['model_sha256'], total_frames=len(frames), keypoint_coordinates='local cropped-panel pixels', imu_used_for_detection=False, interval_convention='[start_video_s, end_video_s)')
    save_json(dest / 'metadata.json', metadata)
    return (data, intervals, frames)

def grade_view(dest, intervals, config_path):
    if not intervals:
        return
    import pandas as pd
    from grade_repetitions import load_config, video_features
    from individual_quality import assign
    cfg = load_config(config_path)
    reps = pd.DataFrame(intervals)
    features = video_features(dest, reps, cfg)
    table = assign(reps.merge(features, on='rep_id', validate='one_to_one'), cfg)
    view = json.loads((dest / 'metadata.json').read_text(encoding='utf-8'))['parameters']['view']
    for (i, raw) in enumerate(table.to_dict('records')):
        result = fusion.combine(raw if view == 'front' else None, raw if view == 'side' else None)
        for (key, value) in {'candidate_grade_code': result['grade_code'], 'candidate_grade_en': result['grade'], 'candidate_grade_zh': result['grade_zh'], 'grade_reasons': result['grade_reasons'], 'grade_provisional': 0, 'indirect_review_required': 0, 'grade_confidence': f"observed_{result['observed_items']}_of_6", 'observable_criteria_count': result['observed_items'], 'expected_observable_criteria_count': 6, 'grading_basis': 'VIDEO_ONLY'}.items():
            table.loc[i, key] = value
    (dest / 'grading').mkdir(exist_ok=True)
    table.to_csv(dest / 'grading' / 'rep_metrics_and_grades.csv', index=False, encoding='utf-8-sig')
    save_json(dest / 'grading' / 'grading_config_used.json', cfg)
    lookup = {int(r['rep_id']): r for r in table.to_dict('records')}
    for event in intervals:
        event['_grade'] = lookup[int(event['rep_id'])]

def filter_edge_truncations(events, data, cfg, view):
    t, k = np.asarray(data['time_s'], float), np.asarray(data['keypoints'], float)
    dt = float(np.median(np.diff(t)))
    rejected, kept = [], []
    for e in events:
        reason = None
        for edge, direction in (('start', 1), ('end', -1)):
            endpoint = t[0] if edge == 'start' else t[-1]
            if abs(e[edge + '_video_s'] - endpoint) > 2 * dt + 1e-6:
                continue
            mask = t <= t[0] + cfg['edge_context_window_s'] if edge == 'start' else t >= t[-1] - cfg['edge_context_window_s']
            best = None
            for ids in ([5, 11, 13, 15], [6, 12, 14, 16]):
                good = mask & np.isfinite(k[:, ids]).all(axis=(1, 2)) & (k[:, ids, 2] >= .5).all(axis=1)
                if good.sum() >= 4 and (best is None or good.sum() > best[0].sum()):
                    best = good, ids
            if best is None:
                continue
            good, ids = best
            points = k[good][:, ids, :2]
            scale = np.median(np.linalg.norm(points[:, 1] - points[:, 2], axis=1) + np.linalg.norm(points[:, 2] - points[:, 3], axis=1))
            height = points[:, :2, 1].mean(axis=1)
            slope = float(np.polyfit(t[good] - t[good][0], height, 1)[0]) / max(float(scale), 1.)
            if direction * slope > cfg['edge_moving_speed_leg_s']:
                reason = 'video_starts_during_descent' if edge == 'start' else 'video_ends_during_ascent'
                break
        if reason:
            rejected.append(dict(kind='truncated_video_edge', view=view, original_rep_id=int(e['rep_id']),
                                 start_video_s=e['start_video_s'], end_video_s=e['end_video_s'], reason=reason))
        else:
            kept.append(e)
    return kept, rejected


def write_report(path, events):
    rows = []
    for e in events:
        cells = [e['rep_id'], e['grade_zh'] or '\u4ec5\u533a\u95f4', f"{e['start_video_s']:.3f}", f"{e['bottom_video_s']:.3f}", f"{e['end_video_s']:.3f}", e['front_grade'] or '\u65e0\u89c2\u6d4b\u9879', e['side_grade'] or '\u65e0\u89c2\u6d4b\u9879', f"{e['observed_items']}/8", e['boundary_source'], e['grade_reasons']]
        content = ''.join(('<td>' + html.escape(str(x)) + '</td>' for x in cells))
        rows.append(f"""<tr onclick="playRep({e['start_video_s']},{e['end_video_s']})">{content}</tr>""")
    text = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>\u6b63\u4fa7\u9762\u7efc\u5408\u6df1\u8e72\u6807\u6ce8</title><style>body{font:15px system-ui;margin:24px;color:#203040}video{max-width:100%;max-height:70vh}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #ccc}tr{cursor:pointer}</style><h1>\u6b63\u4fa7\u9762\u7efc\u5408\u6df1\u8e72\u6807\u6ce8</h1><p>\u6bcf\u6b21\u52a8\u4f5c\u53ea\u6709\u4e00\u4e2a\u7efc\u5408\u597d\u4e2d\u5dee\u6807\u7b7e，\u4e24\u5e45\u753b\u9762\u540c\u6b65\u663e\u793a\u540c\u4e00\u7f16\u53f7、\u9636\u6bb5、\u7efc\u5408\u8bc4\u5206。\u6b63\u9762\u548c\u4fa7\u9762\u5404\u81ea\u7684\u7ed3\u679c\u4ec5\u4e3a\u5206\u89c6\u89d2\u8bc1\u636e。\u540c\u4e00\u6307\u6807\u53ea\u8ba1\u4e00\u6b21；\u7f3a\u5931\u9879\u4e0d\u8ba1\u5165\u901a\u8fc7\u9879，\u4e5f\u4e0d\u81ea\u52a8\u964d\u7ea7。\u6ca1\u6709\u4efb\u4f55\u53ef\u9760\u89c2\u6d4b\u9879\u7684\u5019\u9009\u4e0d\u8bc4\u5206，\u8bb0\u5f55\u4e8e\u8bca\u65ad\u6587\u4ef6。\u4e8c\u7ef4\u5de5\u7a0b\u89c4\u5219\u672a\u7ecf\u8fc7\u72ec\u7acb\u4e13\u5bb6\u51c6\u786e\u7387\u9a8c\u8bc1。</p><video id="v" src="review.mp4" controls preload="metadata"></video><p>\u70b9\u51fb\u8868\u683c\u884c\u64ad\u653e\u8be5\u6b21\u52a8\u4f5c。</p><table><thead><tr><th>\u6b21\u6570</th><th>\u7efc\u5408</th><th>\u5f00\u59cb\u79d2</th><th>\u6700\u4f4e\u70b9\u79d2</th><th>\u7ed3\u675f\u79d2</th><th>\u6b63\u9762\u8bc1\u636e</th><th>\u4fa7\u9762\u8bc1\u636e</th><th>\u5df2\u89c2\u6d4b</th><th>\u8fb9\u754c\u6765\u6e90</th><th>\u539f\u56e0</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table><script>const v=document.getElementById("v");let end=null;function playRep(a,b){end=b;v.currentTime=a;v.play()}v.addEventListener("timeupdate",()=>{if(end!==null&&v.currentTime>=end){v.pause();end=null}});</script></html>'
    Path(path).write_text(text, encoding='utf-8')

def event_at(events, starts, time):
    index = bisect.bisect_right(starts, time) - 1
    return events[index] if index >= 0 and time < events[index]['end_video_s'] else None

def frame_records(times, ends, events):
    starts = [e['start_video_s'] for e in events]
    for (i, time) in enumerate(times):
        event = event_at(events, starts, float(time))
        yield dict(video_frame=i + 1, video_time_s=float(time), frame_end_s=float(ends[i]), squat_rep_id=event['rep_id'] if event else 0, squat_label=int(event is not None), squat_phase=('DOWN' if time <= event['bottom_video_s'] else 'UP') if event else 'BETWEEN_REPS', overall_grade=event['grade'] if event else '', overall_grade_code=event['grade_code'] if event else '', observed_items=event['observed_items'] if event else '', possible_items=8, front_grade=event['front_grade'] if event else '', side_grade=event['side_grade'] if event else '')

def draw_frame(image, time, index, total, events, starts, data, rois):
    import cv2
    event = event_at(events, starts, time)
    code = event['grade_code'] if event else None
    color = {3: (80, 220, 80), 2: (60, 210, 255), 1: (80, 80, 255)}.get(code, (190, 190, 190))
    for (view, roi) in rois.items():
        (x0, y0, x1, y1) = roi
        panel = image[y0:y1, x0:x1]
        (h, w) = panel.shape[:2]
        points = np.asarray(data[view]['keypoints'][index])
        for (a, b) in SKELETON:
            pair = points[[a, b]]
            if np.isfinite(pair).all() and (pair[:, 2] >= 0.5).all():
                cv2.line(panel, tuple(np.rint(pair[0, :2]).astype(int)), tuple(np.rint(pair[1, :2]).astype(int)), color, 2, cv2.LINE_AA)
        for (x, y, conf) in points:
            if np.isfinite([x, y, conf]).all() and conf >= 0.5:
                cv2.circle(panel, (int(round(x)), int(round(y))), 3, (80, 255, 80), -1, cv2.LINE_AA)
        cues = []
        if event and event.get('_' + view):
            row = event['_' + view].get('_grade', {})
            cues = [c for c in json.loads(row.get('indirect_cues_json', '[]')) if c['start_s'] <= time < c['stop_s']]
            if cues:
                ids = json.loads(row.get('indirect_body_ids_json', '[[5,6],[11,12]]'))[:2]
                groups = [points[group] for group in ids]
                if all((np.isfinite(g).all() and (g[:, 2] >= 0.5).all() for g in groups)):
                    (a, b) = [tuple(np.rint(g[:, :2].mean(axis=0)).astype(int)) for g in groups]
                    cv2.line(panel, a, b, (240, 90, 230), 5, cv2.LINE_AA)
        line_step = max(16, min(25, h // 23))
        header_h = min(h, 6 * line_step + 8)
        shade = panel[:header_h].copy()
        panel[:header_h] = cv2.addWeighted(shade, 0.25, np.full_like(shade, 15), 0.75, 0)
        if event:
            phase = 'DOWN' if time <= event['bottom_video_s'] else 'UP'
            title = f"OVERALL: {event['grade'] or 'LABEL ONLY'}  |  SQUAT #{event['rep_id']}/{len(events)}  {phase}"
            local = event[view + '_grade'] or 'NO OBSERVED ITEMS'
            detail = f"{view.upper()} evidence: {local}  ({event[view + '_observed_items']}/6 observed)"
            count = f"FUSED: {event['observed_items']}/8 observed  |  {event['start_video_s']:.3f}-{event['end_video_s']:.3f}s"
            cs = json.loads(event['criteria_json'])
            issues = [f"{','.join(sorted({e['view'].upper() for e in c['evidence'] if e['level']}))}:{key}" for (key, c) in cs.items() if c['level']]
            reason = 'Issues: ' + (', '.join(issues) if issues else 'none in observed items')
            markers = [kind.upper() for kind in ('start', 'end') if 0 <= time - event[kind + '_video_s'] < 0.25]
            if abs(time - event['bottom_video_s']) <= 1 / 60 + 1e-06:
                markers.append('BOTTOM')
            state = ' | '.join(markers) or ('ACTIVE CUE: ' + cues[0]['label'] if cues else 'VIDEO ONLY | synchronized views')
        else:
            (title, detail, count, reason, state) = ('BETWEEN REPETITIONS', view.upper(), '', '', 'VIDEO ONLY')
        lines = [title, detail, count, reason, state, f'{time:.3f}s  |  Source frame {index + 1}/{total}']
        for (n, text) in enumerate(lines, 1):
            size = min(0.62, line_step / 36)
            measured = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, 1)[0][0]
            size *= min(1.0, (w - 18) / max(measured, 1))
            cv2.putText(panel, text, (8, n * line_step), cv2.FONT_HERSHEY_SIMPLEX, size, color if n == 1 else (240, 240, 240), 1, cv2.LINE_AA)
        cv2.rectangle(panel, (1, 1), (w - 2, h - 2), color, 3)
    return image

def render(source, destination, events, data, rois):
    import av
    t = np.asarray(data['front']['time_s'], dtype=float)
    starts = [e['start_video_s'] for e in events]
    tb = Fraction(1, 90000)
    count = 0
    with av.open(str(source)) as reader:
        rate = reader.streams.video[0].average_rate or Fraction(30)
        with av.open(str(destination), 'w') as writer:
            stream = None
            first = None
            for (index, frame) in enumerate(reader.decode(video=0)):
                if frame.pts is None or frame.time_base is None or index >= len(t):
                    raise ValueError('Source frame count/timing does not match pose caches.')
                absolute = frame.pts * frame.time_base
                first = absolute if first is None else first
                time = float(absolute - first)
                if abs(time - t[index]) > 1 / 90000:
                    raise ValueError('Source PTS differs from pose-cache PTS.')
                image = annotation.display_image(frame)
                if stream is None:
                    (h, w) = image.shape[:2]
                    stream = writer.add_stream('libx264', rate=rate)
                    (stream.width, stream.height) = (w, h)
                    stream.pix_fmt = 'yuv420p' if w % 2 == h % 2 == 0 else 'yuv444p'
                    stream.time_base = stream.codec_context.time_base = tb
                    stream.options = {'crf': '22', 'preset': 'veryfast', 'bf': '0'}
                draw_frame(image, time, index, len(t), events, starts, data, rois)
                output = av.VideoFrame.from_ndarray(image, format='bgr24')
                (output.pts, output.time_base) = (round(time * 90000), tb)
                for packet in stream.encode(output):
                    writer.mux(packet)
                count += 1
            if count != len(t):
                raise ValueError('Rendered frame count differs from pose caches.')
            if stream:
                for packet in stream.encode():
                    writer.mux(packet)
    with av.open(str(destination)) as reader:
        saved = np.array([float(f.pts * f.time_base) for f in reader.decode(video=0)])
    if len(saved) != len(t) or np.max(np.abs(saved - t), initial=0.0) > 1 / 90000 + 1e-08:
        raise ValueError('Saved preview failed PTS/frame-count verification.')
    return dict(frames=count, maximum_pts_error_s=float(np.max(np.abs(saved - t), initial=0.0)), output_sha256=annotation.sha256_file(destination), audio_included=False)

def run(source, dest, command, *, layout, split_ratio=None, fusion_config=None, grading_config=None, no_preview=False, labels_only=False, resume=False):
    (source, dest) = (Path(source), Path(dest))
    cfg = fusion.load_config(fusion_config)
    ratio = cfg['default_split_ratio'] if split_ratio is None else split_ratio
    (shape, rois) = display_geometry(source, layout, ratio)
    if dest.exists() and (not resume):
        raise FileExistsError('Output exists; use a new directory or --resume: ' + str(dest))
    dest.mkdir(parents=True, exist_ok=True)
    (data, intervals, frames) = ({}, {}, {})
    edge_diagnostics = []
    for view in ('front', 'side'):
        print('DUAL_VIEW_START ' + view, flush=True)
        (d, rows, fs) = prepare_view(source, dest / view, view, rois[view], command, resume)
        if not labels_only:
            grade_view(dest / view, rows, grading_config)
        rows, excluded = filter_edge_truncations(rows, d, cfg, view)
        edge_diagnostics.extend(excluded)
        (data[view], intervals[view], frames[view]) = (d, rows, fs)
    if not np.array_equal(data['front']['time_s'], data['side']['time_s']):
        raise ValueError('Front and side PTS arrays differ; refusing to combine.')
    (events, diagnostics) = fusion.fuse_events(intervals['front'], intervals['side'], cfg, labels_only=labels_only)
    diagnostics.extend(edge_diagnostics)
    times = np.asarray(data['front']['time_s'], dtype=float)
    ends = np.array([x['frame_end_s'] for x in frames['front']])
    for e in events:
        e['event_uid'] = source.stem + '_R' + f"{e['rep_id']:03d}"
        e['start_video_frame'] = int(np.searchsorted(times, e['start_video_s'], side='left')) + 1
        e['end_video_frame'] = int(np.searchsorted(times, e['end_video_s'], side='left'))
        e['bottom_video_frame'] = int(np.argmin(np.abs(times - e['bottom_video_s']))) + 1
    public = [{k: v for (k, v) in e.items() if not k.startswith('_')} for e in events]
    base_fields = ['rep_id', 'event_uid', 'start_video_s', 'end_video_s', 'bottom_video_s', 'grade', 'grade_zh', 'grade_code', 'observed_items', 'possible_items']
    save_csv(dest / 'repetitions.csv', public, base_fields)
    save_csv(dest / 'intervals.csv', public, base_fields)
    save_json(dest / 'events.json', public)
    write_report(dest / 'review.html', public)
    frame_rows = list(frame_records(times, ends, events))
    save_csv(dest / 'frame_labels.csv', frame_rows, list(frame_rows[0]))
    save_json(dest / 'fusion_diagnostics.json', diagnostics)
    save_json(dest / 'multiview_config_used.json', cfg)
    save_json(dest / 'layout_used.json', dict(layout=layout, split_ratio=ratio, crop_rois=rois, source_display_shape=shape, synchronization='User-declared same performance, synchronized panels in one original video; no offset inferred'))
    preview = None
    if not no_preview:
        preview = render(source, dest / 'review.mp4', events, data, rois)
    report = dict(version=VERSION, source_sha256=annotation.sha256_file(source), layout=layout, front_detections=len(intervals['front']), side_detections=len(intervals['side']), fused_repetitions=len(events), matched_both=sum((e['view_match'] == 'both' for e in events)), front_only=sum((e['view_match'] == 'front_only' for e in events)), side_only=sum((e['view_match'] == 'side_only' for e in events)), grade_counts={label: sum((e['grade'] == label for e in events)) for label in fusion.GRADE.values()}, scored_repetitions_have_one_of_three_grades=all((e['grade_code'] in (1, 2, 3) for e in events)) if not labels_only else None, no_evidence_detections_skipped=sum((d['kind'] == 'zero_observed_items' for d in diagnostics)), both_panels_show_same_overall_label=True, interval_convention='[start,end)', grade_basis='VIDEO_ONLY', sensor_csv_read=False, source_video_modified=False, preview=preview, expert_validated_accuracy=False)
    save_json(dest / 'multiview_audit.json', report)
    print('DUAL_READY ' + str(dest / ('repetitions.csv' if no_preview else 'review.mp4')), flush=True)
    return report
