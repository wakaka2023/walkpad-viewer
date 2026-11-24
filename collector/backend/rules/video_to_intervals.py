import argparse
import csv
import json
import shutil
import hashlib
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, savgol_filter

def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))

def display_image(frame):
    rotation = float(getattr(frame, "rotation", 0) or 0)
    quarter_turns = int(round(rotation / 90))
    if abs(rotation - quarter_turns * 90) > 0.01:
        raise ValueError("Unsupported non-right-angle display rotation: {}".format(rotation))
    return np.ascontiguousarray(np.rot90(frame.to_ndarray(format="bgr24"), quarter_turns % 4))

def extract_pose(args, destination):
    import av
    from ultralytics import YOLO

    model = YOLO(args.model)
    if model.task != "pose":
        raise ValueError("Use a COCO 17-keypoint *-pose model.")
    selected_id = args.track_id
    timestamps, raw_pts, points, track_ids = [], [], [], []
    first_time = None
    shape = None
    tracking_events = []
    last_seen_time = None
    pending_id, pending_since = None, None
    with av.open(str(args.video)) as source:
        for index, frame in enumerate(source.decode(video=0)):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Video has no PTS; refusing to invent timing from FPS.")
            absolute_time = frame.pts * frame.time_base
            if first_time is None:
                first_time = absolute_time
            t = float(absolute_time - first_time)
            if timestamps and t <= timestamps[-1]:
                raise ValueError("Video PTS must increase strictly.")
            image = display_image(frame)
            roi = getattr(args, '_crop_roi', None)
            if roi is not None:
                x0, y0, x1, y1 = roi
                if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
                    raise ValueError('Crop ROI is outside source display dimensions.')
                image = np.ascontiguousarray(image[y0:y1, x0:x1])
            shape = image.shape[:2]
            options = dict(persist=True, tracker="bytetrack.yaml", verbose=False,
                           conf=args.box_conf, imgsz=args.imgsz)
            if args.device:
                options["device"] = args.device
            result = model.track(image, **options)[0]
            pose = np.full((17, 3), np.nan, dtype=np.float32)
            found_id = -1
            if result.boxes is not None and result.boxes.id is not None:
                ids = result.boxes.id.cpu().numpy().astype(int)
                if selected_id is None:
                    if len(ids) > 1:
                        raise ValueError("Multiple people: rerun with --track-id to select the participant.")
                    if len(ids) == 1:
                        observed = result.keypoints.data[0].cpu().numpy()

                        eligible = any(np.isfinite(observed[chain]).all() and (observed[chain, 2] >= args.keypoint_conf).all()
                                       for chain in ([5,11,13,15],[6,12,14,16]))
                        if eligible:
                            selected_id = int(ids[0])
                            tracking_events.append(dict(time_s=t,kind='initial_full_chain_selection',track_id=selected_id))
                matches = np.flatnonzero(ids == selected_id)
                if selected_id is not None and not len(matches) and args.track_id is None and len(ids)==1:
                    observed = result.keypoints.data[0].cpu().numpy()
                    eligible = any(np.isfinite(observed[chain]).all() and (observed[chain,2] >= args.keypoint_conf).all()
                                   for chain in ([5,11,13,15],[6,12,14,16]))
                    candidate = int(ids[0])
                    if eligible:
                        if pending_id != candidate:
                            pending_id, pending_since = candidate, t
                        if last_seen_time is not None and t-last_seen_time >= .5 and t-pending_since >= .3:
                            previous_id, selected_id = selected_id, candidate
                            tracking_events.append(dict(time_s=t,kind='single_person_reacquisition',
                                                        previous_track_id=previous_id,track_id=selected_id))
                            matches = np.flatnonzero(ids == selected_id)
                    else:
                        pending_id, pending_since = None, None
                elif selected_id is not None and not len(matches):
                    pending_id, pending_since = None, None
                if len(matches) == 1 and result.keypoints is not None:
                    k = result.keypoints.data[int(matches[0])].cpu().numpy()
                    if k.shape != (17, 3):
                        raise ValueError("Expected COCO 17x3 keypoints including confidence.")
                    pose = k
                    found_id = int(selected_id)
                    last_seen_time = t
                    pending_id, pending_since = None, None
            else:
                pending_id, pending_since = None, None
            timestamps.append(t)
            raw_pts.append(frame.pts)
            points.append(pose)
            track_ids.append(found_id)
            if index % 100 == 0:
                print("frame={}, video_time={:.3f}s, selected_track={}".format(index, t, selected_id), flush=True)
    if len(timestamps) < 3:
        raise ValueError("Need at least three decoded frames.")
    np.savez_compressed(destination, time_s=timestamps, raw_pts=raw_pts,
                        keypoints=points, track_id=track_ids,
                        height=shape[0], width=shape[1],
                        first_pts_seconds=float(first_time),
                        selected_track_id=-1 if selected_id is None else selected_id,
                        source_sha256=sha256_file(args.video),model_sha256=sha256_file(args.model),
                        tracking_events_json=json.dumps(tracking_events))

def get_signal(data, args):
    t = np.asarray(data["time_s"], dtype=float)
    k = np.asarray(data["keypoints"], dtype=float)
    if t.ndim != 1 or len(t) < 3 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("Cached times must be finite and strictly increasing.")
    if k.shape != (len(t), 17, 3):
        raise ValueError("Cached keypoints must have shape (frames, 17, 3).")
    choices = {}
    for side, indices in [("left", [11, 13, 15]), ("right", [12, 14, 16])]:
        p = k[:, indices, :2]
        c = k[:, indices, 2]
        a, b = p[:, 0] - p[:, 1], p[:, 2] - p[:, 1]
        lengths = np.linalg.norm(a, axis=1) + np.linalg.norm(b, axis=1)
        denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        valid = (np.isfinite(p).all(axis=(1, 2)) & np.isfinite(c).all(axis=1)
                 & (c >= args.keypoint_conf).all(axis=1) & (denom > 1e-6))
        angles = np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1) / np.maximum(denom, 1e-12), -1, 1)))
        choices[side] = (valid, angles, p[:, 0, 1], lengths)

    side = args.side
    if side == "auto":
        side = max(choices, key=lambda name: int(choices[name][0].sum()))
    valid, angle, hip_y, lengths = choices[side]
    angle[~valid] = np.nan
    dt = float(np.median(np.diff(t)))

    blocks = []
    for lo, hi in runs(valid):
        cuts = np.flatnonzero(np.diff(t[lo:hi]) > max(0.25, 3 * dt)) + lo + 1
        boundaries = [lo] + cuts.tolist() + [hi]
        blocks.extend(zip(boundaries[:-1], boundaries[1:]))
    smooth = np.full(len(t), np.nan)
    smooth_hip = np.full(len(t), np.nan)
    for lo, hi in blocks:
        if hi - lo < 3:
            smooth[lo:hi], smooth_hip[lo:hi] = angle[lo:hi], hip_y[lo:hi]
            continue
        grid = np.linspace(t[lo], t[hi - 1], max(3, round((t[hi - 1] - t[lo]) / dt) + 1))
        step = grid[1] - grid[0]
        window = max(3, int(round(args.smooth_seconds / step)) | 1)
        window = min(window, len(grid) if len(grid) % 2 else len(grid) - 1)
        for original, target in [(angle, smooth), (hip_y, smooth_hip)]:
            values = np.interp(grid, t[lo:hi], original[lo:hi])
            if window >= 3:
                values = savgol_filter(values, window, min(2, window - 1))
            target[lo:hi] = np.interp(t[lo:hi], grid, values)
    return t, valid, angle, smooth, smooth_hip, lengths, blocks, side

def label_knee_poses(data, args):
    t, valid, angle, smooth, hip, lengths, blocks, side = get_signal(data, args)
    dt = float(np.median(np.diff(t)))
    ends = np.r_[t[1:], t[-1] + dt]
    ends = np.minimum(ends, t + max(2 * dt, 0.05))
    labels = np.full(len(t), -1, dtype=int)
    rep_ids = np.zeros(len(t), dtype=int)
    intervals, rejected = [], []
    for lo, hi in blocks:
        stable = []
        for a, b in runs(smooth[lo:hi] >= args.standing_angle):
            a, b = a + lo, b + lo
            if t[b - 1] - t[a] + dt >= args.standing_hold:
                stable.append((a, b))
                labels[a:b] = 0
        if hi - lo < 5:
            continue
        peaks, _ = find_peaks(-smooth[lo:hi], prominence=args.min_excursion)
        groups = {}
        for p in peaks + lo:
            before = [(j, a, b) for j, (a, b) in enumerate(stable) if b - 1 < p]
            after = [(j, a, b) for j, (a, b) in enumerate(stable) if a > p]
            if not before or not after:
                rejected.append(dict(bottom_video_s=float(t[p]), reason="incomplete_or_missing_standing"))
                continue
            pre, post = before[-1], after[0]
            groups.setdefault((pre[0], post[0]), []).append((int(p), pre[1], pre[2], post[1], post[2]))
        for values in groups.values():
            if len(values) != 1:
                rejected.append(dict(bottom_video_s=float(t[values[0][0]]), reason="multiple_bottoms_between_stands"))
                continue
            p, pre_a, pre_b, post_a, post_b = values[0]
            a, b = pre_a, post_b - 1
            top = min(float(np.quantile(smooth[pre_a:pre_b], 0.8)),
                      float(np.quantile(smooth[post_a:post_b], 0.8)))
            excursion = top - float(smooth[p])
            level = top - args.boundary_fraction * excursion
            starts = np.flatnonzero(smooth[a:p + 1] >= level) + a
            finishes = np.flatnonzero(smooth[p:b + 1] >= level) + p
            if not len(starts) or not len(finishes):
                continue
            start, end = int(starts[-1] + 1), int(finishes[0])
            duration = t[end] - t[start]
            leg = float(np.nanmedian(lengths[a:b + 1]))
            top_hip = (float(np.median(hip[pre_a:pre_b])) + float(np.median(hip[post_a:post_b]))) / 2
            drop = float(hip[p] - top_hip) / max(leg, 1e-12)
            reason = None
            if excursion < args.min_excursion or smooth[p] > args.max_bottom_angle:
                reason = "insufficient_projected_knee_excursion"
            elif not args.min_duration <= duration <= args.max_duration:
                reason = "duration_out_of_range"
            elif not np.isfinite(drop) or drop < args.min_hip_drop:
                reason = "insufficient_projected_hip_drop"
            if reason:
                rejected.append(dict(bottom_video_s=float(t[p]), reason=reason))
                continue
            intervals.append(dict(rep_id=0, start_video_frame=start + 1, end_video_frame=end,
                                  start_video_s=float(t[start]), end_video_s=float(t[end]),
                                  bottom_video_s=float(t[p]), duration_s=float(duration),
                                  knee_excursion_deg=float(excursion), hip_drop_leg_ratio=drop,
                                  reviewed=0))
    intervals.sort(key=lambda x: x["start_video_s"])
    for number, event in enumerate(intervals, 1):
        event["rep_id"] = number
        a, b = event["start_video_frame"] - 1, event["end_video_frame"]
        if np.any(rep_ids[a:b]):
            raise ValueError("Overlapping detected intervals; inspect cached keypoints.")
        labels[a:b], rep_ids[a:b] = 1, number
    frames = []
    for i in range(len(t)):
        reason = "automatic_squat" if labels[i] == 1 else "standing_rule" if labels[i] == 0 else "uncertain_or_other_motion"
        if not valid[i]:
            reason = "missing_or_low_confidence_pose"
        frames.append(dict(video_frame=i + 1, video_time_s=float(t[i]), frame_end_s=float(ends[i]),
                           pose_valid=int(valid[i]), knee_angle_deg=float(angle[i]),
                           knee_angle_smoothed_deg=float(smooth[i]), squat_label=int(labels[i]),
                           rep_id=int(rep_ids[i]), reviewed=0, reason=reason))
    return intervals, frames, rejected, side

def label_poses(data, args):
    if args.boundary_mode == 'body':
        from .view_body_rules import label_view_poses
        events, frames, rejected, chosen = label_view_poses(data, args)
        if args._view_info['view'] == 'front' and getattr(args, 'front_onset_policy', 'raw-descent') == 'raw-descent':
            from .front_onset_rules import refine_front_onsets
            events, frames = refine_front_onsets(data, args, events, frames)
        if args._view_info['view'] == 'front' and getattr(args, 'front_recovery_policy', 'first-standing') == 'first-standing':
            from .front_recovery_rules import refine_front_recoveries
            events, frames = refine_front_recoveries(data, args, events, frames)
        if args._view_info['view'] == 'side' and getattr(args, 'side_onset_policy', 'grounded-descent') == 'grounded-descent':
            from .side_onset_rules import refine_side_onsets
            events, frames = refine_side_onsets(data, args, events, frames)
        return events, frames, rejected, chosen
    if args.boundary_mode == 'hip':
        from .activity_rules import label_hip_poses
        return label_hip_poses(data, args)
    return label_knee_poses(data, args)

def render_preview(video, out, data, frames, intervals, segments):
    from .pipeline_preview import render_preview as render
    return render(video,out,data,frames,intervals,segments,preset='veryfast')

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", nargs="?", type=Path)
    p.add_argument("--output", required=True, type=Path, help="A NEW output directory")
    p.add_argument("--keypoints", type=Path, help="Rerun rules on a previously saved keypoints.npz")
    bundled_model = Path(__file__).resolve().parent/'models'/'yolo26s-pose.pt'
    p.add_argument("--model", default=str(bundled_model) if bundled_model.is_file() else "yolo26s-pose.pt")
    p.add_argument("--device", default="")
    p.add_argument("--track-id", type=int)
    p.add_argument("--side", choices=["auto", "left", "right"], default="auto")
    p.add_argument('--boundary-mode', choices=['hip','knee','body'], default='body')
    p.add_argument('--view', choices=['auto','front','side'],default='auto')
    p.add_argument('--front-onset-policy', choices=['raw-descent','legacy'], default='raw-descent', help='Raw source body descent onset; legacy reproduces previous front onset rules')
    p.add_argument('--front-recovery-policy', choices=['first-standing','legacy'], default='first-standing', help='Check upright recovery stop for every front repetition; legacy retains previous ends')
    p.add_argument('--side-onset-policy', choices=['grounded-descent','legacy'], default='grounded-descent')
    p.add_argument('--side-onset-foot-motion', type=float, default=.10)
    p.add_argument('--side-onset-step-hold', type=float, default=.10)
    p.add_argument('--side-onset-reset-prominence', type=float, default=.02)
    p.add_argument('--side-onset-pretrend-max', type=float, default=.08)
    p.add_argument('--side-onset-min-context', type=float, default=.15)
    p.add_argument('--side-onset-torso-guard', type=float, default=.025)
    p.add_argument('--body-fraction',type=float,default=.03,help='Observed joint posture excursion threshold; combined with quiet-reference noise floor')
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--box-conf", type=float, default=0.3)
    p.add_argument("--keypoint-conf", type=float, default=0.5)
    p.add_argument("--smooth-seconds", type=float, default=0.27)
    p.add_argument('--hip-prominence',type=float,default=.12,help='Minimum relative hip-height excursion / leg length')
    p.add_argument('--min-peak-separation',type=float,default=.7)
    p.add_argument('--max-ankle-motion',type=float,default=.20,help='Reject cycles with larger foot-coordinate range / leg length')
    p.add_argument('--jump-rise-speed',type=float,default=1.7,help='Upward hip speed / leg length / second')
    p.add_argument('--jump-ankle-lift',type=float,default=.10,help='Minimum simultaneous bilateral ankle lift / leg length; avoids small projected ankle shifts during deep squats')
    p.add_argument('--walk-translation-speed',type=float,default=.25,help='Lateral torso speed / leg length / second; still requires foot-motion support')
    p.add_argument("--standing-angle", type=float, default=160)
    p.add_argument("--standing-hold", type=float, default=0.2)
    p.add_argument("--min-excursion", type=float, default=25)
    p.add_argument("--max-bottom-angle", type=float, default=140)
    p.add_argument("--boundary-fraction", type=float, default=0.1)
    p.add_argument("--min-hip-drop", type=float, default=0.04)
    p.add_argument("--min-duration", type=float, default=0.6)
    p.add_argument("--max-duration", type=float, default=20)
    p.add_argument("--preview", action="store_true")
    return p

def main(argv=None):
    args = parser().parse_args(argv)
    if not args.keypoints and not args.video:
        raise ValueError("Provide a video or --keypoints.")
    if not 0 < args.boundary_fraction < 0.5:
        raise ValueError("--boundary-fraction must be between 0 and 0.5.")
    if args.boundary_mode == 'body' and not 0 < args.body_fraction < .3:
        raise ValueError('--body-fraction must be between 0 and 0.3.')
    for name in ('side_onset_foot_motion','side_onset_step_hold','side_onset_reset_prominence',
                 'side_onset_pretrend_max','side_onset_min_context','side_onset_torso_guard'):
        if not np.isfinite(getattr(args,name)) or getattr(args,name)<=0:
            raise ValueError(name+' must be finite and positive.')
    if args.preview and not args.video:
        raise ValueError("--preview requires the matching source video.")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.keypoints:
        cache = args.output/'keypoints.npz'
        shutil.copy2(args.keypoints,cache)
    else:
        cache = args.output / "keypoints.npz"
        extract_pose(args, cache)
    with np.load(cache, allow_pickle=False) as data:
        if args.video and 'source_sha256' in data and str(data['source_sha256'])!=sha256_file(args.video):
            raise ValueError('Pose cache source hash differs from the supplied video.')
        intervals, frames, rejected, side = label_poses(data, args)
        from .activity_rules import classify_activities
        segments = classify_activities(data,frames,intervals,rejected,args)
        interval_fields = ["rep_id", "start_video_frame", "end_video_frame", "start_video_s", "end_video_s",
                           "bottom_video_s", "duration_s", "knee_excursion_deg", "hip_drop_leg_ratio", "reviewed",
                           'candidate_id','bottom_video_frame','end_exclusive_video_frame','ankle_motion_leg_ratio',
                           'start_sensitivity_first_frame','start_sensitivity_last_frame',
                           'end_sensitivity_first_frame','end_sensitivity_last_frame']
        for event in intervals:
            interval_fields += [k for k in event if k not in interval_fields]
        write_csv(args.output / "intervals.csv", intervals, interval_fields)
        frame_fields=list(dict.fromkeys(k for row in frames for k in row))
        write_csv(args.output / "frame_labels.csv", frames, frame_fields)
        write_csv(args.output / "rejected_candidates.csv", rejected, ['candidate_id','start_video_frame','end_video_frame',
                  'start_video_s','end_video_s','bottom_video_s','bottom_video_frame','ankle_motion_leg_ratio',
                  'upward_hip_speed_leg_s','bilateral_ankle_lift_ratio','reason'])
        write_csv(args.output/'activities.csv',segments,list(segments[0]))
        metadata = dict(algorithm_version='4.2.0',parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                        chosen_side=side, detected_complete_squats=len(intervals),
                        view_information=getattr(args,'_view_info',{}),
                        source_sha256=sha256_file(args.video) if args.video else None,
                        model_sha256=str(data['model_sha256']) if 'model_sha256' in data else None,
                        cache_source_hash_checked=bool(args.video and 'source_sha256' in data),
                        total_frames=len(frames), valid_pose_frames=sum(x["pose_valid"] for x in frames),
                        video_start_s=frames[0]["video_time_s"], video_end_s=frames[-1]["frame_end_s"],
                        timing="source PTS relative to first decoded frame; last frame duration estimated from median PTS spacing",
                        interval_convention="[start_video_s, end_video_s)",
                        frame_convention="1-based; start and end video frame numbers are both inclusive",
                        boundary_definition="View-specific 2-D body proposals. Front body cores crossing a verified intercycle upright hold are recomputed from that local reset. Shared upright holds separate adjacent body proposals. Ambiguous overlaps retain disjoint supported hip cores with CHECK BOUNDARY. Side rules retain v4.1 behavior. Side grounded-descent checks observed feet and raw shoulder/hip descent. Earlier torso movement must have coherent hip or knee motion above the local excursion threshold. Torso adjustment without that support is excluded. When lower-body support exists, a separate raw torso change-point fit can preserve a genuine torso-first start. Missing, ambiguous or poorly fitted evidence retains the original candidate and review flag. Set side-onset-policy=legacy to replay v3 side rules. No independent anatomical onset reference.",
                        label_status="automatic_unreviewed; rule revised using these recordings; no independent manual frame-boundary reference",
                        imu_used_for_detection=False,
                        activity_scope='video-only squat/jump/walking/standing proposals; walking from torso translation and visible foot-motion support',
                        preview_timing="source presentation timestamps preserved within 1/90000 second; no audio")
        (args.output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.preview:
            render_preview(args.video, args.output / "review.mp4", data, frames, intervals, segments)
    print("Detected {} candidate complete squats. Review intervals.csv before using as reference labels.".format(len(intervals)))

if __name__ == "__main__":
    main()
