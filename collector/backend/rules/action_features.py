import hashlib
from pathlib import Path

import numpy as np

from .activity_rules import smooth_valid, derivative, runs
from .video_to_intervals import display_image, sha256_file


def interpolate_short(t, values, valid, maximum):
    result = np.array(values, dtype=float, copy=True)
    good = np.asarray(valid, dtype=bool).copy()
    result[~good] = np.nan
    for a, b in runs(~good):
        if a > 0 and b < len(t) and t[b] - t[a - 1] <= maximum:
            weights = (t[a:b] - t[a - 1]) / (t[b] - t[a - 1])
            weights = weights.reshape((-1,) + (1,) * (result.ndim - 1))
            result[a:b] = result[a - 1] * (1 - weights) + result[b] * weights
            good[a:b] = True
    return result, good


def window_stat(t, values, seconds, statistic='median', minimum=.4):
    values = np.asarray(values, float)
    result = np.full(len(t), np.nan)
    left = np.searchsorted(t, t - seconds / 2)
    right = np.searchsorted(t, t + seconds / 2, side='right')
    if not len(t) or not len(values):
        return result
    lengths = right - left
    width = int(lengths.max())
    if not width:
        return result
    block_size = max(1, 262144 // width)
    offsets = np.arange(width)
    for start in range(0, len(t), block_size):
        stop = min(start + block_size, len(t))
        n = lengths[start:stop]
        index = left[start:stop, None] + offsets
        sample = values[np.minimum(index, len(values) - 1)]
        valid = (offsets < n[:, None]) & (index < len(values)) & np.isfinite(sample)
        counts = valid.sum(axis=1)
        ready = counts >= np.maximum(2, minimum * n)
        for size in np.unique(counts[ready]):
            rows = np.flatnonzero(ready & (counts == size))
            batch = sample[rows][valid[rows]].reshape(len(rows), size)
            if statistic == 'range':
                low, high = np.percentile(batch, [10, 90], axis=1)
                stats = high - low
            elif statistic == 'mean':
                stats = np.mean(batch, axis=1)
            elif statistic == 'upper':
                stats = np.percentile(batch, 75, axis=1)
            else:
                stats = np.median(batch, axis=1)
            result[start + rows] = stats
    return result


def finite_max(values):
    stacked = np.asarray(values)
    good = np.isfinite(stacked).any(axis=0)
    result = np.max(np.where(np.isfinite(stacked), stacked, -np.inf), axis=0)
    result[~good] = np.nan
    return result


def finite_mean(values):
    stacked = np.asarray(values)
    count = np.isfinite(stacked).sum(axis=0)
    return np.divide(np.nansum(stacked, axis=0), count,
                     out=np.full(stacked.shape[1:], np.nan), where=count > 0)


def joint_angle(points):
    first, last = points[:, 0] - points[:, 1], points[:, 2] - points[:, 1]
    denom = np.linalg.norm(first, axis=1) * np.linalg.norm(last, axis=1)
    angle = np.degrees(np.arccos(np.clip(np.sum(first * last, axis=1) /
                                        np.maximum(denom, 1e-9), -1, 1)))
    angle[denom < 1] = np.nan
    return angle


def pose_features(data, config):
    t, k = np.asarray(data['time_s'], float), np.asarray(data['keypoints'], float)
    if t.ndim != 1 or len(t) < 3 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Expected at least three strictly increasing source timestamps.')
    if k.shape != (len(t), 17, 3):
        raise ValueError('Expected source COCO keypoints with shape (frames, 17, 3).')
    valid = np.isfinite(k).all(axis=2) & (k[:, :, 2] >= config['keypoint_conf'])
    x = k[:, :, :2].copy()
    x[~valid] = np.nan
    for joint in range(17):
        filled, good = interpolate_short(t, x[:, joint], valid[:, joint], config['pose_gap_s'])
        x[:, joint], _ = smooth_valid(t, filled, good, config['pose_smooth_s'])
    lengths = []
    for s, h in [(5, 11), (6, 12)]:
        lengths.append(np.linalg.norm(x[:, s] - x[:, h], axis=1))
    torso = finite_mean(lengths)
    leg_lengths = []
    for h, n, a in [(11, 13, 15), (12, 14, 16)]:
        leg_lengths.append(np.linalg.norm(x[:, h] - x[:, n], axis=1) +
                           np.linalg.norm(x[:, n] - x[:, a], axis=1))
    legs = finite_mean(leg_lengths)
    ratio = torso / legs
    good_ratio = np.isfinite(ratio) & (ratio > .3) & (ratio < 1.5)
    ratio = float(np.clip(np.median(ratio[good_ratio]), .4, 1.)) if good_ratio.any() else .6
    torso_length_ratio = torso / (ratio * legs)
    torso = np.where(np.isfinite(torso), torso, ratio * legs)
    torso[torso < 8] = np.nan
    torso = window_stat(t, torso, .6)
    torso, _ = interpolate_short(t, torso, np.isfinite(torso), config['pose_gap_s'])
    shoulders = finite_mean([x[:, 5], x[:, 6]])
    hips = finite_mean([x[:, 11], x[:, 12]])
    torso_visible = np.isfinite(torso) & np.isfinite(hips).all(axis=1)
    out = dict(time_s=t, points=x, raw_valid=valid, torso_px=torso,
               shoulder=shoulders, hip=hips, torso_visible=torso_visible,
               both_legs_visible=valid[:, [11, 12, 13, 14, 15, 16]].all(axis=1),
               torso_length_ratio=torso_length_ratio)
    knee_angles, knee_motion, foot_motion, arm_motion, arm_reach, lean = [], [], [], [], [], []
    for name, s, e, w, h, n, a in [('left', 5, 7, 9, 11, 13, 15), ('right', 6, 8, 10, 12, 14, 16)]:
        relative_knee = (x[:, n] - x[:, h]) / torso[:, None]
        relative_foot = (x[:, a] - x[:, h]) / torso[:, None]
        relative_wrist = (x[:, w] - x[:, s]) / torso[:, None]
        angle = joint_angle(x[:, [h, n, a]])
        angle[~valid[:, [h, n, a]].all(axis=1)] = np.nan
        knee_angles.append(angle)
        out[name + '_knee_angle_deg'] = angle
        for value, destination in [(relative_knee, knee_motion), (relative_foot, foot_motion),
                                   (relative_wrist, arm_motion)]:
            finite = np.isfinite(value).all(axis=1)
            blocks = runs(finite)
            speed = np.linalg.norm(derivative(t, value, blocks), axis=1)
            destination.append(window_stat(t, speed, config['motion_window_s'], 'median'))
        axis = (x[:, h] - x[:, s]) / torso[:, None]
        drop = np.sum(relative_wrist * axis, axis=1)
        transverse = np.abs(relative_wrist[:, 0] * axis[:, 1] - relative_wrist[:, 1] * axis[:, 0])
        arm_reach.append((drop > config['arm_drop_ratio']) & (transverse > config['arm_transverse_ratio']))
        lean.append(np.degrees(np.arctan2(x[:, h, 0] - x[:, s, 0], x[:, h, 1] - x[:, s, 1])))
        out[name + '_arm_drop'] = drop
        out[name + '_arm_transverse'] = transverse
        out[name + '_knee_relative_y'] = relative_knee[:, 1]
        out[name + '_foot_relative_x'] = relative_foot[:, 0]
    out['knee_motion'] = finite_max(knee_motion)
    out['foot_motion'] = finite_max(foot_motion)
    out['arm_motion'] = finite_max(arm_motion)
    out['arm_reach'] = np.any(arm_reach, axis=0)
    changes = []
    for side in ('left', 'right'):
        drop, transverse = out[side + '_arm_drop'], out[side + '_arm_transverse']
        visible = np.isfinite(drop) & np.isfinite(transverse)
        if visible.any():
            first = int(np.flatnonzero(visible)[0])
            reference = visible & (t >= t[first]) & (t <= t[first] + 1.)
            changes.append(np.maximum(np.abs(drop - np.median(drop[reference])),
                                      np.abs(transverse - np.median(transverse[reference]))))
    out['arm_posture_change'] = finite_max(changes) if changes else np.full(len(t), np.nan)
    out['knee_flexion'] = 180 - finite_mean(knee_angles)
    out['knee_asymmetry'] = np.abs(knee_angles[0] - knee_angles[1])
    out['torso_lean_deg'] = finite_mean(lean)
    out['stance_width'] = np.linalg.norm(x[:, 15] - x[:, 16], axis=1) / torso
    out['hip_foot_height'] = finite_mean([(x[:, a, 1] - x[:, h, 1]) / torso for h, a in [(11, 15), (12, 16)]])
    out['head_hip_height'] = (hips[:, 1] - x[:, 0, 1]) / torso
    out['local_torso_ratio'] = torso / window_stat(t, torso, 6., 'upper')
    return out


def background_mask(shape, points, confidence, shrink):
    import cv2
    height, width = shape
    mask = np.full((height, width), 255, np.uint8)
    good = np.isfinite(points).all(axis=1) & (points[:, 2] >= confidence)
    if good.any():
        xy = points[good, :2] * shrink
        lo = np.floor(xy.min(axis=0) - 24).astype(int)
        hi = np.ceil(xy.max(axis=0) + 24).astype(int)
        cv2.rectangle(mask, (max(0, lo[0]), max(0, lo[1])),
                      (min(width - 1, hi[0]), min(height - 1, hi[1])), 0, -1)
    return mask


def mask_person_regions(shape, boxes, shrink):
    import cv2
    height, width = shape
    mask = np.full(shape, 255, np.uint8)
    for x0, y0, x1, y1 in boxes * shrink:
        pad = max(12., .1 * max(x1 - x0, y1 - y0))
        lo = np.maximum(0, np.floor([x0 - pad, y0 - pad])).astype(int)
        hi = np.minimum([width - 1, height - 1], np.ceil([x1 + pad, y1 + pad])).astype(int)
        cv2.rectangle(mask, tuple(lo), tuple(hi), 0, -1)
    return mask


def extract_camera(video, data, features, config, destination, *, frame_image=display_image, person_regions=None):
    import av
    import cv2
    t = features['time_s']
    k = np.asarray(data['keypoints'], float) if person_regions is None else None
    stride = max(1, round(1 / (config['camera_hz'] * np.median(np.diff(t)))))
    samples, matrices, valid, inliers, residual, indices = [], [], [], [], [], []
    previous_gray = previous_mask = None
    previous_index = None
    first = None
    source_count = 0
    cv2.setRNGSeed(0)
    with av.open(str(video)) as reader:
        for index, frame in enumerate(reader.decode(video=0)):
            source_count += 1
            absolute = frame.pts * frame.time_base
            first = absolute if first is None else first
            if index >= len(t) or abs(float(absolute - first) - t[index]) > 1 / 90000:
                raise ValueError('Source PTS differs from the pose cache during camera extraction.')
            if index % stride != 0:
                continue
            image = frame_image(frame)
            shrink = min(1., config['camera_width'] / image.shape[1])
            image = cv2.resize(image, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mask = (background_mask(gray.shape, k[index], .25, shrink) if person_regions is None
                    else mask_person_regions(gray.shape, person_regions(index), shrink))
            matrix = np.array([[1., 0., 0.], [0., 1., 0.]])
            good, fraction, error = False, 0., float('nan')
            if previous_gray is not None:
                corners = cv2.goodFeaturesToTrack(previous_gray, maxCorners=350, qualityLevel=.01,
                                                 minDistance=8, mask=previous_mask, blockSize=7)
                if corners is not None and len(corners) >= 16:
                    following, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, corners, None,
                                                                  winSize=(21, 21), maxLevel=3)
                    if following is None or status is None:
                        following = corners.copy()
                        status = np.zeros((len(corners), 1), np.uint8)
                    back, backward_status, _ = cv2.calcOpticalFlowPyrLK(gray, previous_gray, following, None,
                                                                       winSize=(21, 21), maxLevel=3)
                    if back is None or backward_status is None:
                        back = corners.copy()
                        backward_status = np.zeros_like(status)
                    p, q = corners[:, 0], following[:, 0]
                    ok = (status[:, 0] > 0) & (backward_status[:, 0] > 0)
                    ok &= np.linalg.norm(back[:, 0] - p, axis=1) < 1.5
                    inside = (q[:, 0] >= 0) & (q[:, 0] < gray.shape[1]) & (q[:, 1] >= 0) & (q[:, 1] < gray.shape[0])
                    ok &= inside
                    qint = np.clip(np.round(q).astype(int), [0, 0], [gray.shape[1] - 1, gray.shape[0] - 1])
                    ok &= mask[qint[:, 1], qint[:, 0]] > 0
                    if ok.sum() >= 16:
                        fit, flags = cv2.estimateAffinePartial2D(p[ok], q[ok], method=cv2.RANSAC,
                                                                ransacReprojThreshold=2., maxIters=1500,
                                                                confidence=.99, refineIters=10)
                        if fit is not None and flags is not None:
                            selected = flags[:, 0] > 0
                            fraction = float(selected.mean())
                            scale = float(np.linalg.norm(fit[:, 0]))
                            prediction = p[ok] @ fit[:, :2].T + fit[:, 2]
                            if selected.any():
                                error = float(np.median(np.linalg.norm(prediction[selected] - q[ok][selected], axis=1)))
                            good = fraction >= config['camera_min_inliers'] and .8 <= scale <= 1.25 and selected.sum() >= 12
                            if good:
                                matrix = fit.copy()
                                matrix[:, 2] /= shrink
            samples.append(t[index])
            matrices.append(matrix)
            valid.append(good)
            inliers.append(fraction)
            residual.append(error)
            indices.append(index)
            previous_gray, previous_mask, previous_index = gray, mask, index
    if source_count != len(t):
        raise ValueError('Camera extraction source count differs from the pose cache.')
    np.savez_compressed(destination, time_s=samples, matrices=matrices, valid=valid,
                        inlier_fraction=inliers, residual_px=residual, source_indices=indices,
                        source_sha256=sha256_file(video))
    return dict(time_s=np.array(samples), matrices=np.array(matrices), valid=np.array(valid),
                inlier_fraction=np.array(inliers), residual_px=np.array(residual), source_indices=np.array(indices))


def camera_features(features, camera, config, width, height):
    t, x, torso = features['time_s'], features['points'], features['torso_px']
    indices, matrices = camera['source_indices'].astype(int), camera['matrices']
    measurements = {name: np.full(len(indices), np.nan) for name in
                    ['camera_speed', 'hip_speed', 'foot_speed', 'hip_vertical_speed',
                     'hip_velocity_x', 'hip_velocity_y']}
    for j in range(1, len(indices)):
        if not camera['valid'][j]:
            continue
        a, b = indices[j - 1], indices[j]
        dt, scale = t[b] - t[a], torso[b]
        if dt > .5 or not np.isfinite(scale):
            continue
        matrix = matrices[j]
        center = np.array([width / 2, height / 2])
        movement = center @ matrix[:, :2].T + matrix[:, 2] - center
        measurements['camera_speed'][j] = np.linalg.norm(movement) / dt / scale
        hip = features['hip']
        if np.isfinite(hip[[a, b]]).all():
            change = hip[b] - (hip[a] @ matrix[:, :2].T + matrix[:, 2])
            measurements['hip_speed'][j] = np.linalg.norm(change) / dt / scale
            measurements['hip_vertical_speed'][j] = change[1] / dt / scale
            measurements['hip_velocity_x'][j] = change[0] / dt / scale
            measurements['hip_velocity_y'][j] = change[1] / dt / scale
        pair = x[[a, b]][:, [15, 16]]
        valid_feet = np.isfinite(pair).all(axis=(0, 2))
        if valid_feet.any():
            change = pair[1, valid_feet] - (pair[0, valid_feet] @ matrix[:, :2].T + matrix[:, 2])
            measurements['foot_speed'][j] = np.max(np.linalg.norm(change, axis=1)) / dt / scale
    for key, values in measurements.items():
        sample_times = t[indices]
        filled, valid = interpolate_short(sample_times, values, np.isfinite(values), .5)
        out = np.full(len(t), np.nan)
        for a, b in runs(valid):
            inside = (t >= sample_times[a]) & (t <= sample_times[b - 1])
            out[inside] = np.interp(t[inside], sample_times[a:b], filled[a:b])
        features[key + '_raw'] = out
        features[key] = window_stat(t, out, config['motion_window_s'], 'median')
    features['camera_coverage'] = window_stat(t, np.isfinite(features['camera_speed_raw']).astype(float),
                                             config['motion_window_s'], 'mean')
    features['hip_net_speed'] = np.hypot(window_stat(t, features['hip_velocity_x_raw'], 2., 'mean'),
                                          window_stat(t, features['hip_velocity_y_raw'], 2., 'mean'))
    return features
