import numpy as np
from scipy.signal import savgol_filter

def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))

def smooth_valid(t, values, valid, seconds):
    out = np.full_like(values, np.nan, dtype=float)
    dt = float(np.median(np.diff(t)))
    blocks = []
    for lo, hi in runs(valid):
        cuts = np.flatnonzero(np.diff(t[lo:hi]) > max(.25, 3 * dt)) + lo + 1
        bounds = [lo] + cuts.tolist() + [hi]
        blocks.extend(zip(bounds[:-1], bounds[1:]))
    for lo, hi in blocks:
        if hi - lo < 3:
            out[lo:hi] = values[lo:hi]
            continue
        grid = np.linspace(t[lo], t[hi-1], max(3, round((t[hi-1]-t[lo])/dt)+1))
        window = max(3, round(seconds/(grid[1]-grid[0])) | 1)
        window = min(window, len(grid) if len(grid) % 2 else len(grid)-1)
        original = values[lo:hi].reshape(hi-lo, -1)
        result = np.empty_like(original)
        for j in range(original.shape[1]):
            v = np.interp(grid, t[lo:hi], original[:, j])
            v = savgol_filter(v, window, min(2, window-1))
            result[:, j] = np.interp(t[lo:hi], grid, v)
        out[lo:hi] = result.reshape(values[lo:hi].shape)
    return out, blocks

def derivative(t, values, blocks):
    out = np.full_like(values, np.nan, dtype=float)
    for a, b in blocks:
        if b-a >= 3:
            out[a:b] = np.gradient(values[a:b], t[a:b], axis=0)
    return out

def features(data, args):
    t = np.asarray(data['time_s'], dtype=float)
    k = np.asarray(data['keypoints'], dtype=float)
    if t.ndim != 1 or len(t) < 3 or np.any(~np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError('Cached PTS must be finite and strictly increasing.')
    if k.shape != (len(t), 17, 3):
        raise ValueError('Expected COCO keypoints with shape (frames, 17, 3).')
    from pose_geometry import choose_geometry
    geometry=choose_geometry(data,args)
    valid,scale=geometry['valid'],geometry['scale']
    if not np.isfinite(scale) or scale <= 1:
        raise ValueError('Not enough visible lower-body points to estimate scale.')
    hip,ankles=geometry['hip'],geometry['ankles']
    height, blocks = smooth_valid(t, hip[:, 1]-ankles[:, :, 1].mean(axis=1), valid, args.smooth_seconds)
    hip_s, _ = smooth_valid(t, hip, valid, args.smooth_seconds)
    ankle_s, _ = smooth_valid(t, ankles, valid, args.smooth_seconds)
    hip_velocity = derivative(t, hip_s, blocks)/scale
    ankle_velocity = derivative(t, ankle_s, blocks)/scale
    return dict(t=t, k=k, valid=valid, scale=scale, hip=hip_s, ankles=ankle_s,
                height=height, blocks=blocks, hip_velocity=hip_velocity,
                ankle_speed=np.linalg.norm(ankle_velocity, axis=2).max(axis=1),
                ankle_ids=geometry['ankle_ids'],geometry_metadata=geometry['metadata'])
