import numpy as np


def parse_roi(value):
    from argparse import ArgumentTypeError
    try:
        a = tuple(float(x) for x in value.split(','))
        x1,y1,x2,y2 = a
    except ValueError as exc:
        raise ArgumentTypeError('ROI requires x1,y1,x2,y2 in normalized display coordinates') from exc
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ArgumentTypeError('ROI coordinates must be ordered and within [0,1]')
    return a


def filter_tracks(data, rois):
    ids = data['track_id']
    if not rois or not len(ids):
        return data, []
    removed = []
    units = np.array([data['width'],data['height'],data['width'],data['height']])
    for u in np.unique(ids):
        mask = ids == u
        b = data['boxes'][mask]
        if len(b)<8 or not np.isfinite(b).all():
            continue
        center = (b[:,2:]+b[:,:2])*.5
        height = b[:,3]-b[:,1]
        scale = float(np.median(height))
        extent = np.r_[b[:,:2].min(axis=0),b[:,2:].max(axis=0)]/units
        if scale<=0 or np.linalg.norm(np.ptp(np.quantile(center,[.05,.95],axis=0),axis=0))>.085*scale:
            continue
        if np.ptp(np.quantile(height,[.05,.95])) > .15*scale:
            continue
        for region in rois:
            rr = np.array(region)
            if np.all(extent[:2]>=rr[:2]) and np.all(extent[2:]<=rr[2:]):
                seen = data['frame_index'][mask]
                removed.append(dict(yolo_track_id=int(u),start_video_s=float(data['time_s'][seen[0]]),
                                    end_video_s=float(data['frame_end_s'][seen[-1]]),removed_observations=len(seen),
                                    reason='static_track_wholly_inside_configured_glass_roi',roi=','.join(map(str,region))))
                break
    if not removed:
        return data, removed
    keep = ~np.isin(ids, [r['yolo_track_id'] for r in removed])
    cleaned = dict(data)
    for name in ('frame_index','track_id','boxes','confidence','keypoints'):
        cleaned[name]=data[name][keep]
    return cleaned, removed
