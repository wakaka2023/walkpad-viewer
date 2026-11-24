from itertools import product

import cv2
import numpy as np

from person_evidence import OBSERVATION_FIELDS
from glass_surface import persistent_foreground_slats


FIELDS = ('yolo_track_id', 'status', 'reason', 'sampled_frames', 'registered_frames',
          'observed_frames', 'removed_observations', 'box_height_ratio', 'max_body_change',
          'max_wall_change', 'minimum_sheet_edge_contrast', 'foreground_surface_score', 'start_video_s', 'end_video_s')


def _register(images, box):
    reference = images[len(images)//2]
    h, w = reference.shape
    x1, y1, x2, y2 = box
    cw, ch = x2-x1, y2-y1
    mask = np.full((h,w),255,np.uint8)
    mask[max(0,int(y1-.27*ch)):int(y2+.31*ch)+1, max(0,int(x1-.36*cw)):int(x2+.40*cw)+1] = 0
    refs = cv2.goodFeaturesToTrack(reference, maxCorners=100, qualityLevel=.012, minDistance=5, mask=mask)
    if refs is None or len(refs) < 10:
        return None
    result = []
    for image in images:
        if image is reference:
            result.append(reference)
            continue
        matches, found, _ = cv2.calcOpticalFlowPyrLK(reference, image, refs, None, winSize=(19,19), maxLevel=2)
        if matches is None or found is None or np.count_nonzero(found) < 10:
            return None
        good = found[:,0].astype(bool)
        matrix, inliers = cv2.estimateAffinePartial2D(matches[good,0], refs[good,0], method=cv2.RANSAC,
                                                     ransacReprojThreshold=1.25)
        if matrix is None or inliers is None or np.count_nonzero(inliers) < 10 or inliers.mean() < .75:
            return None
        anchors = refs[good,0][inliers[:,0].astype(bool)]
        if np.ptp(anchors[:,0]) < cw or np.ptp(anchors[:,1]) < ch:
            return None
        result.append(cv2.warpAffine(image, matrix, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
    return np.asarray(result), mask != 0


def _side_options(image, center, along, side, radius):
    horizontal = side in (0, 2)
    fixed = np.linspace(along[0], along[1], 44, dtype=np.float32)
    origin = float(np.mean(along))
    slopes = np.linspace(-.24,.24,17)
    offsets = np.arange(np.ceil(center[0]),np.floor(center[1])+1)
    if not len(offsets):
        return []
    mm, oo = np.meshgrid(slopes, offsets, indexing='ij')
    estimate = (oo.reshape(-1,1) + mm.reshape(-1,1)*(fixed-origin)).astype(np.float32)
    vals = np.broadcast_to(fixed, estimate.shape).copy()
    sign = 1 if side in (0,3) else -1

    def sample(offset):
        q = estimate + offset
        return cv2.remap(image, vals if horizontal else q, q if horizontal else vals, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    delta = (sample(sign*radius) - sample(-sign*radius)).astype(np.float32)
    med = np.median(delta,axis=1)
    balance = np.mean(np.sign(med[:,None])*delta > 3, axis=1)
    ranked = np.argsort(-(abs(med)*balance), kind='stable')
    output = []
    for i in ranked:
        if abs(med[i]) < 8 or balance[i] < .75:
            continue
        m, q = float(mm.flat[i]),float(oo.flat[i])
        if any(abs(q-b0) <= 2 and abs(m-m0) <= .07 for m0,b0,_,_ in output):
            continue
        output.append((m,q,float(med[i]),float(balance[i])))
        if len(output) == 5:
            break
    return [(m, q-m*origin, a, b) for m,q,a,b in output]


def _sheet(image, box):
    x1,y1,x2,y2 = box
    w,h = x2-x1,y2-y1
    rad = np.clip(h*.021, 2., 4.)
    x = (x1+.11*w,x2-.11*w)
    y = (y1+.10*h,y2-.10*h)
    regions = (((y1-.20*h,y1+.09*h),x), ((x2-.12*w,x2+.32*w),y),
               ((y2-.09*h,y2+.28*h),x), ((x1-.25*w,x1+.12*w),y))
    choices = [_side_options(image, bounds, along, side, rad) for side,(bounds,along) in enumerate(regions)]
    best = None
    obj = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],np.float32)
    for top,right,bottom,left in product(*choices):
        sides = (top, right, bottom, left)
        signs = np.sign([z[2] for z in sides])
        if not np.all(signs == signs[0]) or abs(top[0]-bottom[0])>.13 or abs(left[0]-right[0])>.13:
            continue

        def cross(v, u):
            xp = (v[0]*u[1]+v[1])/(1-v[0]*u[0])
            return (xp, u[0]*xp+u[1])

        quad = np.array([cross(left,top),cross(right,top),cross(right,bottom),cross(left,bottom)],np.float32)
        if not cv2.isContourConvex(quad) or np.any(quad < 2) or np.any(quad >= np.array(image.shape[::-1])-2):
            continue
        area = cv2.contourArea(quad)
        if not 1.03*w*h <= area <= 1.85*w*h:
            continue
        shared,_ = cv2.intersectConvexConvex(obj,quad)
        if shared < .93*w*h:
            continue
        contrast = float(min(abs(z[2]) for z in sides))
        score = sum(abs(z[2])*z[3] for z in sides)
        if best is None or score > best[0]:
            best = (score,quad,contrast)
    return best


def _evidence(images, box):
    fit = _register(images,box)
    if fit is None:
        return 0, None, None, None, False
    frames, wall = fit
    smooth = np.stack([cv2.GaussianBlur(f,(5,5),.9).astype(np.float32) for f in frames])
    middle = np.median(smooth,axis=0)
    sheet = _sheet(middle,box)
    if sheet is None:
        return len(frames), None, None, None, False
    _, _, contrast = sheet
    x1, y1, x2, y2 = np.asarray(box,int)
    body = smooth[:,y1:y2,x1:x2] - middle[y1:y2,x1:x2]
    outside = smooth[:,wall] - middle[wall]
    tone = np.median(outside,axis=1)
    body_change = float(np.max(np.quantile(abs(body-tone[:,None,None]),.9,axis=(1,2))))
    wall_change = float(np.max(np.quantile(abs(outside-tone[:,None]),.9,axis=1)))
    return len(frames), body_change, wall_change, contrast, body_change <= 7.5 and body_change <= 2.5*max(wall_change,1.)


def filter_flat_depictions(source, data, frame_image):
    from collections import defaultdict
    import av

    ids = data['track_id']
    if not len(ids):
        return data, []
    plan = {}
    want = defaultdict(list)
    time, ends = data['time_s'], data['frame_end_s']
    hh, ww = int(data['height']), int(data['width'])
    for identity in np.unique(ids):
        obs = np.flatnonzero(ids == identity)
        seen = data['frame_index'][obs]
        if len(seen) < 12 or time[seen[-1]] - time[seen[0]] < 1.2:
            continue
        b = data['boxes'][obs]
        xy = np.median(b, axis=0)
        w, h = xy[2:] - xy[:2]
        if w < 24 or h < 35:
            continue
        if np.any(np.ptp(np.quantile(b,[.02,.98],axis=0),axis=0) > .18*h):
            continue
        center = (b[:,2:]+b[:,:2])/2
        if np.max(np.linalg.norm(center-np.median(center,axis=0),axis=1)) > .20*h:
            continue
        origin = np.floor(xy[:2]-[.95*w,.51*h]).astype(int)
        stop = np.ceil(xy[2:]+[.95*w,.55*h]).astype(int)
        if np.any(xy[:2] < [.27*w,.25*h]) or np.any(xy[2:]+[.36*w,.30*h] > [ww,hh]):
            continue
        ox,oy = np.maximum(origin,0)
        ex,ey = np.minimum(stop,[ww,hh])
        sample = seen[np.linspace(0,len(seen)-1,9).round().astype(int)]
        plan[identity] = (seen, xy-[ox,oy,ox,oy], [], len(sample), float(h/hh))
        for index in sample:
            want[index].append((identity,oy,ey,ox,ex))
    if not plan:
        return data, []
    with av.open(str(source)) as video:
        last = max(want)
        for n, frame in enumerate(video.decode(video=0)):
            if n in want:
                rgb = frame_image(frame)
                if rgb.shape[:2] != (hh,ww):
                    return data, []
                for u,oy,ey,ox,ex in want[n]:
                    plan[u][2].append(cv2.cvtColor(rgb[oy:ey,ox:ex],cv2.COLOR_RGB2GRAY))
            if n == last:
                break
    drop = []
    audit = []
    for u,(seen,box,images,expected,ratio) in sorted(plan.items()):
        r, body, wall, contrast, positive = _evidence(images,box) if len(images)==expected else (0,None,None,None,False)
        surface = persistent_foreground_slats(images, box) if not positive and len(images)==expected else None
        if positive or surface is not None:
            drop.append(u)
        if positive:
            status, reason = 'rejected_flat_depiction', 'stationary_wall_registered_texture_with_four_sided_sheet_border'
        elif surface is not None:
            status, reason = 'rejected_behind_surface', 'repeated_foreground_slat_phase_continues_over_body_and_flanks'
        else:
            status, reason = 'retained', 'insufficient_depiction_or_foreground_surface_evidence'
        audit.append(dict(yolo_track_id=int(u), status=status, reason=reason, sampled_frames=len(images), registered_frames=r,
                          observed_frames=len(seen), removed_observations=len(seen) if u in drop else 0,
                          box_height_ratio=ratio, max_body_change=body, max_wall_change=wall,
                          minimum_sheet_edge_contrast=contrast, foreground_surface_score=surface, start_video_s=float(time[seen[0]]),
                          end_video_s=float(ends[seen[-1]])))
    if not drop:
        return data, audit
    keep = ~np.isin(ids,drop)
    return dict(data, **{key:data[key][keep] for key in OBSERVATION_FIELDS}), audit
