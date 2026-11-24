import numpy as np


def clothing_samples(source, data, min_seen_s, conf=.48):
    import av
    import cv2
    from multi_track import oriented_image

    f, uid, t, end, pose = (data[k] for k in ('frame_index','track_id','time_s','frame_end_s','keypoints'))
    boxes = data['boxes']
    width, height = int(data['width']), int(data['height'])
    core = pose[:, [5,6,11,12,13,14], :]
    full = np.isfinite(core).all(axis=(1,2)) & (core[:,:,2] >= conf).all(axis=1)
    full &= ((core[:,:,0] > 5) & (core[:,:,0] < width-5) &
             (core[:,:,1] > 5) & (core[:,:,1] < height-5)).all(axis=1)
    torso = np.linalg.norm(core[:,:2,:2].mean(axis=1)-core[:,2:4,:2].mean(axis=1),axis=1)
    full &= (torso >= 34) & (boxes[:, 3]-boxes[:, 1] >= 125)
    by_frame, observations = {}, {}
    for u in np.unique(uid):
        ix = np.flatnonzero(uid == u)
        frames = f[ix]
        if (end[frames]-t[frames]).sum() < min_seen_s:
            continue
        use = ix[full[ix]]
        if len(use) < 6:
            continue
        for j in use[np.unique(np.rint(np.linspace(0, len(use)-1, min(18, len(use)))).astype(int))]:
            by_frame.setdefault(int(f[j]), []).append(int(j))
    if not by_frame:
        return {}

    with av.open(str(source)) as container:
        last = max(by_frame)
        for i, frame in enumerate(container.decode(video=0)):
            if i in by_frame:
                image = oriented_image(frame, int(data['rotate']))
                for j in by_frame[i]:
                    q = descriptor(image, core[j,:,:2], torso[j], cv2)
                    if q is not None:
                        observations.setdefault(int(uid[j]), []).append(q)
            if i >= last:
                break
    return {u: np.asarray(x) for u,x in observations.items() if len(x) >= 5}


def descriptor(image, points, unit, cv2):
    shoulders = points[:2].mean(axis=0)
    hips = points[2:4].mean(axis=0)
    upper = [shoulders*(1-k)+hips*k for k in (.29,.48,.63)]
    lower = [h*(1-k)+z*k for h,z in zip(points[2:4],points[4:6]) for k in (.26,.43,.60)]
    parts = []
    for centers, radius in ((upper, .13),(lower, .072)):
        chunks = []
        r = max(3, round(radius*unit))
        for x,y in np.rint(centers).astype(int):
            if x-r<0 or y-r<0 or x+r>=image.shape[1] or y+r>=image.shape[0]:
                return None
            rgb = image[y-r:y+r+1,x-r:x+r+1]
            samples = cv2.cvtColor(rgb,cv2.COLOR_BGR2HSV).reshape(-1,3).astype(float)
            chunks.append(samples)
        hue,sat,val = np.concatenate(chunks).T
        chroma = np.clip((sat-25.)/67.,0.,1.)*np.clip((val-40.)/54.,0.,1.)
        blue = np.clip(1-abs(hue-108.)/32.,0.,1.)
        dvals = (np.median(val)/255., np.median(sat)/255., np.mean(chroma*blue),
                 np.mean(np.clip((val-118.)/91.,0.,1.)*(1-chroma)),
                 np.mean(chroma*(1-blue)))
        parts.append(dvals)
    return np.asarray(parts)


def distances(a, b):
    delta = abs(a[:,None]-b[None,:])
    x = delta @ np.array([.31,.16,.24,.23,.06])
    near = np.min(x @ np.array([.39,.61]), axis=1)
    back = np.min(x @ np.array([.39,.61]), axis=0)
    typical = abs(np.median(a,axis=0)-np.median(b,axis=0)) @ np.array([.31,.16,.24,.23,.06])
    return float(.52*(typical @ np.array([.39,.61]))+.48*max(np.median(near),np.median(back))), float(typical[1])


def join_returns(data, canonical, samples, max_return_s, *, threshold=.080, pants_limit=.13, ambiguity=.026):
    raw = data['track_id']
    out = canonical.copy()
    firstlast = {}
    groups = {}
    for k in np.unique(canonical):
        k = int(k)
        hit = canonical == k
        frames = data['frame_index'][hit]
        firstlast[k] = (int(frames[0]),int(frames[-1]))
        examples = {int(r): int(np.count_nonzero(raw[hit] == r)) for r in np.unique(raw[hit]) if int(r) in samples}
        groups[k] = {r for r,n in examples.items() if n >= .19 * max(examples.values())}
    costs = {}
    keys = list(groups)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            cross = [distances(samples[x],samples[y]) for x in groups[a] for y in groups[b]]
            if cross:
                costs[frozenset((a,b))] = max(c[0] for c in cross), max(c[1] for c in cross)
    def overlapping(a,b):
        return max(firstlast[a][0],firstlast[b][0]) <= min(firstlast[a][1],firstlast[b][1])
    t,e = data['time_s'],data['frame_end_s']
    order = sorted(keys, key=lambda a: firstlast[a][0])
    clusters = {a: {a} for a in keys}
    last_used = {a:a for a in keys}
    evaluated = []
    for b in order:
        start = firstlast[b][0]
        rows = []
        for a in sorted(clusters):
            if a == b or last_used[a] == b:
                continue
            prior = last_used[a]
            last = firstlast[prior][1]
            if last >= start:
                continue
            gap = float(t[start]-e[last])
            if gap > max_return_s:
                continue
            cross = [costs.get(frozenset((x,b))) for x in clusters[a]]
            if not cross or any(c is None for c in cross):
                continue
            dist, pants = max(c[0] for c in cross), max(c[1] for c in cross)
            if dist > threshold or pants > pants_limit:
                continue
            close_owner = False
            for other in keys:
                if other==b or other in clusters[a]:
                    continue
                if not (overlapping(other,b) or any(overlapping(other,x) for x in clusters[a])):
                    continue
                rival = costs.get(frozenset((other,b)))
                if rival and rival[1] <= pants_limit and rival[0] <= min(threshold,dist+ambiguity):
                    close_owner = True
                    break
            edge = dict(from_track_id=prior,to_track_id=b,gap_s=gap,appearance_distance=dist,
                        clothing_pants_distance=pants,method='pose_aligned_clothing_reentry',accepted=False,
                        reason='competing_covisible_clothing' if close_owner else 'eligible')
            evaluated.append(edge)
            if not close_owner:
                rows.append((dist,a,edge))
        rows.sort(key=lambda row: row[:2])
        if not rows:
            continue
        if len(rows)>1 and rows[1][0]-rows[0][0]<ambiguity:
            for _,_,edge in rows:
                edge['reason']='competing_reentry_match'
            continue
        _, a, edge = rows[0]
        edge.update(accepted=True,reason='unique_compatible_clothing')
        clusters[a].update(clusters.pop(b))
        last_used[a]=last_used.pop(b)
        out[canonical==b]=a
    return out, [edge for edge in evaluated if edge['accepted']], evaluated
