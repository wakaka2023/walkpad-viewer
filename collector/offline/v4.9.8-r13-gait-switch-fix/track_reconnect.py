import numpy as np

from multi_track import oriented_image


def endpoint_samples(data, min_seen_s):
    i, uid = data['frame_index'], data['track_id']
    t, e = data['time_s'], data['frame_end_s']
    tracks = {}
    for identity in np.unique(uid):
        obs = np.flatnonzero(uid == identity)
        frames = i[obs]
        if len(obs) >= 3 and (e[frames] - t[frames]).sum() >= min_seen_s:
            def sample(indices):
                return indices[np.unique(np.rint(np.linspace(0, len(indices) - 1, min(6, len(indices)))).astype(int))]
            tracks[int(identity)] = dict(begin=int(frames[0]), end=int(frames[-1]),
                                        start_samples=sample(obs[t[frames] <= t[frames[0]] + .65]),
                                        end_samples=sample(obs[t[frames] >= t[frames[-1]] - .65]))
    return tracks


def appearance(image, box):
    import cv2
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w < 20 or h < 45:
        return None
    parts = []
    for xa, xb, ya, yb in [(.20, .80, .19, .48), (.21, .79, .52, .81)]:
        patch = image[round(y1 + ya*h):round(y1 + yb*h), round(x1 + xa*w):round(x1 + xb*w)]
        if patch.size == 0:
            return None
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.int16)
        hue, sat, val = hsv.reshape(-1, 3).T
        labels = np.where((sat < 34) | (val < 42), np.minimum(val//32, 7),
                          8 + (hue//15) * 8 + (sat >= 116)*4 + np.minimum(val//64, 3))
        hist = np.bincount(labels, minlength=104)
        parts.append(np.sqrt(hist / hist.sum()))
    return np.asarray(parts)


def read_appearances(source, data, samples):
    import av
    by_frame = {}
    for j in sorted(samples):
        by_frame.setdefault(int(data['frame_index'][j]), []).append(int(j))
    found = {}
    if not by_frame:
        return found
    with av.open(str(source)) as reader:
        last = max(by_frame)
        for i, frame in enumerate(reader.decode(video=0)):
            if i in by_frame:
                image = oriented_image(frame, int(data['rotate']))
                for j in by_frame[i]:
                    desc = appearance(image, data['boxes'][j])
                    if desc is not None:
                        found[j] = desc
            if i >= last:
                break
    return found


def appearance_distance(left, right):
    values = np.sqrt(np.clip(1 - np.einsum('aij,bij->abi', left, right), 0, 1))
    combined = .45*values[:, :, 0] + .55*values[:, :, 1]
    return float(np.quantile(combined, .25))


def choose_links(candidates, ambiguity=.10):
    def best(edges):
        edges = sorted(edges, key=lambda e: e['cost'])
        if edges and (len(edges) == 1 or edges[1]['cost'] - edges[0]['cost'] >= ambiguity):
            return edges[0]
        return None
    outgoing = {e['from_track_id'] for e in candidates}
    incoming = {e['to_track_id'] for e in candidates}
    left = [best([e for e in candidates if e['from_track_id'] == cid]) for cid in sorted(outgoing)]
    right = [best([e for e in candidates if e['to_track_id'] == cid]) for cid in sorted(incoming)]
    return [e for e in left if e is not None and any(e is f for f in right)]


def reconnect(source, data, camera, min_seen_s, max_gap_s):
    if max_gap_s == 0:
        return data['track_id'].copy(), [], []
    tracks = endpoint_samples(data, min_seen_s)
    t, ends = data['time_s'], data['frame_end_s']
    cam_frame = camera['source_indices']
    geom_candidates = []
    for a, old in tracks.items():
        for b, new in tracks.items():
            if old['end'] >= new['begin']:
                continue
            gap = float(t[new['begin']] - ends[old['end']])
            if gap > max_gap_s:
                continue
            camera_between = (cam_frame > old['end']) & (cam_frame <= new['begin'])
            if camera_between.any() and not np.all(camera['valid'][camera_between]):
                continue
            left = np.median(data['boxes'][old['end_samples']], axis=0)
            right = np.median(data['boxes'][new['start_samples']], axis=0)
            h1, h2 = left[3]-left[1], right[3]-right[1]
            scale = min(h1,h2)/max(h1,h2)
            center_dist = float(np.linalg.norm((left[:2]+left[2:]-right[:2]-right[2:])*.5)/max(h1,h2))
            if scale < .55 or center_dist > .88:
                continue
            geom_candidates.append(dict(from_track_id=a, to_track_id=b, gap_s=gap,
                                        center_distance_box_heights=center_dist, height_ratio=float(scale)))
    needed = set()
    for edge in geom_candidates:
        needed.update(tracks[edge['from_track_id']]['end_samples'])
        needed.update(tracks[edge['to_track_id']]['start_samples'])
    found = read_appearances(source, data, needed) if needed else {}
    candidates = []
    for edge in geom_candidates:
        old, new = (tracks[edge[key]] for key in ('from_track_id', 'to_track_id'))
        left = [found[x] for x in old['end_samples'] if x in found]
        right = [found[x] for x in new['start_samples'] if x in found]
        if min(len(left), len(right)) < 3:
            continue
        d = appearance_distance(np.asarray(left), np.asarray(right))
        edge.update(appearance_distance=d,
                    cost=d + .20*edge['center_distance_box_heights'] + .04*edge['gap_s']/max_gap_s)
        if d <= .46 and edge['cost'] <= .62:
            candidates.append(edge)
    accepted = choose_links(candidates)
    previous = {e['to_track_id']: e['from_track_id'] for e in accepted}
    result = data['track_id'].copy()
    for identity in previous:
        canonical = identity
        while canonical in previous:
            canonical = previous[canonical]
        result[data['track_id'] == identity] = canonical
    return result, accepted, [dict(e, accepted=any(e is k for k in accepted)) for e in candidates]
