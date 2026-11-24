import numpy as np
from scipy.ndimage import median_filter

from .activity_rules import runs


STAIR_STATES = frozenset(('stairs_up', 'stairs_down'))


def phases(t, foot, scale, depth_bound):
    dt = float(np.median(np.diff(t)))
    reach = max(2, round(.09/dt))
    blur = 2*max(1, round(.04/dt))+1
    xy = median_filter(foot, size=(blur, 1), mode='nearest')
    speed = np.linalg.norm(xy[2*reach:]-xy[:-2*reach],axis=1)/(t[2*reach:]-t[:-2*reach])
    if len(speed) < 8:
        return []
    low, high = np.quantile(speed,[.25,.87])
    quiet = np.pad(speed <= min(low+.23*(high-low), .34*scale), reach, mode='edge')
    contacts = [(a,b) for a,b in runs(quiet) if t[b-1]-t[a]+dt >= .12]
    events = []
    for (a,b),(c,d) in zip(contacts, contacts[1:]):
        if not .12 <= t[c]-t[b-1] <= 1.25:
            continue
        pre = np.median(xy[a:b],axis=0)
        post = np.median(xy[c:d],axis=0)
        spread = max(np.ptp(xy[a:b,1]), np.ptp(xy[c:d,1]))
        dy = post[1]-pre[1]
        if abs(dy) <= max(.15*scale, 2.1*spread, depth_bound(a,b,c,d)):
            continue
        events.append((b,c,a,d,dy))
    return events


def detect(t, pose, valid, scale, camera_quiet, blocked, height):
    labels = np.full(len(t),'',object)
    evidence = np.full(len(t), np.nan)
    supported = np.zeros(len(t), bool)
    core = [5,6,11,12,13,14,15,16]
    usable = valid[:,core].all(axis=1) & camera_quiet & ~blocked & np.isfinite(scale)
    hips = pose[:, [11,12], 1].mean(axis=1)
    shoulders = pose[:, [5,6], 1].mean(axis=1)
    for a,b in runs(usable):
        span = t[a:b]
        if len(span) < 20 or span[-1]-span[0] < .78:
            continue
        dt = float(np.median(np.diff(span)))
        if dt > .085 or np.max(np.diff(span)) > 2.5*dt:
            continue
        sizes = scale[a:b]
        unit = float(np.median(sizes))
        if unit <= 12.:
            continue
        loc_hip, loc_shoulder = hips[a:b], shoulders[a:b]
        reach = max(3,round(.24/dt))

        def depth_bound(x,y,z,w):
            before = np.median(sizes[max(x,y-reach):y])
            after = np.median(sizes[z:min(w,z+reach)])
            return .60*height*abs(np.log(after/before))

        feet = [phases(span, pose[a:b,i,:2], unit, depth_bound) for i in (15,16)]
        sequence = sorted((p[0], side, p) for side, moves in enumerate(feet) for p in moves)
        previous = None
        for k, ((_,side,left),(_,other,right)) in enumerate(zip(sequence,sequence[1:])):
            start,end0,anchor0,_,dy0 = left
            st1,end,_,anchor1,dy1 = right
            if side==other or dy0*dy1<=0 or not -.25 <= span[st1]-span[end0] <= 1.15:
                continue
            if min(abs(dy0),abs(dy1)) < .38*max(abs(dy0),abs(dy1)):
                continue
            pre = slice(max(anchor0,start-reach),start)
            post = slice(end,min(anchor1,end+reach))
            vert = [np.median(v[post])-np.median(v[pre]) for v in (loc_hip,loc_shoulder)]
            sign = np.sign(dy0)
            if min(sign*np.asarray(vert)) <= max(.12*unit, depth_bound(pre.start,pre.stop,post.start,post.stop)):
                continue
            if min(sign*np.asarray(vert)) < .35*min(abs(dy0),abs(dy1)):
                continue
            kind = 'stairs_up' if sign<0 else 'stairs_down'
            strength = float(min(abs(dy0),abs(dy1))/unit)
            x, y = a+start, a+end+1
            better = ~np.isfinite(evidence[x:y]) | (strength > evidence[x:y])
            labels[x:y][better] = kind
            evidence[x:y][better] = strength
            if previous and previous[0] == k-1 and previous[1] == kind:
                supported[previous[2]:y] = True
            previous = (k,kind,x)
    recover_repeated_location(t, pose, valid, scale, usable, labels, evidence, supported)
    extend_anchored_traversals(t, pose, valid, scale, camera_quiet, blocked, labels, evidence, supported)
    return labels, evidence, supported


def recover_repeated_location(t, pose, valid, scale, usable, labels, evidence, supported):
    anchors = labels != ''
    if set(labels[anchors]) != STAIR_STATES:
        return
    dt = float(np.median(np.diff(t)))
    if not .012 <= dt <= .085:
        return
    ankle = pose[:,15:17,:2]
    anchor_feet = ankle[anchors]
    xys = anchor_feet.reshape(-1,2)
    center = np.median(xys,axis=0)
    unit = float(np.median(scale[anchors]))
    radius = np.maximum(np.quantile(abs(xys-center),.94,axis=0)+np.array([.36,.68])*unit,
                        np.array([.38,.85])*unit)
    if radius[0] > 1.25*unit or radius[1] > 2.4*unit:
        return
    up,down = [np.median(ankle[labels==q],axis=(0,1)) for q in ('stairs_up','stairs_down')]
    if np.any(abs(up-down) > np.array([.84,1.1])*unit):
        return
    span = max(2, round(.20/dt))
    compare = max(span+1,round(.50/dt))
    hips = pose[:,[11,12],1].mean(axis=1)
    shoulders = pose[:,[5,6],1].mean(axis=1)
    joints = np.column_stack((hips,shoulders,ankle[:,:,1]))
    available = np.column_stack((valid[:,[11,12]].all(axis=1),valid[:,[5,6]].all(axis=1),valid[:,15:17]))
    joints = np.where(available,joints,np.nan)
    feet_center = np.mean(ankle,axis=1)
    plausible = usable & (np.abs(feet_center-center) < radius).all(axis=1)
    step_change = np.zeros((len(t),2),bool)
    strengths = np.zeros(len(t))
    bilateral = np.zeros(len(t), bool)
    for i in np.flatnonzero(plausible):
        a,b = i-compare,i+compare+1
        if a<0 or b>len(t) or usable[a:b].mean()<.82:
            continue
        delta = np.median(joints[b-span:b],axis=0)-np.median(joints[a:a+span],axis=0)
        torso = delta[:2]
        feet = delta[2:]
        near = np.median(joints[i:i+span,:2],axis=0)-np.median(joints[i-span:i,:2],axis=0)
        sign = np.sign(torso[0])
        norm = scale[i]
        if not (np.isfinite(delta).all() and min(sign*torso) > .095*norm and max(sign*feet) > .125*norm
                and min(sign*feet) > -.065*norm and min(sign*near) > .010*norm):
            continue
        feet_delta_x = np.mean(ankle[b-span:b,:,0])-np.mean(ankle[a:a+span,:,0])
        if min(abs(torso)) < .70*abs(feet_delta_x):
            continue
        leg = ankle[a:b,0,1]-ankle[a:b,1,1]
        stagger = np.quantile(leg,.9)-np.quantile(leg,.1)
        if stagger < .12*norm:
            continue
        step_change[i, int(sign>0)] = True
        strengths[i] = max(sign*feet)/norm
        bilateral[i] = min(sign*feet) >= .105*norm
    for col,kind in enumerate(('stairs_up','stairs_down')):
        opposite = 'stairs_down' if col==0 else 'stairs_up'
        eligible = plausible & (labels!=opposite)
        direct = (labels==kind)
        state = (step_change[:,col] & eligible) | direct
        for a,b in runs(eligible & ~state):
            if a>0 and b<len(t) and state[a-1] and state[b] and t[b]-t[a] <= .47:
                state[a:b] = True
        for a,b in runs(state):
            if t[b-1]-t[a]+dt < .24:
                continue
            has_direct = direct[a:b].any()
            high = (np.quantile(strengths[a:b],.60) >= .19 and bilateral[a:b].mean() >= .22
                    and abs(hips[b-1]-hips[a]) >= .21*unit)
            supported[a:b] |= (has_direct or high)
            missing = ~np.isfinite(evidence[a:b])
            if missing.any():
                evidence[a:b][missing] = np.median(strengths[a:b][strengths[a:b]>0])
            labels[a:b] = kind


def extend_anchored_traversals(t, pose, valid, scale, camera_quiet, blocked, labels, evidence, supported):
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    anchors = labels != ''
    if set(labels[anchors]) != STAIR_STATES:
        return
    trunk = [5, 6, 11, 12]
    xy = pose[:, trunk, :2].mean(axis=1)
    usable = valid[:, trunk].all(axis=1) & camera_quiet & ~blocked & np.isfinite(xy).all(axis=1)
    dt = float(np.median(np.diff(t)))
    for lo, hi in runs(usable):
        local = np.flatnonzero(anchors[lo:hi]) + lo
        if len(local) < 8 or set(labels[local]) != STAIR_STATES:
            continue
        y = gaussian_filter1d(xy[lo:hi, 1].astype(float), max(1., .14/dt))
        route = np.quantile(xy[local, 0], [.02, .98])
        top, bottom = np.quantile(y[local-lo], [.04, .96])
        rise = float(bottom - top)
        if rise < .5*np.nanmedian(scale[local]):
            continue
        prominence = .42 * rise
        p = np.pad(y, (1, 1), mode='edge')
        extrema = sorted([(int(i-1), -1) for i in find_peaks(-p, prominence=prominence)[0]] +
                         [(int(i-1), 1) for i in find_peaks(p, prominence=prominence)[0]] +
                         [(0, 1 if y[0] > (top+bottom)/2 else -1),
                          (len(y)-1, 1 if y[-1] > (top+bottom)/2 else -1)])
        for (a, high), (b, low) in zip(extrema, extrema[1:]):
            sign = -high
            travel = sign*(y[b]-y[a])
            if high == low or travel < .72*rise:
                continue
            kind = 'stairs_up' if sign < 0 else 'stairs_down'
            opposite = 'stairs_down' if sign < 0 else 'stairs_up'
            part = slice(lo+a, lo+b+1)
            if not (anchors[part] & (labels[part] == kind)).any():
                continue
            edge = .04 * travel
            entered = np.flatnonzero(sign*(y[a:b+1]-y[a]) <= edge)
            exited = np.flatnonzero(sign*(y[b]-y[a:b+1]) <= edge)
            s, e = a+int(entered[-1])+lo, a+int(exited[0])+lo+1
            if e <= s or t[e-1]-t[s] < .35:
                continue
            segment = slice(s, e)
            if ((labels[segment] == opposite) & supported[segment]).mean() > .18:
                continue
            x = xy[segment, 0]
            margin = np.nanmedian(scale[local])*.30
            if ((x < route[0]-margin) | (x > route[1]+margin)).mean() > .17:
                continue
            if travel / max(travel, np.abs(np.diff(y[s-lo:e-lo])).sum()) < .76:
                continue
            same = labels[segment] == kind
            solid = supported[segment] & same
            strengths = evidence[segment][same & np.isfinite(evidence[segment])]
            if not len(strengths):
                continue
            islands = [(u,v) for u,v in runs(same & np.isfinite(evidence[segment]))
                       if t[s+v-1]-t[s+u]+dt >= .20]
            endpoint_tolerance = .22*rise
            landings = abs(y[s-lo]-(top if sign > 0 else bottom)) <= endpoint_tolerance
            landings &= abs(y[e-lo-1]-(bottom if sign > 0 else top)) <= endpoint_tolerance
            core_y = pose[s:e,trunk,1].reshape(e-s,2,2).mean(axis=2)
            k = max(1, min((e-s)//4, round(.18/dt)))
            change = sign*(np.median(core_y[-k:],axis=0)-np.median(core_y[:k],axis=0))
            separated = len(islands) >= 2 and t[s+islands[-1][0]]-t[s+islands[0][1]-1] >= .24
            label_supported = solid.any() or (separated and landings and min(change) >= .64*rise)
            while s > lo+a and labels[s-1] == kind:
                s -= 1
            while e <= lo+b and labels[e] == kind:
                e += 1
            segment = slice(s, e)
            same = labels[segment] == kind
            evidence[segment] = np.where(same, evidence[segment], np.median(strengths))
            labels[segment] = kind
            supported[segment] = label_supported
