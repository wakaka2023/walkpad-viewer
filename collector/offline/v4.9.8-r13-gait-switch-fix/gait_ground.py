import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from activity_rules import runs


def support_handoffs(time, feet, hip, scale, *, progression=False):
    if len(time) < 24:
        return []
    dt = float(np.median(np.diff(time)))
    if not (0 < dt <= .065) or not np.isfinite(scale) or scale <= 0:
        return []
    grid = np.arange(time[0], time[-1]+.1*dt, dt)
    if np.max(np.diff(time)) > 3.1*dt:
        return []
    xy = np.column_stack([np.interp(grid, time, v) for v in feet.reshape(-1,4).T]).reshape(-1,2,2)
    smooth = gaussian_filter1d(xy, .045/dt, axis=0)
    speed = np.linalg.norm(np.gradient(smooth, dt, axis=0),axis=2)
    diff = speed[:,0]-speed[:,1]
    lo,hi = np.quantile(speed,[.18,.87])
    if hi <= 3.0*lo or hi <= 0:
        return []
    swing_threshold = np.quantile(abs(diff),.75)*.58
    peaks=[]
    for sign in (1,-1):
        picks=find_peaks(sign*diff, prominence=swing_threshold, height=swing_threshold,
                         distance=max(2,round(.22/dt)))[0]
        peaks.extend((int(k),sign) for k in picks)
    peaks.sort()
    quiet = np.max(speed,axis=1) <= lo+.25*(hi-lo)
    vertical = np.interp(grid,time,hip)
    vertical -= gaussian_filter1d(vertical,.27/dt,mode='nearest')
    result=[]
    plates=[]
    for (a,left),(b,right) in zip(peaks,peaks[1:]):
        period = grid[b]-grid[a]
        if left==right or not .22 < period < 1.55:
            continue
        bridge = quiet[a+1:b]
        pauses = [(x,y) for x,y in runs(bridge) if (y-x)*dt >= .060]
        if not pauses:
            continue
        x,y = max(pauses,key=lambda pair:pair[1]-pair[0]); x+=a+1; y+=a+1
        if (y-x)*dt > .63*period:
            continue
        stride_span = max(np.linalg.norm(smooth[b,0]-smooth[a,0]),np.linalg.norm(smooth[b,1]-smooth[a,1]))
        bounce = float(np.quantile(vertical[a:b+1],.85)-np.quantile(vertical[a:b+1],.15))
        if stride_span <= 0 or bounce > .62*stride_span:
            continue
        result.append((grid[a],grid[b],len(pauses)))
        stance = smooth[x:y]
        center = np.median(stance,axis=0)
        jitter = np.max(np.linalg.norm(stance-center,axis=2),axis=0)
        plates.append((a,b,center,jitter))
    if not progression:
        return result
    advancing = []
    for (a,b,left,noise_left),(c,d,right,noise_right) in zip(plates,plates[1:]):
        if b != c:
            continue
        travel = right-left
        forward = np.sum(travel,axis=0)
        magnitude = np.linalg.norm(forward)
        if magnitude <= 0 or np.any((travel @ forward)/magnitude <= noise_left+noise_right):
            continue
        advancing.append((float(grid[a]),float(grid[d]),float(min(grid[b]-grid[a],grid[d]-grid[c]))))
    return advancing


def grounded_walk(t, pose, valid, scale, candidate, quiet_cam, runs_direct):
    support = np.zeros(len(t),bool)
    if len(t) < 24:
        return support
    both = valid[:,[11,12,15,16]].all(axis=1) & quiet_cam & np.isfinite(scale)
    eligible = candidate & quiet_cam & (runs_direct != 'running')
    t_run=t[runs_direct=='running']
    dt = float(np.median(np.diff(t)))
    stride = max(1,round(.17/dt))
    hips = pose[:,[11,12],1].mean(axis=1)
    for a,b in runs(eligible):
        if t[b-1]-t[a] < .35:
            continue
        for i in range(a,b,stride):
            lo,hi = np.searchsorted(t,[t[i]-1.12,t[i]+1.12])
            keep=np.flatnonzero(both[lo:hi])+lo
            if len(keep) < max(24,.78*(hi-lo)) or not both[i]:
                continue
            if len(t_run) and np.min(abs(t[i]-t_run)) <= .58:
                continue
            crossings=support_handoffs(t[keep],pose[keep,15:17,:2],hips[keep],np.median(scale[keep]))
            for x,y,_ in crossings:
                if not x <= t[i] <= y:
                    continue
                s,e = np.searchsorted(t,[x-.18*(y-x),y+.18*(y-x)])
                support[max(s,a):min(e,b)] = True
    return support


def motion_above_own_baseline(values, reference, *, allow_lower_tail=False):
    finite = np.isfinite(values)
    quiet = values[finite & reference]
    observed = values[finite]
    if len(observed) < 16:
        return np.zeros(len(values), bool)
    if len(quiet) < 8 and allow_lower_tail:
        quiet = np.sort(observed)[:max(8, len(observed)//4)]
    if len(quiet) < 8:
        return np.zeros(len(values), bool)
    lo, hi = np.quantile(quiet, [.5,.90])
    noise = hi-lo
    return finite & (values > hi+noise)


def continuity(t, direct, compatible, both_feet):
    added = np.zeros(len(t), bool)
    for a,b in runs(compatible & ~direct):
        left = a > 0 and compatible[a-1] and direct[a-1]
        right = b < len(t) and compatible[b] and direct[b]
        if left and right and t[b]-t[a-1] <= 2.15 and both_feet[a:b].mean() >= .55:
            added[a:b] = True
        elif left and t[b-1]-t[a-1] <= .40 and both_feet[a:b].all():
            added[a:b] = True
        elif right and t[b]-t[a] <= .40 and both_feet[a:b].all():
            added[a:b] = True
    return added


def nearby_transition(t, regimes):
    transition = np.zeros(len(t), bool)
    for a,b in runs(regimes == ''):
        if a == 0 or b == len(t) or t[b]-t[a-1] > 1.4:
            continue
        if {regimes[a-1],regimes[b]} == {'gait_walk','running'}:
            transition[a:b] = True
    return transition


def step_switches(time, ankles):
    if len(time) < 12:
        return np.empty(0)
    delta = np.diff(time)
    dt = float(np.median(delta))
    if dt <= 0 or np.max(delta) > 3.1*dt:
        return np.empty(0)
    feet = gaussian_filter1d(ankles.astype(float), max(.75, .05/dt), axis=0, mode='nearest')
    speed = np.linalg.norm(np.gradient(feet, time, axis=0), axis=2)
    contrast = speed[:,0]-speed[:,1]
    resolution = np.quantile(abs(np.diff(contrast)),.75)
    movement = np.quantile(abs(contrast),.8)
    threshold = max(resolution,.25*movement)
    phases = sorted((a,b,sign) for sign in (1,-1) for a,b in runs(sign*contrast > threshold) if b-a >= 3)
    switches = []
    for (a,b,first),(c,d,second) in zip(phases,phases[1:]):
        if first != second:
            switches.append(.5*(time[b-1]+time[c]))
    return np.asarray(switches)


def bracketed_walk_gaps(t, pose, direct, can_bridge, feet_observed):
    filled = np.zeros(len(t), bool)
    for a,b in runs(~direct):
        if a == 0 or b == len(t) or not can_bridge[a:b].all():
            continue
        if not feet_observed[a:b].any():
            continue
        l,r = np.searchsorted(t,[t[a]-.95,t[b]+.95])
        good = np.flatnonzero(feet_observed[l:r])+l
        switches = step_switches(t[good],pose[good,15:17,:2])
        if len(switches) < 2:
            continue
        typical = float(np.median(np.diff(switches)))
        if t[b]-t[a-1] > min(2*typical,t[good[-1]]-t[good[0]]):
            continue
        if not np.any((switches >= t[a]-.5*typical) & (switches <= t[b]+.5*typical)):
            continue
        filled[a:b] = True
    return filled


def advancing_walk(t, pose, valid, scale, allowed, quiet_cam):
    supported = np.zeros(len(t),bool)
    if len(t) < 24:
        return supported
    seen = valid[:,[11,12,15,16]].all(axis=1) & np.isfinite(scale) & quiet_cam
    dt = float(np.median(np.diff(t)))
    stride = max(1,round(.17/dt))
    hip = pose[:,11:13,1].mean(axis=1)
    for a,b in runs(allowed):
        for i in range(a,b,stride):
            lo,hi = np.searchsorted(t,[t[i]-1.12,t[i]+1.12])
            lo,hi = max(a,lo),min(b,hi)
            good = np.flatnonzero(seen[lo:hi])+lo
            if len(good) < max(24,.78*(hi-lo)):
                continue
            pairs = support_handoffs(t[good],pose[good,15:17,:2],hip[good],np.median(scale[good]),progression=True)
            for start,end,step in pairs:
                if not start <= t[i] <= end:
                    continue
                l,r = np.searchsorted(t,[start-.45*step,end+.45*step])
                l,r = max(l,a),min(r,b)
                supported[l:r] |= seen[l:r]
    return supported


def walk_edges(t, pose, valid, direct, eligible):
    inferred = np.zeros(len(t),bool)
    both = valid[:,15:17].all(axis=1)
    for a,b in runs(eligible & ~direct):
        left = a > 0 and direct[a-1]
        right = b < len(t) and direct[b]
        if not (left or right):
            continue
        anchor = a-1 if left else b
        lo,hi = np.searchsorted(t,[t[anchor]-1.2,t[anchor]+1.2])
        good = np.flatnonzero(both[lo:hi])+lo
        switches = step_switches(t[good],pose[good,15:17,:2])
        if not len(switches):
            continue
        step = np.median(np.diff(switches)) if len(switches) > 1 else np.nan
        if not np.isfinite(step):
            continue
        lag = np.abs(t[a:b]-t[anchor])
        nearest = np.min(np.abs(t[a:b,None]-switches[None,:]),axis=1)
        inferred[a:b] |= (lag <= .7*step) & (nearest <= .65*step)
    return inferred
