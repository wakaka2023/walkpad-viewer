import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from activity_rules import runs


def sequence_candidates(t, stride, rebound, *, prominence=.104):
    dt = float(np.median(np.diff(t)))
    peak = lambda sig, p, sec: find_peaks(sig,prominence=p,distance=max(1,round(sec/dt)))[0]
    foot = sorted([(i,1) for i in peak(stride,prominence,.27)] + [(i,-1) for i in peak(-stride,prominence,.27)])
    steps = []
    for i, direction in foot:
        if steps and t[i] - t[steps[-1][0]] < .17:
            if abs(stride[i]) > abs(stride[steps[-1][0]]):
                steps[-1] = (i,direction)
        else:
            steps.append((i,direction))
    bounce = peak(rebound,.078,.23)
    bounce = bounce[rebound[bounce] >= .036]
    return steps, bounce


def detect(t, pose, valid, eligible, scale, *, step_batches=None):
    t = np.asarray(t, float)
    labels = np.full(len(t), False)
    slow = np.full(len(t), False)
    cadence = np.full(len(t), np.nan)
    amplitude = cadence.copy()
    if len(t)<20:
        return labels, slow, cadence, amplitude
    dt = float(np.median(np.diff(t)))
    if not 0 < dt <= 1/18:
        return labels, slow, cadence, amplitude
    xy = pose[:,:,:2]
    good = (eligible & np.isfinite(scale) & (scale >= 12) &
            valid[:,[5,6,11,12,15,16]].all(axis=1))
    bridge = good.copy()
    for a,b in runs(~good & eligible):
        if a and b<len(t) and good[a-1] and good[b] and t[b]-t[a-1] <= 2.25*dt:
            bridge[a:b]=True
    torso = np.mean(xy[:,[5,6,11,12]],axis=1)
    arms = np.zeros(len(t),bool)
    for shoulder, elbow, wrist in ((5,7,9),(6,8,10)):
        u, v = xy[:,shoulder]-xy[:,elbow],xy[:,wrist]-xy[:,elbow]
        norm = np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)
        cosine = np.sum(u*v,axis=1)/np.maximum(1.,norm)
        arms |= valid[:,[shoulder,elbow,wrist]].all(axis=1) & (cosine > np.cos(np.deg2rad(114.)))
    for a,b in runs(bridge):
        src = np.flatnonzero(good[a:b])+a
        if len(src)<19 or t[b-1]-t[a]<.87 or np.max(np.diff(t[a:b]))>2.2*dt:
            continue
        grid = np.arange(t[a],t[b-1]+.1*dt,dt)
        uniform = lambda x: np.interp(grid,t[src],x[src])
        smooth = lambda v, s: gaussian_filter1d(v,s/dt,mode='nearest')
        unit = smooth(uniform(scale),.16)
        ax = uniform(xy[:,15,1]-xy[:,16,1])/unit
        y = uniform(torso[:,1])
        spread = smooth(ax,.037)-smooth(ax,.25)
        rebound = (smooth(y,.037)-smooth(y,.25))/unit
        steps,bounces = sequence_candidates(grid,spread,rebound)
        if step_batches is not None:
            step_batches.append((grid, steps))
        chains, chain = [], []
        for i,direction in steps:
            other = bounces[abs(grid[bounces]-grid[i])<=.143]
            if not len(other):
                if chain: chains.append(chain)
                chain=[]
                continue
            match = int(other[np.argmin(abs(grid[other]-grid[i]))])
            if chain:
                prev, sign, previous_bounce = chain[-1]
                interval = grid[i]-grid[prev]
                if (direction == sign or not .205 <= interval <= .495 or match <= previous_bounce
                        or not .21 <= grid[match]-grid[previous_bounce] <= .53):
                    chains.append(chain)
                    chain=[]
            chain.append((i,direction,match))
        if chain:
            chains.append(chain)
        here = (np.arange(len(t)) >= a) & (np.arange(len(t)) < b)
        for q in chains:
            if len(q)<4:
                continue
            steps_i, _, rebound_i = np.asarray(q).T
            gaps = np.diff(grid[steps_i]); period = float(np.median(gaps))
            if np.max(abs(gaps-period))>.105 or period>.455:
                continue
            peak_amp = float(np.median(rebound[rebound_i]))
            l, r = grid[steps_i[0]]-.50*period, grid[steps_i[-1]]+.53*period
            target = good & here & (t>=l) & (t<=r)
            n = np.flatnonzero(target)
            if len(n)<5:
                continue
            shift = np.linalg.norm(np.median(torso[n[-3:]],axis=0)-np.median(torso[n[:3]],axis=0))
            shift /= np.median(scale[n]) * max(period,t[n[-1]]-t[n[0]])
            if shift < .21 or not (peak_amp>=.059 or (peak_amp>=.046 and arms[target].mean()>=.21)):
                continue
            labels[target]=True
            amplitude[target]=peak_amp
            cadence[target]=1./period
        for q in range(len(steps)-2):
            k = steps[q:q+3]
            ii, sign = np.asarray(k).T
            gaps = np.diff(grid[ii])
            if sign[0] == sign[1] or sign[1] == sign[2] or not np.all((gaps>=.48)&(gaps<=.86)):
                continue
            gg = (grid>=grid[ii[0]]-.08)&(grid<=grid[ii[-1]]+.07)
            target = good & here & (t>=grid[ii[0]]-.08)&(t<=grid[ii[-1]]+.07)
            if target.any() and np.quantile(abs(rebound[gg]),.80)<.039 and arms[target].mean()<.23:
                slow[target] = True
    return labels, slow, cadence, amplitude
