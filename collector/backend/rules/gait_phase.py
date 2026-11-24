import numpy as np
from scipy.signal import detrend, find_peaks

from .action_rules import clean_mask
from .activity_rules import runs


def dominant_hz(values, dt, lo, hi):
    grid = np.fft.rfftfreq(512, dt)
    search = (grid >= lo) & (grid <= hi)
    power = np.abs(np.fft.rfft(values * np.hanning(len(values)), 512))**2
    if not search.any() or power.sum() < 1e-6:
        return np.nan, 0.
    k = np.flatnonzero(search)[np.argmax(power[search])]
    return grid[k], float(power[np.abs(grid-grid[k]) < .25].sum()/power.sum())


def window_evidence(time, points, scale):
    t = np.asarray(time, float)
    if len(t) < 30 or t[-1]-t[0] < 1.65 or not np.all(np.isfinite(points)) or not np.isfinite(scale) or scale < 8.:
        return None
    t = t - t[0]
    dt = np.median(np.diff(t))
    if dt <= 0 or np.max(np.diff(t)) > 2.25*dt or dt > 1/18.:
        return None
    grid = np.arange(0., t[-1]+dt*.1, dt)
    values = np.column_stack([np.interp(grid, t, s) for s in points.T])
    ankle = detrend(values[:, :2], axis=0)
    _, strengths, basis = np.linalg.svd(ankle, full_matrices=False)
    if strengths[0] <= 1e-3 or strengths[1] > .53*strengths[0]:
        return None
    spread = ankle @ basis[0]
    amp = np.quantile(spread, .92)-np.quantile(spread, .08)
    if amp/scale < .2:
        return None
    stride, leg_power = dominant_hz(spread, dt, .42, 2.20)
    if not np.isfinite(stride) or leg_power < .51 or 1/(stride*dt) < 11:
        return None
    prominence = amp*.37
    minimum = max(1, round(.58/(stride*dt)))
    positive = find_peaks(spread, prominence=prominence, distance=minimum)[0]
    negative = find_peaks(-spread, prominence=prominence, distance=minimum)[0]
    order = sorted([(p,1) for p in positive] + [(p,-1) for p in negative])
    if len(order)<4 or any(l[1]==r[1] for l,r in zip(order,order[1:])):
        return None
    torso = detrend(values[:, 2:],axis=0)
    close = np.abs(spread) <= np.quantile(np.abs(spread),.29)
    left = spread >= np.quantile(spread,.80)
    right = spread <= np.quantile(spread,.20)
    phase, rhythm = [], [leg_power]
    for vertical in torso.T:
        hz, power = dominant_hz(vertical,dt,max(.6,stride*2-.55),stride*2+.55)
        width = float(np.quantile(vertical,.92)-np.quantile(vertical,.08))
        if not np.isfinite(hz) or abs(hz-2*stride) > max(.28,.13*2*stride) or power < .34 or width < .025*scale:
            return None
        rhythm.append(power)
        mid = np.median(vertical[close])
        phase.extend((mid-np.median(vertical[left]),mid-np.median(vertical[right])))
    phase = np.asarray(phase)/scale
    if np.all(phase >= .016):
        label = 'running'
    elif np.all(phase <= -.016):
        label = 'gait_walk'
    else:
        label = ''
    walk_support = np.any(np.all(phase.reshape(2,2) <= -.016, axis=1))
    run_support = np.any(np.all(phase.reshape(2,2) >= .016, axis=1))
    return dict(activity=label, walking_support=float(walk_support), running_support=float(run_support), inferred_step_hz=float(2*stride), rhythm_agreement=float(min(rhythm)),
                midstance_phase_signed=float(np.median(phase)), phase_min_amplitude=float(np.min(abs(phase))),
                alternating_extrema=len(order), method='projected_ankle_crossing_and_hip_shoulder_phase')


def classify_windows(t, pose, valid, scale, eligible, camera_quiet, overground=None, translation=None):
    if overground is None:
        overground = np.zeros(len(t), bool)
    if translation is None:
        translation = np.full(len(t), np.nan)
    ground_mode = np.zeros(len(t), bool)
    profile_mode = np.zeros(len(t), bool)
    stats = np.full((len(t), 3), np.nan)
    result = np.full(len(t),'',object)
    names = ('inferred_step_hz','rhythm_agreement','midstance_phase_signed','phase_min_amplitude','walking_support','running_support')
    evidence = {key: np.full(len(t),np.nan) for key in names}
    required = [5,6,11,12,15,16]
    observed = valid[:,required].all(axis=1) & camera_quiet & np.isfinite(scale)
    foot = pose[:,15,:2]-pose[:,16,:2]
    hips = pose[:,[11,12],1].mean(axis=1)
    shoulders = pose[:,[5,6],1].mean(axis=1)
    series = np.column_stack([foot, hips, shoulders])
    step = max(1, round(.13/np.median(np.diff(t))))
    ground_bounds = np.zeros((len(t),2), int)
    for a,b in runs(overground & eligible):
        ground_bounds[a:b] = (a,b)
    for a,b in runs(eligible):
        positions = np.arange(a,b,step)
        bounds = np.r_[a,(positions[1:]+positions[:-1]+1)//2,b]
        for j,i in enumerate(positions):
            if not observed[i]:
                continue
            lo, hi = np.searchsorted(t, [t[i]-1.55,t[i]+1.55])
            lo,hi = max(lo,a),min(hi,b)
            if hi-lo > 1 and t[hi-1]-t[lo] < 2.45:
                if lo == a:
                    hi = min(b, int(np.searchsorted(t, t[a]+2.45)))
                elif hi == b:
                    lo = max(a, int(np.searchsorted(t, t[b-1]-2.45)))
            direct = np.flatnonzero(observed[lo:hi])+lo
            row = None
            if len(direct) >= .90*(hi-lo):
                row = window_evidence(t[direct], series[direct], np.median(scale[direct]))
            start,stop = bounds[j:j+2]
            if overground[i]:
                left,right = ground_bounds[i]
                l,r = max(lo,left),min(hi,right)
                if r-l > 1 and t[r-1]-t[l] < 2.35:
                    if l == left:
                        r = min(right, int(np.searchsorted(t,t[left]+2.35)))
                    elif r == right:
                        l = max(left, int(np.searchsorted(t,t[right-1]-2.35)))
                near = np.flatnonzero(observed[l:r])+l
                if len(near) >= .75*(r-l) and len(near) >= 45:
                    extra = overground_window(t[near], series[near], pose[near,15:17,:2], np.median(scale[near]))
                    if extra is not None:
                        stats[start:stop] = [np.median(translation[near]), extra['inferred_step_hz'], extra['hip_oscillation']]
                        if row is None or not row['activity']:
                            row = extra
                            ground_mode[start:stop] = bool(extra['activity'])
            if row is None:
                continue
            result[start:stop] = row['activity']
            for name in names:
                evidence[name][start:stop] = row[name]
    apply_person_profile(t, stats, observed & overground, result, profile_mode, evidence)
    find_distinct_bouncing_gait(t, stats, observed & overground, result, profile_mode, evidence)
    result[~observed] = ''
    for name in ('walking_support','running_support'):
        evidence[name][~observed] = np.nan
    evidence['inferred'] = profile_mode.copy()
    for a,b in runs(result==''):
        if a == 0 or b == len(t) or not (observed[a:b] & eligible[a:b]).all():
            continue
        kind = result[a-1]
        if not kind or kind != result[b] or t[b]-t[a] > .44:
            continue
        same_direction = 1 if kind=='running' else -1
        opposite = 'walking_support' if kind == 'running' else 'running_support'
        if np.any(same_direction*evidence['midstance_phase_signed'][a:b] < -.016) or np.any(evidence[opposite][a:b] == 1):
            continue
        result[a:b] = kind
        evidence['inferred'][a:b] = True
    report = []
    for label in ('gait_walk','running'):
        supported = clean_mask(t,result==label,(result==label),.75,0.)
        result[(result==label)&~supported] = ''
        for a,b in runs(supported):
            end = float(t[b]) if b < len(t) else float(t[-1]+np.median(np.diff(t)))
            report.append(dict(activity=label,start_video_s=float(t[a]),end_video_s=end,
                               median_step_hz=float(np.nanmedian(evidence['inferred_step_hz'][a:b])),
                               signed_phase_median=float(np.nanmedian(evidence['midstance_phase_signed'][a:b])),
                               inferred_frames=int(evidence['inferred'][a:b].sum()),
                               basis=('same_person_three_cue_profile' if profile_mode[a:b].any() else
                                      'projected_overground_or_contact_style_phase' if ground_mode[a:b].any() else
                                      'projected_phase_not_direct_foot_contact')))
    evidence['overground_mode'] = ground_mode
    evidence['profile_mode'] = profile_mode
    evidence['hip_oscillation'] = stats[:, 2]
    return result,evidence,sorted(report,key=lambda row:row['start_video_s'])


def overground_window(time, points, ankles, scale):
    from scipy.ndimage import gaussian_filter1d

    if len(time) < 45 or time[-1]-time[0] < 1.60 or not np.isfinite(scale) or scale < 8.:
        return None
    dt = float(np.median(np.diff(time)))
    if dt <= 0 or np.max(np.diff(time)) > 3.2*dt or dt > 1/18:
        return None
    grid = np.arange(time[0], time[-1]+.1*dt, dt)
    values = np.column_stack([np.interp(grid, time, v) for v in points.T])
    spread_xy = detrend(values[:, :2], axis=0)
    _, singular, axes = np.linalg.svd(spread_xy, full_matrices=False)
    if singular[0] <= 1e-6 or singular[1] > .90*singular[0]:
        return None
    spread = spread_xy @ axes[0]
    amp = np.quantile(spread, .92) - np.quantile(spread, .08)
    stride, leg_power = dominant_hz(spread, dt, .42, 2.20)
    if not np.isfinite(stride) or leg_power < .48 or amp < .23*scale:
        return None
    distance = max(1, round(.52/(stride*dt)))
    highs = find_peaks(spread, prominence=.28*amp, distance=distance)[0]
    lows = find_peaks(-spread, prominence=.28*amp, distance=distance)[0]
    alternation = sorted([(k, 1) for k in highs] + [(k, -1) for k in lows])
    if len(alternation) < 3 or any(x[1] == y[1] for x,y in zip(alternation, alternation[1:])):
        return None
    torso = values[:, 2:]
    torso -= gaussian_filter1d(torso, max(1., .44/(stride*dt)), axis=0, mode='nearest')
    close = abs(spread) <= np.quantile(abs(spread), .30)
    wide = (spread >= np.quantile(spread, .80), spread <= np.quantile(spread, .20))
    signed, quality = [], []
    for signal in torso.T:
        freq, power = dominant_hz(signal, dt, max(.55, 2*stride-.55), 2*stride+.55)
        phase = np.array([np.median(signal[close])-np.median(signal[q]) for q in wide])/scale
        quality.append(power if abs(freq-2*stride) <= .25*stride+.15 else 0.)
        signed.append(phase)
    signed = np.array(signed)
    noise = np.median(abs(np.diff(torso, axis=0)), axis=0)/scale
    strongest = int(np.argmax(quality))
    if quality[strongest] < .37:
        return None
    velocity = np.column_stack([np.interp(grid,time,v) for v in ankles.reshape(-1,4).T]).reshape(-1,2,2)
    velocity = np.linalg.norm(np.gradient(gaussian_filter1d(velocity, .047/dt, axis=0),dt,axis=0),axis=2)
    quiet, swing = np.quantile(velocity,[.18,.88])
    margins = []
    for cutoff in (.18, .23, .28):
        still = velocity <= quiet+cutoff*(swing-quiet)
        margins.append(float((~still).all(axis=1).mean() - still.all(axis=1).mean()))
    minimum = max(1.45*noise[strongest],np.finfo(float).eps)
    phase = signed[strongest]
    other = signed[1-strongest]
    walk = (phase < -minimum).all() and np.median(other) < 0
    run = (phase > minimum).all() and np.median(other) > 0
    run |= (np.median(phase) > 0 and (phase > minimum).any() and margins[1] > max(.03,-margins[2])
            and np.median(other) > -noise[1-strongest])
    return dict(activity='running' if run else 'gait_walk' if walk else '', walking_support=float(walk), running_support=float(run),
                inferred_step_hz=float(2*stride), rhythm_agreement=float(min(quality[strongest],leg_power)),
                midstance_phase_signed=float(np.median(phase)), phase_min_amplitude=float(np.min(abs(phase))),
                hip_oscillation=float((np.quantile(torso[:,0],.90)-np.quantile(torso[:,0],.10))/scale),
                alternating_extrema=len(alternation), method='overground_detrended_ankle_torso_phase')


def apply_person_profile(t, measurements, eligible, result, filled, evidence):
    dt = np.median(np.diff(t))
    useful = eligible & np.isfinite(measurements).all(axis=1) & (measurements > 0).all(axis=1)
    codes = ('gait_walk', 'running')
    selected = [useful & (result == k) for k in codes]
    if min(np.count_nonzero(s) for s in selected)*dt < .52:
        return
    values = np.log(np.maximum(measurements,np.finfo(float).tiny))
    centers = np.array([np.median(values[s],axis=0) for s in selected])
    widths = np.array([np.quantile(values[s],.80,axis=0)-np.quantile(values[s],.20,axis=0) for s in selected])
    delta = centers[1]-centers[0]
    separated = delta > np.maximum(np.max(widths,axis=0)*.7, np.finfo(float).eps)
    if separated.sum() < 2:
        return
    scores = (values[:,separated]-centers[0,separated])/delta[separated]
    standouts = np.nan_to_num(evidence['rhythm_agreement']) > 0
    for j,code in enumerate(codes):
        own = scores if j else 1-scores
        want = (np.median(own,axis=1) > .64) & ((own > .48).sum(axis=1) >= 2)
        can = useful & standouts & (result == '') & want
        opposition = 'walking_support' if j else 'running_support'
        can &= evidence[opposition] != 1
        for a,b in runs(can):
            if t[b-1]-t[a]+dt < .45:
                continue
            result[a:b] = code
            filled[a:b] = True


def find_distinct_bouncing_gait(t, measurements, eligible, result, filled, evidence):
    dt = float(np.median(np.diff(t)))
    good = eligible & np.isfinite(measurements).all(axis=1) & (measurements > 0).all(axis=1)
    walk = good & (result == 'gait_walk') & ~filled
    if np.count_nonzero(walk)*dt < 2.0:
        return
    values = np.log(np.maximum(measurements,np.finfo(float).tiny))
    center = np.median(values[walk],axis=0)
    spread = np.median(abs(values[walk]-center),axis=0)
    stride_noise = np.median(np.abs(np.diff(values[walk,1])))
    hip_noise = np.median(np.abs(np.diff(values[walk,2])))
    variation = np.maximum(spread, [0,2*stride_noise,2*hip_noise])
    advantage = values-center
    pattern = (good & (advantage[:,0] >= -spread[0])
               & (advantage[:,1] > 3.2*variation[1]) & (advantage[:,2] > 3.2*variation[2]))
    for a,b in runs(pattern):
        if t[b-1]-t[a]+dt < 1.3 or np.mean(result[a:b]=='gait_walk') > .65:
            continue
        cadence = measurements[a:b,1]
        if np.quantile(cadence,.8)-np.quantile(cadence,.2) > .28*np.median(cadence):
            continue
        reliability = abs(evidence['midstance_phase_signed'][a:b])
        conflict = ((result[a:b] == 'gait_walk') & (reliability > np.nanmedian(reliability)))
        indices = np.arange(a,b)[~conflict & (result[a:b] != 'running')]
        result[indices] = 'running'
        filled[indices] = True
