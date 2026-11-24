"""Recognize a walk-to-run change using this ID's observed gait profiles.

Periods size the observation windows; no speed or cadence separates the classes.
References may come from disjoint fragments of the same accepted track. Never
interpolate poses or extend a label through the gap between fragments.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .activity_rules import runs
from .running_turn import _world_xy

WALK = ('walking', 'gait_walk')
RUN_REFS = ('short_running_rebound', 'running_rhythm_candidate')
WALK_REFS = ('walking_support_handoff', 'walking_advancing_handoffs', 'walking_rhythm_candidate')
REASON = 'same_person_walk_to_run_with_repeated_rebound_and_swing'


def reference(row):
    if row.get('gait_inferred') or row.get('action_hypotheses') or row.get('locomotion_partial'):
        return ''
    kind, evidence = row['activity'], row.get('gait_evidence')
    if evidence == 'person_local_cycle_morphology':
        if row.get('pre_cycle_gait_activity') != kind:
            return ''
        evidence = row.get('pre_cycle_gait_evidence')
    if kind in WALK and evidence in WALK_REFS:
        return 'walk'
    if kind == 'running' and evidence in RUN_REFS:
        return 'run'
    return ''


def observations(t, pose, valid, rows, measured, camera):
    result = np.full((len(t), 3), np.nan)
    periods = np.full(len(t), np.nan)
    eligible = np.asarray(measured, bool) & valid[:, [5,6,11,12,13,14,15,16]].all(axis=1)
    eligible &= np.array([r['activity'] in (*WALK,'running','unknown','body_motion','stepping','fast_stepping')
                          and not (r['action_event_id'] or r.get('stair_sequence_supported') or r.get('locomotion_partial'))
                          for r in rows])
    rates = np.array([r.get('gait_step_hz_candidate') or np.nan for r in rows], float)
    for a,b in runs(eligible):
        time = t[a:b]
        rate = rates[a:b]
        rate = rate[np.isfinite(rate) & (rate > 0)]
        if len(time) < 6 or not len(rate):
            continue
        step = float(np.median(1. / rate))
        dt = np.median(np.diff(time))
        if not (0 < dt < step / 4 and time[-1]-time[0] >= 1.8*step and np.max(np.diff(time)) < 2*dt):
            continue
        xy = pose[a:b,:,:2].astype(float)
        world = _world_xy(t, xy, camera, a, b)
        if world is None:
            continue
        hips, knees, ankles = xy[:,11:13],xy[:,13:15],xy[:,15:17]
        thigh, shank = hips-knees, knees-ankles
        u,v = np.linalg.norm(thigh, axis=2),np.linalg.norm(shank, axis=2)
        leg = gaussian_filter1d(np.median(u+v,axis=1),step/(3*dt),mode='nearest')
        # Knee flexion and vertical ankle separation are geometric evidence, not speed.
        if not np.all((u > 1) & (v > 1) & (leg[:,None] > 1)):
            continue
        flex = 1-np.clip(np.sum(thigh*shank,axis=2)/(u*v),-1,1)
        gap = (ankles[:,0,1]-ankles[:,1,1])/leg
        # Detrend camera-compensated pelvis and shoulder traces separately, then keep only
        # their common oscillation. Remove linear drift before filtering at the edges.
        y = np.c_[world[:,11:13,1].mean(axis=1),world[:,5:7,1].mean(axis=1)]
        x = time-time.mean()
        y -= x[:,None]*(x @ y)/(x @ x)
        sig = (gaussian_filter1d(y,step/(11*dt),axis=0,mode='reflect')-
               gaussian_filter1d(y,step/(2*dt),axis=0,mode='reflect')) / leg[:,None]
        both = np.where(sig.prod(axis=1)>0,np.sign(sig[:,0])*np.min(np.abs(sig),axis=1),0.)
        # Use centered two-step support, truncated at visible edges; never pad with another fragment.
        starts = np.searchsorted(time, time-step)
        stops = np.searchsorted(time, time+step, side='right')
        for i,(l,r) in enumerate(zip(starts,stops)):
            if time[r-1]-time[l] < step or r-l<6:
                continue
            lo,hi = np.quantile(gap[l:r],[.1,.9])
            if not lo < 0 < hi: # both opposite strides must be visible
                continue
            bottom,top = np.quantile(both[l:r],[.15,.85])
            fold = np.median(np.quantile(flex[l:r],.85,axis=0)-np.quantile(flex[l:r],.15,axis=0))
            result[a+i] = (top-bottom, hi-lo, fold)
            periods[a+i] = step
    return result, periods


def sample_references(t, rows, values, step):
    types = np.array([reference(r) for r in rows],object)
    available = np.isfinite(values).all(axis=1)
    out = {'walk':[], 'run':[]}
    for kind in out:
        for a,b in runs(available & (types==kind)):
            last = -np.inf
            # A whole observation window must be similarly labeled to seed the profile.
            for i in range(a,b):
                if min(t[i]-t[a],t[b-1]-t[i]) >= step[i] and t[i]-last >= step[i]:
                    out[kind].append(values[i]); last=t[i]
    return out


def profile(refs):
    if min(len(refs['run']),len(refs['walk'])) < 3:
        return None
    walk,run = [np.quantile(refs[k],[.25,.5,.75],axis=0) for k in ('walk','run')]
    return walk,run


def apply(t, rows, values, step, profiles):
    if profiles is None:
        return
    walk,run = profiles
    # Keep only features with a reliable person-local contrast. Shared torso rebound
    # is mandatory, and must be corroborated by at least one independent leg feature.
    contrast = run[1] > walk[2]
    if not contrast[0] or not contrast[1:].any():
        return
    midpoint = np.maximum(walk[2], (walk[1]+run[1])/2)
    above = (values > midpoint) & contrast
    shape = above[:,0] & above[:,1:].any(axis=1)
    is_walk = np.array([r['activity'] in WALK and not (r['action_event_id'] or r.get('locomotion_partial')) for r in rows])
    for a,b in runs(is_walk):
        for l,r in runs(shape[a:b]):
            l,r = l+a,r+a
            # This rule resolves a switch: require a prior, observed walk in the same
            # fragment and a new morphology that lasts through repeated local steps.
            if l==a or not np.isfinite(step[l]) or t[l]-t[a] < step[l]:
                continue
            if r-l < 2 or t[r-1]-t[l]+np.median(np.diff(t[l:r])) < 2*step[l]:
                continue
            for i in range(l,r):
                row = rows[i]
                row.update(pre_gait_switch_activity=row['activity'], pre_gait_switch_evidence=row.get('gait_evidence',''),
                           activity='running',gait_evidence='person_local_gait_switch',activity_reason=REASON,
                           action_hypotheses=(), ambiguous_gait=0,gait_inferred=0,
                           personal_gait_step_period_s=float(step[i]),
                           gait_switch_shared_rebound=float(values[i,0]),gait_switch_ankle_excursion=float(values[i,1]),
                           gait_switch_knee_fold=float(values[i,2]))
