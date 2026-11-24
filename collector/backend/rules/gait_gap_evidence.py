"""Bridge a weak motion island only while its observed local gait continues."""
import numpy as np
from scipy.ndimage import median_filter
from .activity_rules import runs
from .running_turn import _world_xy
from .walk_continuity import _step

REASON = 'measured_bilateral_cycle_continues_between_same_gait_anchors'
GAIT = ('walking','running')
UNRESOLVED = ('','gait_type_unresolved','moving_partial_gait_unresolved',
              'translation_with_step_motion_but_unresolved_gait')


def _alternation(t, signal, left, right, a, b, step, noise):
    # Fit both real flanks, not the gap. Time sets the local oscillator phase;
    # the established flank class sets the label, never a common cadence cut.
    anchor = np.r_[left,right]
    n = len(anchor)
    if n < 9:
        return False
    u = t-t[a]
    span = t[right[-1]]-t[left[0]]
    if span < 2*step:
        return False
    # Allow local variation within one frequency bin of the sampled anchor span.
    f0 = .5/step
    bandwidth = min(f0/3, .5/span)
    best = None
    for f in np.linspace(f0-bandwidth,f0+bandwidth,11):
        phase = 2*np.pi*f*u
        x = np.c_[np.ones(len(u)),u/span,np.cos(phase),np.sin(phase)]
        model = x @ np.linalg.lstsq(x[anchor],signal[anchor],rcond=None)[0]
        err = np.sum((model[anchor]-signal[anchor])**2)
        if best is None or err < best[0]:
            best = err, model, x
    error, pred, x = best
    trend = x[:,:2] @ np.linalg.lstsq(x[anchor,:2],signal[anchor],rcond=None)[0]
    baseline = np.sum((trend[anchor]-signal[anchor])**2)
    floor = noise**2*n
    if n*np.log(max(baseline,floor)/max(error,floor)) <= 5*np.log(n):
        return False
    amp = np.sqrt(np.mean(np.sum((pred[anchor]-trend[anchor])**2,axis=1)))
    if amp <= 3*noise:
        return False
    tolerance = max(3*noise, np.sqrt(error/n)*2, amp/2)
    # Only award a continuous interval when the gap itself is consistent with
    # the phase, including each local step; an actual pause must remain visible.
    divisions = np.linspace(t[a],t[b],max(1,round((t[b]-t[a])/step))+1)
    for lo, hi in zip(divisions[:-1],divisions[1:]):
        s,e = np.searchsorted(t,(lo,hi))
        residual = signal[s:e]-pred[s:e]
        if not len(residual) or np.sqrt(np.mean(np.sum(residual**2,axis=1))) > tolerance:
            return False
        if e-s>=3:
            expected = np.linalg.norm(np.ptp(pred[s:e],axis=0))
            observed = np.linalg.norm(np.ptp(signal[s:e],axis=0))
            if expected>3*noise and observed<expected/2:
                return False
    return True


def bridges(t, ends, labels, rows, inferred, protected, reviews, measured, pose, valid, camera):
    result = np.full(len(t), '',object)
    usable = measured & ~protected & valid[:,[5,6,11,12]].all(axis=1) & np.array([
        not (r['action_event_id'] or r.get('stair_sequence_supported')) for r in rows],bool)
    steps = np.array([_step(r) for r in rows])
    dt = ends-t
    weak = np.array([r['activity'] in ('unknown','body_motion','stepping','fast_stepping')
                     and r.get('gait_evidence','') in UNRESOLVED and not r.get('action_hypotheses')
                     and rv['activity_status']=='other' for r,rv in zip(rows,reviews)],bool) & (labels=='unknown')
    for kind in GAIT:
        same = (labels==kind) & np.array([rv['action_candidates'] in ('',kind) for rv in reviews],bool)
        anchor = same & ~inferred & np.array([rv['activity_status']=='supported' for rv in reviews],bool)
        for s,e in runs(usable & (same | weak)):
            if not weak[s:e].any():
                continue
            flanks = []
            for a,b in runs(anchor[s:e]):
                a,b = s+a,s+b
                p = steps[a:b]
                if not np.isfinite(p).any():
                    continue
                period = float(np.nanmedian(p))
                if ends[b-1]-t[a] >= period/2:
                    flanks.append((a,b,period))
            for (l,a,p0),(b,r,p1) in zip(flanks[:-1],flanks[1:]):
                if a==b or not weak[a:b].any():
                    continue
                step = np.sqrt(p0*p1)
                if t[b]-t[a]>4*min(p0,p1):
                    continue
                i=max(l,np.searchsorted(t,t[a]-2*p0))
                j=min(r,np.searchsorted(t,t[b]+2*p1,side='right'))
                if min(a-i,j-b)<3 or np.max(dt[i:j])>step/2:
                    continue
                for feet in ([15,16],[13,14]):
                    if not valid[i:j,feet].all():
                        continue
                    xy = _world_xy(t,pose[i:j,:, :2],camera,i,j)
                    if xy is None:
                        break
                    size = np.linalg.norm(xy[:,5:7].mean(1)-xy[:,11:13].mean(1),axis=1)
                    size = median_filter(size,size=max(1,round(step/np.median(dt[i:j]))|1),mode='nearest')
                    if np.any(size <= 1):
                        break
                    signal = (xy[:,feet[0]]-xy[:,feet[1]])/size[:,None]
                    noise = 1/np.median(size)
                    if _alternation(t[i:j],signal,np.arange(a-i),np.arange(b-i,j-i),a-i,b-i,step,noise):
                        result[np.flatnonzero(weak[a:b])+a] = kind
                        break
    return result
