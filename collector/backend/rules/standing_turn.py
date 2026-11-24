"""An in-place turn must rotate the trunk AND reposition feet, not merely look away."""
import numpy as np
from scipy.ndimage import median_filter
from numpy.lib.stride_tricks import sliding_window_view

from .activity_rules import runs
from .running_turn import _world_xy

TURN = 'standing_turn'
EVIDENCE = 'same_track_shoulders_hips_rotate_with_local_foot_reposition'


def _synchronous_evidence(t, pose, valid, rows, measured, camera, smooth_s):
    joints = [5, 6, 11, 12, 15, 16]
    eligible = measured & valid[:, joints].all(axis=1) & np.array([
        r['activity'] in ('standing','between_reps','unknown','body_motion','arm_activity')
        and not (r['action_event_id'] or r.get('stair_sequence_supported') or r.get('action_hypotheses'))
        for r in rows], bool)
    if len(t) < 3:
        return
    dt = np.median(np.diff(t))
    radius = max(2, round(smooth_s / (2*dt)))
    for s, e in runs(eligible):
        if e-s < 5*radius:
            continue
        p = _world_xy(t, pose[s:e,joints,:2], camera, s, e)
        if p is None:
            continue
        hips, shoulders = p[:, 2:4].mean(1), p[:, :2].mean(1)
        unit = np.linalg.norm(hips-shoulders,axis=1)
        if np.any(unit <= 1):
            continue
        # Signed widths retain a left/right swap during a turn past the front view.
        widths = np.c_[p[:,1,0]-p[:,0,0], p[:,3,0]-p[:,2,0]] / unit[:,None]
        sm = median_filter(widths, size=(2*radius+1,1), mode='nearest')
        resolution = 1 / np.median(unit)
        noise = np.maximum(np.median(abs(widths-sm),axis=0)*1.4826, resolution)
        change = np.zeros_like(sm)
        change[radius:-radius] = sm[2*radius:] - sm[:-2*radius]
        local = np.maximum(3*noise, 2*np.median(abs(change[radius:-radius]),axis=0))
        for direction in (-1,1):
            aligned = change*direction
            motion = np.all(aligned > local,axis=1)
            # Join measurement-sized dips only where the pair continues in the same direction.
            for a,b in runs(~motion):
                if a and b<len(motion) and b-a<=radius and np.all(aligned[a:b] >= -noise):
                    motion[a:b] = True
            for a,b in runs(motion):
                if a < 2*radius or len(sm)-b < 2*radius or b-a<radius:
                    continue
                left,right = slice(a-2*radius,a-radius),slice(b+radius,b+2*radius)
                before,after = np.median(sm[left],axis=0),np.median(sm[right],axis=0)
                quiet = np.maximum(np.ptp(sm[left],axis=0),np.ptp(sm[right],axis=0))
                swing = direction*(after-before)
                # A wide-foot shift or slight sway must not become body yaw
                # just because the detector reports low jitter. Use this
                # person's projected thigh/leg and stable torso widths as the
                # scale of a meaningful change, including near-profile views.
                observed=pose[s+a-2*radius:s+b+2*radius]
                ok=valid[s+a-2*radius:s+b+2*radius]
                thigh=np.linalg.norm(observed[:,13:15,:2]-observed[:,11:13,:2],axis=2)
                usable=ok[:,13:15] & (thigh>2) & np.isfinite(thigh)
                if usable.any():
                    leg=float(np.median(thigh[usable]))
                else:
                    leg=float(np.median(np.linalg.norm(p[a:b,4:]-p[a:b,2:4],axis=2)))/2
                size_floor=leg/(4*np.median(unit[a:b]))
                scale_floor=np.maximum(size_floor,np.maximum(abs(before),abs(after))/3)
                if np.any(swing <= np.maximum(np.maximum(5*noise, 2*quiet),scale_floor)):
                    continue
                base,final = np.median(p[left,4:],axis=0), np.median(p[right,4:],axis=0)
                step_lengths = np.linalg.norm(final-base,axis=1)
                step = np.max(step_lengths)
                foot_noise = max(resolution*np.median(unit), np.linalg.norm(p[left,4:]-base,axis=2).max(),
                                 np.linalg.norm(p[right,4:]-final,axis=2).max())
                center_travel = np.linalg.norm(hips[a-radius:b+radius]-np.median(hips[left],axis=0),axis=1).max()
                foot_reset = np.linalg.norm((final[0]-final[1])-(base[0]-base[1]))
                if min(step,foot_reset) <= 3*foot_noise or center_travel >= step:
                    continue
                # Expand only while both smoothed joint widths are between the end states.
                pre = np.all(direction*(sm-before) > noise,axis=1)
                post = np.all(direction*(after-sm) > noise,axis=1)
                start,end=a,b
                while start > a-radius and pre[start-1]: start-=1
                while end < b+radius and post[end]: end+=1
                for row in rows[s+start:s+end]:
                    row.update(pre_pivot_activity=row['activity'], activity=TURN,
                               activity_reason=EVIDENCE, action_hypotheses=(),
                               ambiguous_gait=0, gait_inferred=0, locomotion_partial=0)


STAGGERED = 'observed_person_local_shoulder_hip_and_foot_reorientation'
JOINTS = [5, 6, 11, 12, 15, 16]
PROTECTED = frozenset({'squat', 'split_squat', 'stairs_up', 'stairs_down', 'running_turn', 'moving_turn'})


def _observed_spans(t, pose, usable, camera):
    """Divide only at real observation, event, or camera-validity boundaries."""
    ids = np.asarray(camera['source_indices'], int)
    cuts = np.r_[0, np.flatnonzero(~np.asarray(camera['valid'], bool)[1:])+1, len(ids)]
    for i, j in zip(cuts[:-1],cuts[1:]):
        if j-i < 2:
            continue
        l, r = max(0, ids[i]), min(len(t),ids[j-1]+1)
        for a,b in runs(usable[l:r]):
            a,b = l+a,l+b
            if b-a < 9:
                continue
            px = _world_xy(t,pose[a:b,JOINTS,:2],camera,a,b)
            if px is not None:
                yield a,b,px


def _transition(z, band, t, margin):
    """Find each channel's own departure/arrival; these need not overlap."""
    n=len(t)
    inside = np.flatnonzero((z>band)&(z<1-band))
    inside = inside[(inside>=margin-1)&(inside<n-margin+1)]
    if len(inside)<2:
        return None  # a single pose swap is not a measured transition
    a,b=inside[0],inside[-1]+1
    if (z[:a]>1-band).any() or (z[b:]<band).any():
        return None  # a completed reversal back to the old state
    core=z[a:b]
    if np.max(np.maximum.accumulate(core)-core)>2*band:
        return None  # oscillation is not a stable-to-stable change
    ramp=np.clip((t-t[a-1])/(t[b]-t[a-1]),0,1)
    trend=np.c_[np.ones(n),t-t.mean()]
    drift=trend@np.linalg.lstsq(trend,z,rcond=None)[0]
    residual=np.mean((ramp-z)**2)
    if residual>=np.mean((drift-z)**2):
        return None
    return a,b,residual


def _staggered_windows(t,p):
    shoulder, hip = p[:,:2].mean(1),p[:,2:4].mean(1)
    foot=p[:,4:]
    trunk=hip-shoulder
    unit=np.linalg.norm(trunk,axis=1)
    if np.any(unit<=1):
        return []
    axis=trunk[:,[1,0]]*[1,-1]/unit[:,None]
    spans=np.stack((p[:,1]-p[:,0],p[:,3]-p[:,2]),axis=1)
    widths=median_filter(np.einsum('nki,ni->nk',spans,axis)/unit[:,None],(3,1),mode='nearest')
    # Near-profile landmark drift can look like a change of sign. Keep a
    # reference from the observed orientations of this same person, rather
    # than treating every statistically visible change as a body turn.
    orientation_scale=np.median(abs(widths),axis=0)
    suggestions=[]
    margin=3  # samples needed to estimate a local endpoint, not a motion threshold
    while 3*margin<=len(t):
        windows=sliding_window_view(widths,margin,axis=0)
        centers=np.median(windows,axis=-1)
        scatter=np.max(abs(windows-centers[:,:,None]),axis=-1)
        # Stable endpoints need not last as long as the turn. Search several
        # lengths measured in samples; the sequence, not a preset time, sets
        # the supported duration.
        flank_starts=[np.arange(len(t)-k*margin+1,step=max(1,margin//3)) for k in (3,4,6)]
        starts=np.concatenate(flank_starts)
        lasts=np.concatenate([a+(k-1)*margin for a,k in zip(flank_starts,(3,4,6))])
        delta=centers[lasts]-centers[starts]
        noise=np.maximum(scatter[starts]+scatter[lasts],(1/unit[starts]+1/unit[lasts])[:,None])
        opposed=(centers[starts]*centers[lasts]<0).all(1)
        distinguished=(np.minimum(abs(centers[starts]),abs(centers[lasts]))>noise).all(1)
        broad=(abs(delta)+noise>=orientation_scale).all(1)
        good=(delta[:,0]*delta[:,1]>0)&(np.abs(delta)>2*noise).all(1)&(broad|(opposed&distinguished))
        for i in np.flatnonzero(good):
            lo,hi=starts[i],lasts[i]+margin
            t0=t[lo:hi]
            local=widths[lo:hi]
            relative=(local-centers[lo])/delta[i]
            info=[_transition(relative[:,j],noise[i,j]/abs(delta[i,j]),t0,margin) for j in (0,1)]
            if any(x is None for x in info):
                continue
            q0,q1=info
            f0,f1=foot[lo:lo+margin],foot[hi-margin:hi]
            pre,post=np.median(f0,axis=0),np.median(f1,axis=0)
            err=max(1.,np.linalg.norm(f0-pre,axis=2).max(),np.linalg.norm(f1-post,axis=2).max())
            displacement=post-pre
            settle=np.linalg.norm(displacement[0]-displacement[1])
            if min(np.linalg.norm(displacement,axis=1).max(),settle)<=2*err:
                continue
            yaw=np.linalg.norm(delta[i]*np.median(unit[lo:hi]))
            if yaw<=settle:
                # Small twists accompanying a step are not sufficient. Compare
                # against this pivot's measured foot rearrangement, not pixels,
                # a universal angle, or the size of some other subject.
                continue
            if np.linalg.norm(displacement,axis=1).min()>err and displacement[0]@displacement[1]>0:
                # Both feet travelled together. Do not call that a planted
                # pivot merely because projected pelvis travel is foreshortened.
                continue
            direction=displacement[0]-displacement[1]
            change=((foot[lo:hi,0]-foot[lo:hi,1])-(pre[0]-pre[1]))@direction/settle
            foot_part=_transition(change/settle,err/settle,t0,margin)
            if foot_part is None:
                continue
            a=lo+min(q0[0],q1[0],foot_part[0]); b=lo+max(q0[1],q1[1],foot_part[1])
            # Exclude straight walking and body-wide side steps: compare the
            # observed pelvis travel with the pivot's local foot rearrangement.
            hip0=np.median(hip[lo:lo+margin],axis=0)
            if np.linalg.norm(hip[a:b]-hip0,axis=1).max()>=settle:
                continue
            strength=float(np.min(abs(delta[i])/noise[i])/(1+q0[2]+q1[2]+foot_part[2]))
            certain=bool(opposed[i] and distinguished[i])
            suggestions.append((a,b,strength,certain))
        margin*=2
    return suggestions


def apply(t, pose, valid, rows, measured, camera, smooth_s):
    """Add a turn when one person's shoulder, pelvis and foot changes agree over time."""
    # Retain previously supported synchronous calls. The additional path
    # searches observations without partitioning on legacy action labels.
    _synchronous_evidence(t,pose,valid,rows,measured,camera,smooth_s)
    usable = np.asarray(measured,bool) & valid[:,JOINTS].all(axis=1)
    usable &= np.array([not (r.get('action_event_id') or r.get('stair_sequence_supported')
                             or r['activity'] in PROTECTED
                             or any(h in PROTECTED for h in r.get('action_hypotheses',()))) for r in rows],bool)
    for lo,hi,p in _observed_spans(t,pose,usable,camera):
        chosen=np.zeros(hi-lo,bool)
        suggestions=_staggered_windows(t[lo:hi],p)
        for a,b,_,certain in sorted(suggestions,key=lambda q:(not q[3],-q[2])):
            if chosen[a:b].any():
                continue
            span=rows[lo+a:lo+b]
            # This is stationary-turn recovery, not a walk/run reclassifier.
            if any(r['activity']=='running' for r in span):
                continue
            # Strong locomotion is not overwritten by a small projected
            # twist. A clearly changed facing OR a stable in-place turn
            # independently confirmed by the older geometry can settle it.
            if (not certain and not any(r['activity']==TURN for r in span)
                    and all(r['activity'] in ('walking','gait_walk') for r in span)):
                continue
            chosen[a:b]=True
            for row in span:
                if row['activity']!=TURN:
                    row.update(pre_pivot_activity=row['activity'],activity=TURN,
                               activity_reason=STAGGERED,action_hypotheses=(),
                               ambiguous_gait=0,gait_inferred=0,locomotion_partial=int(not certain),
                               turn_evidence_level='observed_opposite_facings' if certain else 'partial_facing_change')
