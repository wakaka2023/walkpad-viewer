"""Track-local gait morphology. Frequency locates phase; it does not vote on gait type."""

import numpy as np
from scipy.linalg import lstsq
from scipy.ndimage import median_filter
from .activity_rules import runs

EPS = np.finfo(float).eps


def _fit(t, p, valid):
    """Fit anti-phase foot motion and its common torso harmonic in one window."""
    n = len(t)
    if n < 16:
        return None
    u = t - t.mean()
    span = t[-1] - t[0]
    dt = np.median(np.diff(t))
    if not (0 < dt and span > 10*dt):
        return None
    need = valid[:, [5, 6, 11, 12, 15, 16]].all(axis=1)
    ii = np.flatnonzero(need)
    if len(ii) < .76*n or np.max(np.diff(ii), initial=0) > max(2, n//9):
        return None
    xy = p[ii, :, :2].astype(float)
    hips = xy[:, [11, 12]].mean(axis=1)
    shoulders = xy[:, [5, 6]].mean(axis=1)
    unit = np.linalg.norm(hips-shoulders,axis=1)
    scale = np.median(unit)
    if not scale > 1:
        return None
    smooth_unit = median_filter(unit, size=max(1, (n//5)|1), mode='nearest')
    log_unit = np.log(np.maximum(smooth_unit,1.))
    slope = np.polyfit(u[ii]/span, log_unit, 1)
    scale_drift = abs(slope[0]) > np.ptp(log_unit-np.polyval(slope,u[ii]/span))
    if np.any(smooth_unit <= 1):
        return None
    ankles = xy[:, 15:17, :] - hips[:, None, :]
    feet = ankles / smooth_unit[:, None, None]
    torso = np.c_[hips[:,1], shoulders[:,1]] / scale
    vals = np.c_[feet.reshape(-1,4), torso]
    u = u[ii]
    w = np.hanning(n)[ii] + EPS
    base = np.c_[np.ones(len(u)), u/span, (u/span)**2]
    trend = base @ lstsq(base * w[:,None]**.5, vals * w[:,None]**.5, check_finite=False)[0]
    vals -= trend
    diff = vals[:,:2] - vals[:,2:4]
    energy = np.sum(w[:,None]*diff**2)
    if energy <= EPS:
        return None
    # Search resolution is stated in cycles per window and per sample, never a walk/run boundary.
    search = np.arange(1.4, span/(6*dt), .08)/span
    if len(search) < 3:
        return None
    osc = np.exp(2j*np.pi*search[:,None]*u)
    spec = osc @ (w[:,None]*diff)
    pick = int(np.argmax(np.sum(abs(spec)**2,axis=1)))
    if pick == 0:
        return None
    hz = search[pick]
    # The two feet must move in opposition, not simply be carried by body/camera motion.
    legs = osc[pick] @ (w[:,None]*vals[:,:4])
    left,right = legs.reshape(2,2)
    opposition = -float(np.vdot(left,right).real)
    n_eff = w.sum()**2 / np.sum(w*w)
    coherence = float(np.clip(2*np.sum(abs(spec[pick])**2)/(w.sum()*energy),0,1-EPS))
    leg_amp = 2*np.sqrt(np.sum(abs(spec[pick])**2))/w.sum()
    leg_gain = -n_eff*np.log1p(-coherence) - 6*np.log(n_eff)
    if opposition <= 0 or leg_gain <= 0:
        return None
    ref = spec[pick, int(np.argmax(abs(spec[pick])))]
    cyc = np.exp(4j*np.pi*hz*u)
    body = 2*(cyc @ (w[:,None]*vals[:,4:]))/w.sum()
    # Require agreement of pelvis and upper trunk. A torso swing on its own is not evidence of rebound.
    shared = max(0.,float((body[0]*np.conj(body[1])).real))**.5
    phase = body.mean()*np.conj(ref**2)/max(abs(ref)**2,EPS)
    noise = np.sqrt(np.average(vals[:,4:]**2, weights=w, axis=0)).mean()/np.sqrt(n_eff)
    # Normalize shared vertical movement by the projected leg excursion.
    ratio = shared / max(leg_amp, 1./scale)
    return dict(period=.5/hz, gait_gain=leg_gain, leg_amp=leg_amp, cycles=span*2*hz, scale_drift=scale_drift,
                impulse_ratio=ratio, body_snr=shared/max(noise,1./scale),
                phase=phase/max(abs(phase),EPS), resolution=1./scale)


def _mode_split(signal, uncertainty):
    """Find two evidenced within-person modes; prefer one mode if BIC cannot justify a split."""
    x = np.sort(np.asarray(signal,float))
    n = len(x)
    if n < 6:
        return None
    median = np.median(x)
    floor = max(float(np.median(uncertainty))**2,EPS)
    rss = max(np.mean((x-median)**2),floor)
    base = n*np.log(rss)+2*np.log(n)
    best = base
    result = None
    for j in range(3,n-2):
        if x[j] == x[j-1]:
            continue
        left,right = np.median(x[:j]),np.median(x[j:])
        error = max((np.sum((x[:j]-left)**2)+np.sum((x[j:]-right)**2))/n,floor)
        # Encoding the assignment penalizes tiny islands and splitting continuous noise.
        prob = j / n
        bic = n*np.log(error) + 4*np.log(n) - 2*(j*np.log(prob)+(n-j)*np.log1p(-prob))
        if bic < best:
            best = bic
            result = float((left+right)/2)
    return result


def detect(t, pose, valid, rows, measured, *, diagnostics=None, contextual=None):
    """Return direct window support. No filename, reference timestamps or translation speed."""
    n = len(t)
    labels = np.full(n,'',object)
    periods = np.full(n,np.nan)
    ratios = periods.copy()
    direct = np.zeros(n,bool)
    if n < 16:
        return labels, periods, ratios, direct
    dt = float(np.median(np.diff(t)))
    if not dt > 0:
        return labels, periods, ratios, direct
    # Window sizes are observation supports, not walk/run cadence cutoffs. Use longer context only
    # if short context cannot resolve two-sided leg motion. Refine to local phase where possible.
    eligible = np.asarray(measured,bool) & np.array([
        r['activity'] in ('walking','gait_walk','running','standing','unknown','body_motion','stepping','fast_stepping')
        and not r['action_event_id'] and not r.get('stair_sequence_supported') for r in rows])
    raw = np.array([r['activity'] for r in rows],object)
    old = np.array([r.get('gait_evidence','') for r in rows],object)
    fits = []
    for a,b in runs(eligible):
        step = max(1,round(.167/dt))
        centers = np.arange(a,b,step)
        edges = np.r_[a,(centers[:-1]+centers[1:]+1)//2,b]
        for i,l,r in zip(centers,edges[:-1],edges[1:]):
            if not valid[i,[15,16]].all():
                continue
            fitted=None
            for half in (1.20,1.85,2.45):
                s,e = np.searchsorted(t, (t[i]-half,t[i]+half))
                s,e = max(s,a),min(e,b)
                if e-s < 16:
                    continue
                q = _fit(t[s:e],pose[s:e],valid[s:e])
                if q is None:
                    continue
                if half != 2.45 and q['period']*4.3 >= t[e-1]-t[s] and (e<b or s>a):
                    continue
                fitted=q
                break
            if fitted is not None:
                fits.append((i,l,r,fitted))
    if not fits:
        return labels, periods, ratios, direct
    ii = np.array([p[0] for p in fits])
    values = np.array([q['impulse_ratio'] for _,_,_,q in fits])
    amps = np.array([q['leg_amp'] for _,_,_,q in fits])
    resolution = np.array([q['resolution'] for _,_,_,q in fits])
    snr = np.array([q['body_snr'] for _,_,_,q in fits])
    phase = np.array([q['phase'] for _,_,_,q in fits])
    # Quiet pose has a different leg swing envelope. Infer noise from reliable standing in this track.
    quiet = amps[raw[ii]=='standing']
    moving = amps > resolution
    if len(quiet)>=3:
        middle,upper = np.quantile(quiet,[.25,.5])
        quiet_snr = snr[raw[ii]=='standing']
        qmid,qhi = np.quantile(quiet_snr,[.5,.9])
        moving &= (amps > upper+(upper-middle)) | (snr > qhi+(qhi-qmid))
    keep = moving & (raw[ii] != 'standing')
    # Decimate overlapping observations so mode selection is counted in gait cycles, not video frames.
    selected=[]
    for j in np.flatnonzero(keep):
        if not selected or t[ii[j]]-t[ii[selected[-1]]] >= fits[j][3]['period']:
            selected.append(j)
    selected=np.asarray(selected,int)
    uncertainty = resolution/np.maximum(amps,resolution)
    score = np.log1p(values / max(np.median(uncertainty[keep]) if keep.any() else 1.,EPS))
    split = _mode_split(score[selected],np.log1p(1./np.maximum(1.,snr[selected]))) if len(selected) else None
    run_old = (raw[ii]=='running') & np.isin(old[ii],('short_running_rebound','running_rhythm_candidate','rhythm_candidate'))
    walk_old = (raw[ii]=='gait_walk') & np.isin(old[ii],('walking_rhythm_candidate','walking_support_handoff','walking_advancing_handoffs'))
    proposed = np.full(len(ii),'',object)
    anchor = keep & run_old & (snr>1.)
    periods_fit = np.array([q['period'] for _,_,_,q in fits])
    cycles = np.array([q['cycles'] for _,_,_,q in fits])
    run_ref = np.sum(phase[anchor] * snr[anchor])
    opposing = (phase * np.conj(run_ref)).real < 0
    # A different view can rotate phase; an opposite phase only demotes a coherent high-recoil
    # proposal, not an independently observed running anchor.
    opposing &= ~run_old & (snr>1.)
    agreement = (phase*np.conj(run_ref)).real/max(abs(run_ref), EPS)
    # A concentrated reference gives a meaningful projection-phase check. Near-axis bodies
    # that change apparent size can rotate phase; the divergent portion is not forced to walk.
    ref_concentration = abs(run_ref)/max(snr[anchor].sum(), EPS)
    phase_expected = agreement >= (2*ref_concentration - 1)
    phase_possible = ~opposing & (phase_expected | (score >= np.median(score[anchor]) if anchor.any() else False))
    if split is not None:
        high = score > split
        if (anchor & high).any():
            proposed[high & (snr>1.) & phase_possible] = 'running'
            proposed[~high | (high & ~phase_possible & ~run_old)] = 'gait_walk'
        elif not anchor.any() and (keep & walk_old & ~high).any():
            proposed[~high] = 'gait_walk'
    if anchor.any():
        # A track with a single sampled running mode also has an empirical envelope. This recovers
        # the same gait when the person's motion is in depth rather than across the camera image.
        low = np.quantile(score[anchor],.25)
        spread = np.median(np.abs(score[anchor]-np.median(score[anchor])))
        run_unset = (proposed=='') & (score>=low-spread) & (snr>1.) & phase_possible
        proposed[run_unset] = 'running'
    if (walk_old & keep).any():
        walk_ref = walk_old & keep & (proposed!='running')
        if walk_ref.any():
            center = np.median(score[walk_ref])
            spread = np.median(np.abs(score[walk_ref]-center))
            upper = np.quantile(score[walk_ref],.9)+spread
            proposed[(proposed=='') & ((score<=upper) | opposing)] = 'gait_walk'
    proposed[~keep]=''
    # Retain no one-off override. Run and walk both need repeated support; the minimum is
    # measured in this person's observed steps, not an absolute span or cadence.
    for a,b in runs(proposed=='running'):
        supported = t[fits[b-1][2]-1] + dt - t[fits[a][1]]
        if supported < 2*np.median(periods_fit[a:b]) and not run_old[a:b].any():
            proposed[a:b] = np.where(walk_old[a:b],'gait_walk','')
    # At an onset or offset, a centered window can mix two phases. Expand a direct regime
    # only for a few locally measured steps with coherent leg and torso evidence.
    # This is a consistency check, not a walk/run period cutoff.
    # Do not turn a direct running span into a short walking interruption just because a
    # window is weak. Strongly conflicting phase may still prevent promoting a candidate.
    direct_old_run = (raw[ii]=='running') & np.isin(old[ii],('short_running_rebound','running_rhythm_candidate'))
    proposed[(proposed=='gait_walk') & direct_old_run & ~opposing] = ''
    core = proposed.copy()
    for a,b in runs(core=='running'):
        if b-a<2:
            continue
        for edge,sign in ((a,-1),(b-1,1)):
            near = slice(max(a,edge-2),min(b,edge+3))
            period = np.median(periods_fit[near])
            ref = np.sum(phase[near]*snr[near])
            j = edge+sign
            while 0<=j<len(ii):
                limit = 5.0 if fits[j][3]['scale_drift'] else 2.0
                if abs(t[ii[j]]-t[ii[edge]]) > limit*period:
                    break
                prev = j-sign
                depth = fits[j][3]['scale_drift']
                local_ref = phase[prev] if depth else ref
                expected_period = periods_fit[prev] if depth else period
                if (not keep[j] or snr[j]<=1. or abs(ii[j]-ii[prev])>2*step
                        or (not phase_possible[j] and not depth)
                        or (phase[j]*np.conj(local_ref)).real<=0
                        or abs(np.log(periods_fit[j]/expected_period))>1./cycles[j]):
                    break
                proposed[j]='running'
                j+=sign
    context_windows = np.zeros(len(fits),bool)
    for a,b in runs(proposed=='gait_walk'):
        if a==0 or b==len(ii) or proposed[a-1]!='running' or proposed[b]!='running':
            continue
        if max(np.diff(ii[a-1:b+1]))>2*step or np.any(opposing[a:b]):
            continue
        left,right = periods_fit[a-1],periods_fit[b]
        duration = t[fits[b-1][2]-1]+dt-t[fits[a][1]]
        p = np.sqrt(left*right)
        ref = phase[a-1]+phase[b]
        match = abs(np.log(periods_fit[a:b]/p)) <= 1./cycles[a:b]
        if (duration <= left+right+EPS and np.all(match) and np.all(snr[a:b]>1.)
                and np.vdot(phase[a-1],phase[b]).real>0
                and np.all((phase[a:b]*np.conj(ref)).real>0)):
            # Recoil magnitude alone fluctuated, but the leg and torso phase stayed coherent.
            proposed[a:b] = 'running'
            context_windows[a:b]=True
    if diagnostics is not None:
        diagnostics.extend((float(t[i]),q["period"],a,s, int(k), str(old[i]), str(raw[i]), float(sc), complex(ph), str(pr)) for (i,_,_,q),a,s,k,sc,ph,pr in zip(fits,amps,snr,keep,score,phase,proposed))
    # Keep the evidence around a boundary rather than turn every overlapping window into an island.
    for j,(i,l,r,q) in enumerate(fits):
        if not proposed[j]:
            continue
        kind = proposed[j]
        labels[l:r] = kind
        periods[l:r] = q['period']
        ratios[l:r] = values[j]
        direct[l:r] = valid[l:r,[15,16]].all(axis=1)
        if contextual is not None:
            contextual[l:r] = context_windows[j] & direct[l:r]
    labels[~direct]=''
    return labels, periods, ratios, direct


def apply(t, pose, valid, rows, measured):
    contextual = np.zeros(len(t),bool)
    labels, periods, ratios, direct = detect(t, pose, valid, rows, measured, contextual=contextual)
    for i in np.flatnonzero(direct):
        row = rows[i]
        # Do not alter event-backed or standing frames, even at an observation-window edge.
        if (row['action_event_id'] or row.get('stair_sequence_supported')
                or row['activity'] in ('standing','squat','stairs_up','stairs_down')):
            continue
        kind = labels[i]
        reason = ('bracketed_running_with_matching_leg_torso_phase' if contextual[i]
                  else 'person_local_ankle_opposition_and_shared_torso_cycle_shape')
        row.update(pre_cycle_gait_activity=row['activity'], pre_cycle_gait_evidence=row.get('gait_evidence',''),
                   activity=kind, activity_reason=reason,
                   gait_evidence='person_local_cycle_morphology', personal_gait_step_period_s=float(periods[i]),
                   personal_gait_recoil_leg_ratio=float(ratios[i]),
                   gait_step_hz_candidate=float(1./periods[i]), action_hypotheses=(), ambiguous_gait=0,
                   locomotion_partial=0, gait_inferred=int(contextual[i]))
