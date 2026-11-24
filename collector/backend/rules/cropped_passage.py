"""Use visible lower limbs and adjacent gait for cropped entries/exits."""
import numpy as np
from scipy.ndimage import median_filter
from .activity_rules import runs
from .running_turn import _world_xy
from .walk_continuity import _step
from .gait_gap_evidence import GAIT, UNRESOLVED

LEG = 'visible_lower_limb_cycle_with_adjacent_measured_gait'
TURN = 'cropped_traversal_bilateral_hip_orientation_reversal'
DEPTH = 'cropped_edge_with_same_heading_and_measured_depth_progression'


def _cycle(t, legs, period, resolution):
    if len(t)<9 or t[-1]-t[0]<2*period:
        return False
    u=t-t.mean(); n=len(t)
    lin=np.c_[np.ones(n),u/(t[-1]-t[0])]
    y=legs.reshape(n,4)
    v=y-lin@np.linalg.lstsq(lin,y,rcond=None)[0]
    raw=np.sum((v[:,:2]-v[:,2:])**2); floor=n*resolution**2
    for f in (.46/period,.5/period,.54/period):
        phase=2*np.pi*f*u
        x=np.c_[lin,np.cos(phase),np.sin(phase)]
        beta=np.linalg.lstsq(x,y,rcond=None)[0]
        v=y-x@beta
        err=np.sum((v[:,:2]-v[:,2:])**2)
        l,r=beta[2:,:2],beta[2:,2:]
        if n*np.log(max(raw,floor)/max(err,floor))>5*np.log(n) and np.sum(l*r)<0 and np.linalg.norm(l-r)>3*resolution:
            return True
    return False


def bridges(t, ends, labels, rows, inferred, protected, reviews, measured, pose, valid, camera, height):
    result=np.full(len(t),'',object); why=result.copy()
    weak=(labels=='unknown') & measured & ~protected & np.array([
        rv['activity_status']=='other' and r['activity'] in ('unknown','body_motion','stepping','fast_stepping')
        and r.get('gait_evidence','') in UNRESOLVED and not (r['action_event_id'] or r.get('stair_sequence_supported'))
        for r,rv in zip(rows,reviews)],bool)
    gait=np.isin(labels,GAIT) & measured & ~protected
    direct=gait & ~inferred & np.array([rv['activity_status']=='supported' for rv in reviews],bool)
    if not weak.any() or not direct.any():return result,why
    steps=np.array([_step(r) for r in rows]); dt=ends-t
    for s,e in runs(weak):
        flanks=[]
        for direction,edge in ((-1,s-1),(1,e)):
            i=edge
            while 0<=i<len(t) and gait[i] and not direct[i]: i+=direction
            if not 0<=i<len(t) or not direct[i]:
                flanks.append(None); continue
            kind=labels[i]; nearest=i
            while 0<=i<len(t) and gait[i] and labels[i]==kind: i+=direction
            a,b=(i+1,s) if direction<0 else (e,i)
            ids=np.arange(a,b)[direct[a:b] & (labels[a:b]==kind)]; near=ids[::-1] if direction<0 else ids
            local=near[np.isfinite(steps[near])]
            if not len(local): flanks.append(None); continue
            period=float(np.median(steps[local[:max(3,len(local)//3)]]))
            if abs(t[nearest]-t[edge])>period or dt[ids].sum()<period/2:
                flanks.append(None); continue
            flanks.append((a,b,kind,period))
        left,right=flanks
        if left is None and right is None:continue
        lo=max(left[0],np.searchsorted(t,t[s]-2*left[3])) if left else s
        hi=min(right[1],np.searchsorted(t,t[e-1]+2*right[3],side='right')) if right else e
        q=_world_xy(t,pose[lo:hi,:,:2],camera,lo,hi)
        if q is None:continue
        v=valid[lo:hi]; u=t[lo:hi]; A=s-lo; B=e-lo
        hips=q[:,11:13]; raw=hips[:,1,0]-hips[:,0,0]
        width=median_filter(raw,3,mode='nearest'); hip=v[:,11:13].all(1)
        noise=max(1.,3*np.median(abs(raw-width)))
        # Crop detection is relative to projected person size, not a fixed ROI.
        original=pose[lo:hi]
        thigh=np.linalg.norm(original[:,13:15,:2]-original[:,11:13,:2],axis=2)
        ref=np.nanmedian(np.where(v[:,11:13] & v[:,13:15],thigh,np.nan)) if (v[:,11:13] & v[:,13:15]).any() else 0
        clip=(~v[:,15:17].all(1) & (((height-original[:,11:15,1])<ref) & v[:,11:15]).any(1)) | (~v[:,5:7].all(1) & (original[:,11:13,1]<ref*2).any(1))
        clip |= ~v[:,5:7].all(1) & ~v[:,15:17].any(1) & hip
        if left and right and A>=3 and len(u)-B>=3 and hip.all() and clip[A:B].any() and u[B]-u[A]<=8*min(left[3],right[3]):
            pre,post=width[:A],width[B:]; w0,w1=np.median(pre),np.median(post)
            err=max(noise,2*np.median(abs(pre-w0)),2*np.median(abs(post-w1)))
            if w0*w1<0 and min(abs(w0),abs(w1))>err:
                m=width[A:B]*np.sign(w0); n=width[A:B]*np.sign(w1)
                start=np.flatnonzero(m>=abs(w0)-err); stop=np.flatnonzero(n>=abs(w1)-err)
                # Find the nearest non-overlapping representatives of the opposite views.
                pairs=[(int(start[i-1]),int(b)) for b in stop if (i:=np.searchsorted(start,b,side='left'))>0]
                if pairs:
                    a,b=min(pairs,key=lambda p:p[1]-p[0])
                    pivot=(t[s+a]+t[s+b])/2
                    # The hip width may flip in a single pose frame; report a
                    # bounded transition around it, not spurious frame precision.
                    skirt=min(left[3],right[3])/2
                    idx=np.arange(max(s,np.searchsorted(t,t[s+a]-skirt)),
                                  min(e,np.searchsorted(t,t[s+b]+skirt,side='right')))
                    before_peak=m[:a+1].max()>abs(w0)+noise
                    after_peak=n[b:].max()>abs(w1)+noise
                    if len(idx)>=2 and clip[idx-lo].any() and before_peak and after_peak:
                        if left[2]==right[2]=='running':
                            near=np.arange(s,s+a+1)
                            near=near[(t[near]>=pivot-left[3]*1.5) & clip[near-lo]]
                            if len(near):
                                peak=near[np.argmax(np.sign(w0)*width[near-lo])]
                                if np.sign(w0)*(width[peak-lo]-width[s+a-lo])>err:
                                    idx=np.arange(min(peak,idx[0]),idx[-1]+1)
                        result[idx]='running_turn' if left[2]==right[2]=='running' else 'moving_turn'; why[idx]=TURN
                        # For a confirmed run on both sides, a moving near-view
                        # passage remains a run-turn candidate while both legs
                        # are out of the image. Do not reuse this for walk/run changes.
                        if left[2]==right[2]=='running':
                            near=np.arange(s,e)
                            depth=abs(width[near-lo])
                            cropped=clip[near-lo] & (depth >= min(abs(w0),abs(w1)))
                            for l,r in runs(cropped):
                                if t[near[l]]-skirt<=pivot<=ends[near[r-1]]+skirt:
                                    result[near[l:r]]='running_turn'; why[near[l:r]]=TURN
                            # Before and after the pivot, sustained near-view
                            # progression can carry the run as a candidate, not
                            # a confirmation of contacts while the feet are out.
                            turn_ids=np.flatnonzero(why[s:e]==TURN)+s
                            for ids,sign,period in ((np.arange(s,turn_ids[0]),1,left[3]),
                                                    (np.arange(turn_ids[-1]+1,e),-1,right[3])):
                                for z in ids:
                                    j,k=np.searchsorted(u,(t[z]-period/2,t[z]+period/2))
                                    w_slice=width[j:k]
                                    if len(w_slice)<4 or w_slice.min()*w_slice.max()<=0:continue
                                    x=u[j:k]-u[j:k].mean(); d=abs(w_slice); d=d-d.mean()
                                    slope=x@d/(x@x); progress=sign*slope*(x[-1]-x[0]); error=np.median(abs(d-x*slope))
                                    if progress>max(3*noise,3*error):
                                        result[z]='running'; why[z]='same_run_flanks_and_observed_depth_progression_near_turn'

        for side,flank in enumerate(flanks):
            if flank is None:continue
            period=flank[3]; a,b=(0,A) if side==0 else (B,len(u))
            if b-a<4:continue
            sg=np.sign(np.median(width[a:b])); targets=np.arange(A,B) if side==0 else np.arange(B-1,A-1,-1)
            edge=s if side==0 else e-1
            torso=np.linalg.norm(q[:,5:7].mean(1)-hips.mean(1),axis=1)
            shoulders=q[:,6,0]-q[:,5,0]
            sz=np.c_[torso,abs(shoulders),abs(width)]
            good=hip & v[:,5:7].all(1) & (sz>2).all(1)
            # A clipped gait edge can remain a candidate when the observed
            # trunk still advances in the same view/scale trend as that bout.
            # Demand three agreeing measurements: a changing hip width alone
            # can be body yaw rather than progression in depth.
            def progression(l,r):
                if r-l<5 or not good[l:r].all() or not np.all(sg*width[l:r]>noise):return 0
                x=u[l:r]-u[l:r].mean(); z=np.log(sz[l:r]); z=z-z.mean(0)
                slope=x@z/(x@x); amplitude=slope*(x[-1]-x[0]); err=np.median(abs(z-x[:,None]*slope),axis=0)
                eps=np.maximum(2*err,2/np.median(sz[l:r],axis=0))
                for sign in (-1,1):
                    if np.all(sign*amplitude>eps):return sign
                return 0
            al=max(a,np.searchsorted(u,u[b-1]-period)) if side==0 else a
            ar=b if side==0 else min(b,np.searchsorted(u,u[a]+period,side='right'))
            trend=progression(al,ar)
            for k in targets:
                i=lo+k
                if abs(t[i]-t[edge])>=4*period or why[i]==TURN or sg*width[k]<=noise:break
                if trend and abs(t[i]-t[edge])<2*period and not v[k,15:17].all():
                    dl,dr=np.searchsorted(u,(u[k]-period/2,u[k]+period/2),side='left')
                    if clip[max(0,dl):dr].any() and progression(dl,dr)==trend:
                        if not why[i] or result[i]==flank[2]:result[i]=flank[2]; why[i]=DEPTH
                l,r=np.searchsorted(u,(u[k]-1.5*period,u[k]+1.5*period),side='left'); r=min(r,len(u))
                if r-l<9 or np.max(dt[lo+l:lo+r])>period/3 or not hip[l:r].all() or not (sg*width[l:r]>0).all(): continue
                if not np.any(direct[lo+l:lo+r] & (labels[lo+l:lo+r]==flank[2])):continue
                for pair in ((15,16),(13,14)):
                    if not v[l:r,pair].all() or (pair==(13,14) and v[l:r,15:17].all()):continue
                    leg=q[l:r,pair,:]-hips[l:r]
                    unit=np.linalg.norm(leg,axis=2).mean(1)
                    scale=median_filter(unit,size=max(3,round(period/np.median(dt[lo+l:lo+r]))|1),mode='nearest')
                    if np.any(scale<=2):continue
                    norm=leg/scale[:,None,None]; jitter=1/np.median(scale)
                    j,h=np.searchsorted(u[l:r],(u[k]-.5*period,u[k]+.5*period))
                    if h-j<3 or np.linalg.norm(np.ptp(norm[j:h,0]-norm[j:h,1],axis=0))<=4*jitter:continue
                    if _cycle(u[l:r],norm,period,jitter):
                        if why[i] and result[i]!=flank[2]: result[i]=why[i]=''
                        else:result[i]=flank[2]; why[i]=LEG
                        break
    # A new lower-limb or depth recovery must actually attach to the episode,
    # rather than create another isolated walk/run island in a long unknown.
    for kind in GAIT:
        connected=(labels==kind)|(result==kind)
        for a,b in runs(connected):
            if not (labels[a:b]==kind).any() and (result[a:b]==kind).any():
                ix=np.arange(a,b)[np.isin(why[a:b],(LEG,DEPTH))]
                result[ix]=why[ix]=''
    return result,why

