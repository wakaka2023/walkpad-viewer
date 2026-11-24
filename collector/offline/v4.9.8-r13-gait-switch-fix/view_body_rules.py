from copy import deepcopy
import numpy as np
from activity_rules import features,smooth_valid,derivative,runs,label_hip_poses
from body_boundary_rules import body_features,label_body_poses,extend_event,terminal_standing_stop,GROUP_NAMES

def estimate_view(data,args):
    k=np.asarray(data['keypoints'],float)
    valid=np.isfinite(k[:,[5,6,11,12]]).all(axis=(1,2)) & (k[:,[5,6,11,12],2]>=args.keypoint_conf).all(axis=1)
    shoulder=k[:,[5,6],:2].mean(axis=1);hip=k[:,[11,12],:2].mean(axis=1)
    torso=np.linalg.norm(shoulder-hip,axis=1)
    valid &= torso>1
    if not valid.any():
        if getattr(args,'view','auto')=='auto':
            raise ValueError('Cannot infer view: shoulders/hips are not visible; specify --view and inspect the source.')
        ratio=None
    else:
        upright=valid & (torso>=np.quantile(torso[valid],.65))
        width=np.linalg.norm(k[:,5,:2]-k[:,6,:2],axis=1)
        ratio=float(np.median(width[upright]/torso[upright]))
    view=getattr(args,'view','auto')
    if view=='auto':view='side' if ratio<.35 else 'front'
    scores={}
    for side,ids in [('left',[5,11,13,15]),('right',[6,12,14,16])]:
        good=np.isfinite(k[:,ids]).all(axis=(1,2)) & (k[:,ids,2]>=args.keypoint_conf).all(axis=1)
        confidence=float(np.median(k[good][:,ids,2])) if good.any() else 0
        scores[side]=float(good.mean())+confidence
    side=max(scores,key=scores.get)
    return dict(view=view,visible_side=side,shoulder_torso_ratio=ratio,
                view_uncertain=ratio is None or .25<=ratio<=.45,
                selection='explicit' if getattr(args,'view','auto')!='auto' else 'upper_posture_projected_shoulder_width',
                side_scores=scores)

def side_features(data,args,side):
    f=features(data,args);t=f['t'];k=f['k'];scale=f['scale']
    sh,h,n,a=[5,11,13,15] if side=='left' else [6,12,14,16]
    valid=np.isfinite(k[:,[sh,h,n,a]]).all(axis=(1,2)) & (k[:,[sh,h,n,a],2]>=args.keypoint_conf).all(axis=1)
    positions=(k[:,[sh,h,n],:2]-k[:,a,None,:2])/scale
    positions,blocks=smooth_valid(t,positions,valid,args.smooth_seconds)
    head_valid=np.isfinite(k[:,0]).all(axis=1)&(k[:,0,2]>=args.keypoint_conf)
    head,head_blocks=smooth_valid(t,(k[:,0,:2]-k[:,a,:2])/scale,head_valid,args.smooth_seconds)
    shoulder,hip=positions[:,0],positions[:,1]
    speed=np.linalg.norm(derivative(t,positions,blocks),axis=2).max(axis=1)
    groups=[positions[:,:1],(shoulder-hip)[:,None],positions[:,1:2],positions[:,2:3],head[:,None]]
    return dict(base=f,t=t,valid=valid,head_valid=head_valid,groups=groups,body_speed=speed,speed=speed,
                body_top_signal=.6*shoulder[:,1]+.4*hip[:,1],top_signal=.6*shoulder[:,1]+.4*hip[:,1],
                blocks=blocks,upper_valid=valid)

def adaptive_reference(f,lo,hi,direction='before'):
    t=f['t'];valid=f['valid'];signal=f['body_top_signal'];speed=f['body_speed']
    ok=valid[lo:hi]&np.isfinite(signal[lo:hi])&np.isfinite(speed[lo:hi])
    if ok.sum()<3:return None,'missing_visible_body_reference',None
    values=signal[lo:hi][ok]
    floor=float(np.quantile(values,.03));peak=float(np.quantile(values,.95))
    high=ok & (signal[lo:hi]<=floor+max(.015,.12*(peak-floor)))
    speed_limit=max(.12,.15*float(np.quantile(speed[lo:hi][ok],.95)))
    dt=float(np.median(np.diff(t)))
    quiet=high&(speed[lo:hi]<=speed_limit)
    holds=[(a+lo,b+lo) for a,b in runs(quiet) if t[lo+b-1]-t[lo+a]+dt>=.10]
    if holds:

        a,b=holds[-1] if direction=='before' else holds[0]
        center=a+int(np.argmin(signal[a:b]));a=max(a,center-2);b=min(b,center+3)
        return (a,b),'adaptive_upright_hold',speed_limit
    candidates=[(a+lo,b+lo) for a,b in runs(high) if b-a>=2]
    if not candidates:return None,'no_observed_high_posture',speed_limit
    a,b=candidates[-1] if direction=='before' else candidates[0]
    center=a+int(np.argmin(signal[a:b]));a=max(a,center-2);b=min(b,center+3)
    return (a,b),'high_posture_turnaround_no_stationary_hold',speed_limit

def local_reference_bounds(f,core,lo,hi,rejected):
    t=f['t'];base=f['base'];ankles=base['ankles'];scale=base['scale']
    a=int(core['start_video_frame'])-1;b=int(core['end_video_frame'])
    anchor=np.nanmedian(ankles[a:b],axis=0)
    distance=np.linalg.norm(ankles-anchor,axis=2).max(axis=1)/scale
    core_distance=distance[a:b];core_distance=core_distance[np.isfinite(core_distance)]
    limit=max(.10,float(np.max(core_distance))+.025) if len(core_distance) else .10
    support=f['valid']&np.isfinite(distance)&(distance<=limit)

    dt=float(np.median(np.diff(t)))
    for x,y in runs(~support):
        if x>0 and y<len(t) and f['valid'][x:y].all() and t[y]-t[x]<=.10:
            support[x:y]=True
    for e in rejected:
        if e.get('reason')=='jump_like_bilateral_lift_and_fast_rise':
            x=int(e['start_video_frame'])-1;y=int(e['end_video_frame'])
            if y<=a:lo=max(lo,y)
            if x>=b:hi=min(hi,x)
    before_bad=np.flatnonzero(~support[lo:a])
    after_bad=np.flatnonzero(~support[b:hi])
    if len(before_bad):lo=lo+int(before_bad[-1])+1
    if len(after_bad):hi=b+int(after_bad[0])
    return lo,hi

def label_view_poses(data,args):
    info=estimate_view(data,args);args._view_info=info
    args._geometry_side=info['visible_side'] if info['view']=='side' else ''
    if info['view']=='front':
        events,frames,rejected,chosen=label_body_poses(data,args)

        if all(e['body_boundary_status']=='body_rule_candidate' for e in events):
            return events,frames,rejected,chosen
        f=body_features(data,args)
    else:
        events,frames,rejected,_=label_hip_poses(data,args)
        f=side_features(data,args,info['visible_side'])
        for j,row in enumerate(frames):
            row.update(upper_body_valid=int(f['upper_valid'][j]),head_valid=int(f['head_valid'][j]),
                       body_boundary_valid=int(f['valid'][j]),body_activity_score=float('nan'),movement_phase='')
        for e in events:
            e.update(hip_core_start_video_frame=e['start_video_frame'],hip_core_end_video_frame=e['end_video_frame'],
                     hip_core_start_video_s=e['start_video_s'],hip_core_end_video_s=e['end_video_s'],
                     body_boundary_status='visible_side_core')
        chosen='visible_'+info['visible_side']+'_body'
    if not events:return events,frames,rejected,chosen
    t=f['t'];bottoms=[int(e['bottom_video_frame'])-1 for e in events]
    changed=[];terminal_hold=None
    for i,e in enumerate(events):
        if info['view']=='front' and e['body_boundary_status']=='body_rule_candidate':continue
        old=deepcopy(e)
        core=dict(e,start_video_frame=e['hip_core_start_video_frame'],end_video_frame=e['hip_core_end_video_frame'])
        lo=bottoms[i-1] if i else 0
        hi=bottoms[i+1] if i+1<len(events) else len(t)
        lo,hi=local_reference_bounds(f,core,lo,hi,rejected)
        before,before_quality,_=adaptive_reference(f,lo,bottoms[i],'before')
        after,after_quality,_=adaptive_reference(f,bottoms[i],hi,'after')
        start,end,details,status=extend_event(f,core,before,after,args.body_fraction)

        uncapped=end
        end,recovery=terminal_standing_stop(f,core,before,end)
        terminal=recovery if i==len(events)-1 else dict(status='has_following_cycle')
        if details is None:
            e.update(body_boundary_status='view_adaptive_failed_'+status,
                     before_reference_quality=before_quality,after_reference_quality=after_quality)
            continue

        previous_end=int(events[i-1]['end_video_frame']) if i else 0
        next_start=int(events[i+1]['start_video_frame'])-1 if i+1<len(events) else len(t)-1
        if start<previous_end or end>next_start:

            start=max(start,previous_end);end=min(end,next_start)
        if not start<=int(core['start_video_frame'])-1<int(core['end_video_frame'])<=end:
            e['body_boundary_status']='view_adaptive_failed_neighbor_conflict';continue
        a,b,ratios=details
        for j in range(a,b):
            score=float(np.max(ratios[j-a]))
            frames[j]['body_activity_score']=float(np.fmax(frames[j].get('body_activity_score',np.nan),score))
        e.update(start_video_frame=start+1,end_video_frame=end,end_exclusive_video_frame=end+1,
                 start_video_s=float(t[start]),end_video_s=float(t[end]),duration_s=float(t[end]-t[start]),
                 start_earlier_s=float(e['hip_core_start_video_s']-t[start]),
                 end_later_s=float(t[end]-e['hip_core_end_video_s']),
                 body_boundary_status=info['view']+'_adaptive_body_candidate',
                 recovery_stop_status=recovery['status'],
                 before_reference_quality=before_quality,after_reference_quality=after_quality,
                 head_valid_fraction=float(f['head_valid'][start:end].mean()),
                 start_driver=GROUP_NAMES[int(np.argmax(ratios[start-a]))] if start>=a else 'reference_partition',
                 end_driver='upright_body_hold' if recovery.get('status')=='first_upright_body_hold' else GROUP_NAMES[int(np.argmax(ratios[min(end-1,b-1)-a]))],
                 terminal_stop_status=terminal['status'],
                 terminal_uncapped_end_video_frame=uncapped if i==len(events)-1 else '',
                 terminal_uncapped_end_video_s=float(t[uncapped]) if i==len(events)-1 else '',
                 terminal_speed_limit_leg_s=terminal.get('speed_limit',''),
                 terminal_confirmation_end_video_frame=terminal['confirmation_end']+1 if 'confirmation_end' in terminal else '')
        alternatives=[extend_event(f,core,before,after,q) for q in (.02,.05)]
        alternatives=[(v[0],terminal_standing_stop(f,core,before,v[1])[0],v[2],v[3]) for v in alternatives]
        e.update(start_sensitivity_first_frame=min(v[0]+1 for v in alternatives),
                 start_sensitivity_last_frame=max(v[0]+1 for v in alternatives),
                 end_sensitivity_first_frame=min(v[1] for v in alternatives),
                 end_sensitivity_last_frame=max(v[1] for v in alternatives))
        if terminal.get('status')=='first_upright_body_hold':terminal_hold=terminal
        changed.append((old,e))
    for prior,next_event in zip(events[:-1],events[1:]):
        if prior['end_video_frame']>=next_event['start_video_frame']:
            raise ValueError('Unresolved body boundary overlap; inspect source frames.')

    for old,e in changed:
        for j in range(old['start_video_frame']-1,old['end_video_frame']):
            if frames[j]['rep_id']==e['rep_id']:
                frames[j].update(rep_id=0,squat_label=-1,reason='outside_view_body_interval',movement_phase='')
    for old,e in changed:
        for j in range(e['start_video_frame']-1,e['end_video_frame']):
            phase='BODY_ONSET' if j+1<e['hip_core_start_video_frame'] else 'BODY_RECOVERY' if j+1>e['hip_core_end_video_frame'] else 'DOWN' if t[j]<=e['bottom_video_s'] else 'UP'
            frames[j].update(rep_id=e['rep_id'],squat_label=1,reason=info['view']+'_adaptive_body_boundary',movement_phase=phase)
    if terminal_hold:
        for j in range(terminal_hold['start'],terminal_hold['stop']):
            frames[j].update(rep_id=0,squat_label=0,reason='confirmed_terminal_upright_hold',movement_phase='')
    return events,frames,rejected,chosen
