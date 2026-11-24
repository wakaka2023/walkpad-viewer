import json

import numpy as np

from grading_pose_features import runs
from motion_fusion import sustained_event


VERSION = 'phase_matched_indirect_cues_v1'
LABELS = {'early_trunk_fold': 'Trunk folds before substantial leg descent',
          'late_trunk_recovery': 'Leg height recovers while trunk remains folded',
          'bottom_extra_fold': 'Extra trunk fold with little bottom leg motion',
          'ascent_extra_fold': 'Ascent trunk folds more at matched leg posture'}


def _median(values, mask):
    a=np.asarray(values)[mask]
    return float(np.median(a[np.isfinite(a)])) if np.isfinite(a).any() else np.nan


def _blocks(t, valid):
    out=np.full(len(t),-1,int)
    dt=float(np.median(np.diff(t)))
    for a,b in runs(valid):
        cuts=np.flatnonzero(np.diff(t[a:b])>max(.1,3*dt))+a+1
        edges=[a]+cuts.tolist()+[b]
        for lo,hi in zip(edges[:-1],edges[1:]):out[lo:hi]=lo
    return out


def _angle(a,b,c):
    x,y=a-b,c-b
    d=np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1)
    return np.degrees(np.arccos(np.clip(np.sum(x*y,axis=1)/np.maximum(d,1e-9),-1,1)))


def extract(t, rep, body, valid, scale, view, stable, cfg):
    t=np.asarray(t,float);body=np.asarray(body,float)
    front=view=='front_or_oblique';side=view=='side_or_oblique'
    start,stop,bottom=float(rep.start_video_s),float(rep.end_video_s),float(rep.bottom_video_s)
    active=(t>=start)&(t<stop)
    valid=np.asarray(valid,bool)&np.isfinite(body).all(axis=(1,2))
    blank=dict(indirect_features_version=VERSION,indirect_assessment_available=0,
               indirect_fault_level=np.nan,indirect_cue_count=0,indirect_cues_json='[]',
               indirect_checks_json='{}',indirect_observation_reason='',
               indirect_early_fold_value=np.nan,indirect_late_fold_value=np.nan,
               indirect_bottom_extra_fold_value=np.nan,indirect_ascent_extra_fold_value=np.nan)
    if not (front or side) or not stable:
        blank['indirect_observation_reason']='A stable front or side view is required'
        return blank
    observed=valid&active
    gaps=np.diff(np.r_[start,t[observed],stop])
    if observed.sum()/max(1,active.sum())<cfg['minimum_pose_coverage'] or max(gaps)>cfg['maximum_pose_gap_s']:
        blank['indirect_observation_reason']='Required visible body points have insufficient coverage'
        return blank
    pre=valid&(t>=start-.3)&(t<=start+.03)
    base_ids=np.flatnonzero(pre)
    if len(base_ids)<3 or t[base_ids[-1]]-t[base_ids[0]]<cfg['indirect_reference_min_s']:
        blank['indirect_observation_reason']='Insufficient initial reference duration'
        return blank
    shoulder,hip,knee,ankle=[body[:,i] for i in range(4)]
    trunk=shoulder-hip
    height=(hip-ankle)[:,1]
    base_height=_median(height,pre)
    base_torso=_median(-trunk[:,1],pre)
    base_shin=_median((ankle-knee)[:,1],pre)
    base_length=_median(np.linalg.norm(trunk,axis=1),pre)
    reference_torso=base_torso if front else base_length
    if reference_torso<cfg['minimum_top_torso_leg_ratio']*scale or base_shin<.1*scale:
        blank['indirect_observation_reason']='Initial body projection does not support normalization'
        return blank
    bottom_mask=observed&(np.abs(t-bottom)<=max(.12,.08*(stop-start)))
    excursion=_median(height,bottom_mask)-base_height
    if not np.isfinite(excursion) or excursion<cfg['indirect_min_hip_excursion_leg_ratio']*scale:
        blank['indirect_observation_reason']='Insufficient observed hip-height excursion for phase comparison'
        return blank
    progress=(height-base_height)/excursion
    shin_ratio=(ankle-knee)[:,1]/base_shin
    projection=np.full(len(t),np.nan)
    if front:
        torso_ratio=-trunk[:,1]/base_torso
        adjusted=torso_ratio/np.maximum(shin_ratio,.15)
        projection=np.maximum(torso_ratio,adjusted)
    lean=np.degrees(np.arctan2(np.abs(trunk[:,0]),-trunk[:,1]))
    base_lean=_median(lean,pre)
    knee_angle=_angle(hip,knee,ankle)
    knee_base=_median(knee_angle,pre)
    knee_excursion=knee_base-_median(knee_angle,bottom_mask)
    knee_progress=(knee_base-knee_angle)/knee_excursion if knee_excursion>=cfg['joint_onset_min_excursion_deg'] else np.full(len(t),np.nan)
    if side:
        trunk_length=np.linalg.norm(trunk,axis=1)
        length_ratio=trunk_length/base_length
        valid=valid&(length_ratio>=cfg['indirect_side_length_ratio_min'])&(length_ratio<=cfg['indirect_side_length_ratio_max'])
        accepted=valid&active
        accepted_gaps=np.diff(np.r_[start,t[accepted],stop])
        if accepted.sum()/max(1,active.sum())<cfg['minimum_pose_coverage'] or max(accepted_gaps)>cfg['maximum_pose_gap_s']:
            blank['indirect_observation_reason']='Side torso projection is not sufficiently stable for indirect cues'
            return blank
    block=_blocks(t,valid&active)
    observed=active&valid
    duration=max(cfg['indirect_min_duration_s'],min(cfg['indirect_duration_cap_s'],cfg['indirect_duration_fraction']*(stop-start)))
    near_top=(progress<=cfg['indirect_near_top_progress'])&(progress>=-cfg['indirect_near_top_progress'])
    recovered=shin_ratio>=cfg['indirect_front_min_shin_ratio'] if front else knee_progress<=cfg['indirect_near_top_progress']
    unit='upright_torso_fraction' if front else 'projected_degrees'
    fold=1-projection if front else lean-base_lean
    threshold=1-cfg['indirect_front_max_folded_projection'] if front else cfg['indirect_side_top_fold_deg']
    change_threshold=cfg['indirect_front_extra_fold_fraction'] if front else cfg['indirect_side_extra_fold_deg']
    folded=projection<=cfg['indirect_front_max_folded_projection'] if front else lean-base_lean>=threshold
    q=-projection if front else lean
    hip_relative,knee_relative=hip-ankle,knee-ankle
    events=[];checks={}

    def add(kind,values,mask,limit,reference=None,distances=None,available=True):
        status='PASS' if available else 'UNAVAILABLE'
        result=dict(status=status,threshold=float(limit),unit=unit,persistence_s=float(duration),
                    value=None,event=None)
        event=sustained_event(t,values,mask&observed,limit,duration,kind) if available else None
        finite=mask&observed&np.isfinite(values)
        if finite.any():result['value']=float(np.max(values[finite]))
        if event:
            lo=int(np.searchsorted(t,event['start_s']))
            hi=int(np.searchsorted(t,event['stop_s']))
            ids=np.arange(lo,min(hi,len(t)))
            ids=ids[(mask&observed&np.isfinite(values))[ids]]
            center=int(ids[np.argmin(np.abs(values[ids]-event['value']))])
            ref=int(reference[center]) if reference is not None and reference[center]>=0 else int(base_ids[len(base_ids)//2])
            event.update(label=LABELS[kind],view='front' if front else 'side',level=1,
                         interpretation='Observable coordination cue; anatomical cause unconfirmed',
                         unit=unit,threshold=float(limit),persistence_s=float(duration),
                         frame_1based=center+1,center_s=float(t[center]),
                         reference_frame_1based=ref+1,reference_s=float(t[ref]),
                         hip_progress=float(progress[center]),
                         trunk_projection_ratio=float(projection[center]) if front else None,
                         knee_progress=float(knee_progress[center]) if np.isfinite(knee_progress[center]) else None,
                         shin_projection_ratio=float(shin_ratio[center]),
                         leg_match_error_leg_ratio=float(distances[center]) if distances is not None and np.isfinite(distances[center]) else None)
            event['start_s']=max(start,event['start_s'])
            event['stop_s']=min(stop,event['stop_s'])
            result.update(status='WARN',value=event['value'],event=event)
            events.append(event)
        checks[kind]=result

    phase_ready=front or knee_excursion>=cfg['joint_onset_min_excursion_deg']
    phase_mask=near_top&recovered&folded
    add('early_trunk_fold',fold,phase_mask&(t<bottom),threshold,available=phase_ready)
    add('late_trunk_recovery',fold,phase_mask&(t>bottom),threshold,available=phase_ready)
    tolerance=cfg['indirect_leg_match_tolerance']
    before=np.flatnonzero(observed&(t<bottom))
    after=np.flatnonzero(observed&(t>bottom))
    matched=np.full(len(t),np.nan);ref_ids=np.full(len(t),-1,int);distances=np.full(len(t),np.nan)
    for i in after:
        refs=before[(block[before]==block[i])&(np.abs(progress[before]-progress[i])<=cfg['indirect_matched_progress_tolerance'])]
        if not len(refs):continue
        distance=np.maximum(np.linalg.norm(hip_relative[refs]-hip_relative[i],axis=1),
                            np.linalg.norm(knee_relative[refs]-knee_relative[i],axis=1))/scale
        at=int(np.argmin(distance))
        if distance[at]<=tolerance:
            ref=int(refs[at]);matched[i]=q[i]-q[ref];ref_ids[i]=ref;distances[i]=distance[at]
    add('ascent_extra_fold',matched,np.isfinite(matched),change_threshold,ref_ids,distances)
    bottom_change=np.full(len(t),np.nan);bottom_refs=np.full(len(t),-1,int);bottom_distance=np.full(len(t),np.nan)
    near_bottom=observed&(progress>=cfg['indirect_bottom_progress'])
    near_ids=np.flatnonzero(near_bottom)
    horizon=cfg['indirect_bottom_comparison_fraction']*(stop-start)
    for i in near_ids:
        refs=near_ids[(t[near_ids]<=t[i]-duration)&(t[near_ids]>=t[i]-horizon)&(block[near_ids]==block[i])]
        if not len(refs):continue
        distance=np.maximum(np.linalg.norm(hip_relative[refs]-hip_relative[i],axis=1),
                            np.linalg.norm(knee_relative[refs]-knee_relative[i],axis=1))/scale
        refs,dist=refs[distance<=tolerance],distance[distance<=tolerance]
        if not len(refs):continue
        changes=q[i]-q[refs];best=int(np.argmax(changes));ref=int(refs[best])
        bottom_change[i]=changes[best];bottom_refs[i]=ref;bottom_distance[i]=dist[best]
    add('bottom_extra_fold',bottom_change,near_bottom,change_threshold,bottom_refs,bottom_distance)
    result=blank.copy()
    result.update(indirect_assessment_available=1,indirect_fault_level=int(bool(events)),
                  indirect_cue_count=len(events),indirect_cues_json=json.dumps(events,allow_nan=False),
                  indirect_checks_json=json.dumps(checks,allow_nan=False),
                  indirect_early_fold_value=checks['early_trunk_fold']['value'],
                  indirect_late_fold_value=checks['late_trunk_recovery']['value'],
                  indirect_bottom_extra_fold_value=checks['bottom_extra_fold']['value'],
                  indirect_ascent_extra_fold_value=checks['ascent_extra_fold']['value'])
    return result
