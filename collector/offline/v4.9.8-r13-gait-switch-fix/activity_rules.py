import numpy as np
from scipy.signal import find_peaks, savgol_filter

def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))

def smooth_valid(t, values, valid, seconds):
    out = np.full_like(values, np.nan, dtype=float)
    dt = float(np.median(np.diff(t)))
    blocks = []
    for lo, hi in runs(valid):
        cuts = np.flatnonzero(np.diff(t[lo:hi]) > max(.25, 3 * dt)) + lo + 1
        bounds = [lo] + cuts.tolist() + [hi]
        blocks.extend(zip(bounds[:-1], bounds[1:]))
    for lo, hi in blocks:
        if hi - lo < 3:
            out[lo:hi] = values[lo:hi]
            continue
        grid = np.linspace(t[lo], t[hi-1], max(3, round((t[hi-1]-t[lo])/dt)+1))
        window = max(3, round(seconds/(grid[1]-grid[0])) | 1)
        window = min(window, len(grid) if len(grid) % 2 else len(grid)-1)
        original = values[lo:hi].reshape(hi-lo, -1)
        result = np.empty_like(original)
        for j in range(original.shape[1]):
            v = np.interp(grid, t[lo:hi], original[:, j])
            v = savgol_filter(v, window, min(2, window-1))
            result[:, j] = np.interp(t[lo:hi], grid, v)
        out[lo:hi] = result.reshape(values[lo:hi].shape)
    return out, blocks

def derivative(t, values, blocks):
    out = np.full_like(values, np.nan, dtype=float)
    for a, b in blocks:
        if b-a >= 3:
            out[a:b] = np.gradient(values[a:b], t[a:b], axis=0)
    return out

def features(data, args):
    t = np.asarray(data['time_s'], dtype=float)
    k = np.asarray(data['keypoints'], dtype=float)
    if t.ndim != 1 or len(t) < 3 or np.any(~np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError('Cached PTS must be finite and strictly increasing.')
    if k.shape != (len(t), 17, 3):
        raise ValueError('Expected COCO keypoints with shape (frames, 17, 3).')
    geometry_side = getattr(args, '_geometry_side', '')
    side_indices = {'left':(11,13,15),'right':(12,14,16)}
    ids = list(side_indices[geometry_side]) if geometry_side else [11, 12, 13, 14, 15, 16]
    valid = np.isfinite(k[:, ids]).all(axis=(1, 2)) & (k[:, ids, 2] >= args.keypoint_conf).all(axis=1)
    legs = []
    for h, n, a in ([side_indices[geometry_side]] if geometry_side else [(11, 13, 15), (12, 14, 16)]):
        length = np.linalg.norm(k[:, h, :2]-k[:, n, :2], axis=1) + np.linalg.norm(k[:, n, :2]-k[:, a, :2], axis=1)
        limb_valid = np.isfinite(k[:,[h,n,a]]).all(axis=(1,2)) & (k[:,[h,n,a],2]>=args.keypoint_conf).all(axis=1)
        legs.append(length[limb_valid])
    available = np.concatenate(legs)
    scale = float(np.median(available)) if len(available) else np.nan
    if not np.isfinite(scale) or scale <= 1:
        raise ValueError('Not enough visible lower-body points to estimate scale.')
    hip = k[:, [side_indices[geometry_side][0]] if geometry_side else [11,12], :2].mean(axis=1)
    ankles = k[:, [side_indices[geometry_side][2]] if geometry_side else [15,16], :2]
    height, blocks = smooth_valid(t, hip[:, 1]-ankles[:, :, 1].mean(axis=1), valid, args.smooth_seconds)
    hip_s, _ = smooth_valid(t, hip, valid, args.smooth_seconds)
    ankle_s, _ = smooth_valid(t, ankles, valid, args.smooth_seconds)
    hip_velocity = derivative(t, hip_s, blocks)/scale
    ankle_velocity = derivative(t, ankle_s, blocks)/scale
    return dict(t=t, k=k, valid=valid, scale=scale, hip=hip_s, ankles=ankle_s,
                height=height, blocks=blocks, hip_velocity=hip_velocity,
                ankle_speed=np.linalg.norm(ankle_velocity, axis=2).max(axis=1),geometry_side=geometry_side,
                bilateral_valid=np.isfinite(k[:,[13,14,15,16]]).all(axis=(1,2)) & (k[:,[13,14,15,16],2]>=args.keypoint_conf).all(axis=1))

def detect_jumps(f, args):
    t, k, scale = f['t'], f['k'], f['scale']
    dt = float(np.median(np.diff(t)))
    velocity = f['hip_velocity'][:, 1]
    jumps = []
    for lo, hi in f['blocks']:
        if hi-lo < 5:
            continue
        peaks, _ = find_peaks(-velocity[lo:hi], height=args.jump_rise_speed,
                              distance=max(1, round(.6/dt)), prominence=.4)
        for p in peaks+lo:
            a = max(lo, int(np.searchsorted(t, t[p]-.6)))
            b = min(hi, int(np.searchsorted(t, t[p]+.6, side='right')))
            if not f.get('bilateral_valid',f['valid'])[a:b].all():
                continue
            floor = np.percentile(k[a:b, [15,16], 1], 80, axis=0)
            lift = float(np.min(floor-k[a:b, [15,16], 1], axis=1).max()/scale)
            if lift < args.jump_ankle_lift:
                continue
            quiet = np.abs(velocity) <= .3
            holds = [(x,y) for x,y in runs(quiet[lo:hi])
                     if t[lo+y-1]-t[lo+x]+dt >= .1]
            holds = [(x+lo,y+lo) for x,y in holds]
            before = [(x,y) for x,y in holds if y <= p and t[p]-t[y-1] <= 1.3]

            fall_stop = min(hi, int(np.searchsorted(t,t[p]+1.0,side='right')))
            fall_peak = p + int(np.argmax(velocity[p:fall_stop]))
            after = [(x,y) for x,y in holds if x > fall_peak and t[x]-t[p] <= 1.3]
            start = before[-1][1] if before else max(lo, np.searchsorted(t, t[p]-.45))
            end = after[0][0] if after else min(hi-1, np.searchsorted(t, t[p]+.65))
            if end <= start:
                continue
            jumps.append(dict(start=int(start), end=int(end), peak=int(p),
                              upward_hip_speed=float(-velocity[p]), bilateral_ankle_lift_ratio=lift))
    return jumps

def crossing(t, signal, a, p, b, fraction):
    left = signal[a] + fraction*(signal[p]-signal[a])
    right = signal[b] + fraction*(signal[p]-signal[b])
    aa = np.flatnonzero(signal[a:p+1] <= left)
    bb = np.flatnonzero(signal[p:b+1] <= right)
    if not len(aa) or not len(bb):
        return None
    return int(a+aa[-1]+1), int(p+bb[0])

def label_hip_poses(data, args):
    f = features(data, args)
    if not f['valid'].any():
        raise ValueError('Hip mode needs both hips, knees and ankles visible. Consider knee mode for a side view.')
    t, k, height, scale = f['t'], f['k'], f['height'], f['scale']
    dt = float(np.median(np.diff(t)))
    ends = np.minimum(np.r_[t[1:], t[-1]+dt], t+max(2*dt,.05))
    labels = np.full(len(t), -1, dtype=int)
    reps = np.zeros(len(t), dtype=int)
    events, rejected = [], []
    jumps = detect_jumps(f, args)
    candidate_id = 0
    for lo, hi in f['blocks']:
        if hi-lo < 5:
            continue
        peaks, _ = find_peaks(height[lo:hi], prominence=args.hip_prominence*scale,
                              distance=max(1, round(args.min_peak_separation/dt)))
        troughs, _ = find_peaks(-height[lo:hi], prominence=.01*scale,
                                distance=max(1, round(.15/dt)))
        troughs = np.r_[lo, troughs+lo, hi-1]
        for p in peaks+lo:
            candidate_id += 1
            before = troughs[(troughs<p)&(height[troughs] <= height[p]-args.hip_prominence*scale)]
            after = troughs[(troughs>p)&(height[troughs] <= height[p]-args.hip_prominence*scale)]
            row = dict(candidate_id=candidate_id, bottom_video_s=float(t[p]), bottom_video_frame=int(p+1))
            if not len(before) or not len(after):
                rejected.append(dict(row, reason='incomplete_height_cycle'))
                continue
            a,b = int(before[-1]), int(after[0])
            pair = crossing(t,height,a,p,b,args.boundary_fraction)
            if pair is None:
                rejected.append(dict(row, reason='no_boundary_crossing'))
                continue
            start,end = pair

            support_a = a + np.flatnonzero(height[a:p+1] <= height[a]+.01*scale)[-1]
            support_b = p + np.flatnonzero(height[p:b+1] <= height[b]+.01*scale)[0]
            foot_ids = [15 if f.get('geometry_side')=='left' else 16] if f.get('geometry_side') else [15,16]
            ankle_motion = float(np.max(np.ptp(k[support_a:support_b+1, foot_ids, :2], axis=0))/scale)
            jump = next((j for j in jumps if j['start'] <= p < j['end'] or
                         (start < j['end'] and end > j['start'])), None)
            reason = None
            if jump is not None:
                reason = 'jump_like_bilateral_lift_and_fast_rise'
            elif ankle_motion > args.max_ankle_motion:
                reason = 'excessive_foot_motion_or_locomotion'
            elif not args.min_duration <= t[end]-t[start] <= args.max_duration:
                reason = 'duration_out_of_range'
            elif p-a < 2 or b-p < 2 or start <= a or end >= hi:
                reason = 'incomplete_height_cycle'
            row.update(start_video_frame=int(start+1), end_video_frame=int(end),
                       start_video_s=float(t[start]), end_video_s=float(t[end]), ankle_motion_leg_ratio=ankle_motion)
            if reason:
                if jump:
                    row.update(start_video_frame=jump['start']+1, end_video_frame=jump['end'],
                               start_video_s=float(t[jump['start']]), end_video_s=float(t[jump['end']]),
                               upward_hip_speed_leg_s=jump['upward_hip_speed'],
                               bilateral_ankle_lift_ratio=jump['bilateral_ankle_lift_ratio'])
                rejected.append(dict(row, reason=reason))
                continue
            labels[a:b+1] = 0
            alternative = [crossing(t,height,a,p,b,q) for q in (.05,.15)]
            event = dict(rep_id=0, candidate_id=candidate_id, start_video_frame=start+1,end_video_frame=end,
                         end_exclusive_video_frame=end+1,start_video_s=float(t[start]),end_video_s=float(t[end]),
                         bottom_video_s=float(t[p]),bottom_video_frame=int(p+1),duration_s=float(t[end]-t[start]),
                         knee_excursion_deg='',hip_drop_leg_ratio=float(min(height[p]-height[a],height[p]-height[b])/scale),
                         ankle_motion_leg_ratio=ankle_motion,
                         start_sensitivity_first_frame=min(x[0]+1 for x in alternative),
                         start_sensitivity_last_frame=max(x[0]+1 for x in alternative),
                         end_sensitivity_first_frame=min(x[1] for x in alternative),
                         end_sensitivity_last_frame=max(x[1] for x in alternative),reviewed=0)
            events.append(event)
    events.sort(key=lambda e:e['start_video_s'])
    for number,event in enumerate(events,1):
        event['rep_id'] = number
        a,b = event['start_video_frame']-1,event['end_video_frame']
        if reps[a:b].any():
            raise ValueError('Overlapping squat cycles; inspect pose cache.')
        labels[a:b],reps[a:b] = 1,number

    quiet = f['valid'] & (np.abs(f['hip_velocity'][:,1]) < .2) & (f['ankle_speed'] < .25)
    for a,b in runs(quiet):
        if t[b-1]-t[a]+dt >= args.standing_hold:
            labels[a:b] = np.where(reps[a:b]>0,1,0)
    frames=[]
    for i in range(len(t)):
        reason = 'automatic_squat' if labels[i]==1 else 'standing_rule' if labels[i]==0 else 'uncertain_or_other_motion'
        if not f['valid'][i]:
            reason = 'missing_or_low_confidence_lower_body'
        frames.append(dict(video_frame=i+1,video_time_s=float(t[i]),frame_end_s=float(ends[i]),
                           pose_valid=int(f['valid'][i]),knee_angle_deg='',knee_angle_smoothed_deg='',
                           hip_ankle_height_leg_ratio=float(height[i]/scale),
                           squat_label=int(labels[i]),rep_id=int(reps[i]),reviewed=0,reason=reason))
    return events,frames,rejected,'bilateral_hip_height'

def classify_activities(data, frames, events, rejected, args):
    f = features(data,args)
    t,k,scale=f['t'],f['k'],f['scale']
    dt=float(np.median(np.diff(t)))
    activity=np.array(['unknown']*len(t),dtype=object)
    reasons=np.array(['insufficient_motion_evidence']*len(t),dtype=object)
    for i,row in enumerate(frames):
        if row['squat_label']==0:
            activity[i],reasons[i]='standing','automatic_standing_rule'
        elif row['squat_label']==1:
            activity[i],reasons[i]='squat',('torso_lower_body_joint_boundary' if args.boundary_mode=='body' else 'hip_cycle_with_stable_feet' if args.boundary_mode=='hip' else 'knee_cycle')
    jumps=detect_jumps(f,args)
    for jump in jumps:
        a,b=jump['start'],jump['end']
        if np.any(activity[a:b]=='squat'):
            raise ValueError('A detected jump overlaps a squat. Review the rules.')
        activity[a:b],reasons[a:b]='jump','fast_upward_hip_and_bilateral_ankle_lift'
    torso_ids=[5,6,11,12]
    torso_valid=np.isfinite(k[:,torso_ids]).all(axis=(1,2)) & (k[:,torso_ids,2]>=args.keypoint_conf).all(axis=1)
    shoulders=k[:,[5,6],:2].mean(axis=1)
    hips=k[:,[11,12],:2].mean(axis=1)
    torso=np.linalg.norm(shoulders-hips,axis=1)
    shoulder_width=np.linalg.norm(k[:,5,:2]-k[:,6,:2],axis=1)
    vectors=np.c_[shoulders[:,0]/scale,np.log(np.maximum(torso,1)),np.log(np.maximum(shoulder_width,1))]
    sm,blocks=smooth_valid(t,vectors,torso_valid,args.smooth_seconds)
    velocity=derivative(t,sm,blocks)
    coherent_scale=(velocity[:,1]*velocity[:,2]>0)&(np.minimum(abs(velocity[:,1]),abs(velocity[:,2]))>.12)
    visible_locomotion=(abs(velocity[:,0])>args.walk_translation_speed)&f['valid']&(f['ankle_speed']>.2)
    partial_view_motion=(~f['valid'])&(abs(velocity[:,0])>.55)&(abs(velocity[:,1])>.12)
    moving=torso_valid & (visible_locomotion|partial_view_motion|coherent_scale)
    protected=(activity=='squat')|(activity=='jump')

    for event in events:
        a,b=event['start_video_s']-.35,event['end_video_s']+.35
        protected |= (t>=a)&(t<=b)
    moving &= ~protected

    for a,b in runs(~moving):
        if a>0 and b<len(t) and t[b]-t[a] <= .22 and not protected[a:b].any():
            moving[a:b]=True
    for a,b in runs(moving):
        if t[b-1]-t[a]+dt < .35:
            continue
        torso_change=max(abs(sm[b-1,1]-sm[a,1]),abs(sm[b-1,0]-sm[a,0]))
        foot_evidence=f['valid'][a:b] & (f['ankle_speed'][a:b]>.25)
        if torso_change < .12 or foot_evidence.sum()*dt < .10:
            continue
        activity[a:b]='walking'
        reasons[a:b]='torso_translation_with_visible_step_motion;automatic_candidate'

    for a,b in runs(activity=='unknown'):
        if a>0 and b<len(t) and t[b]-t[a] <= .12 and activity[a-1]==activity[b] and activity[a-1] in ('standing','walking'):
            activity[a:b]=activity[a-1]
            reasons[a:b]='brief_gap_between_same_activity'
    for i,row in enumerate(frames):
        row.update(activity=str(activity[i]),activity_reason=str(reasons[i]),activity_reviewed=0)
        if activity[i] in ('jump','walking'):
            row['squat_label'],row['rep_id'],row['reason']=-1,0,'other_activity_'+activity[i]
    segments=[]
    a=0
    for b in range(1,len(t)+1):
        if b<len(t) and activity[b]==activity[a] and frames[b]['rep_id']==frames[a]['rep_id']:
            continue
        segments.append(dict(segment_id=len(segments)+1,activity=str(activity[a]),rep_id=frames[a]['rep_id'],
                             start_video_frame=a+1,end_video_frame=b,start_video_s=float(t[a]),
                             end_video_s=float(t[b]) if b<len(t) else frames[-1]['frame_end_s'],
                             duration_s=float((t[b] if b<len(t) else frames[-1]['frame_end_s'])-t[a]),
                             lower_body_visible_fraction=float(f['valid'][a:b].mean()),reviewed=0,
                             status='automatic_candidate'))
        a=b
    return segments
