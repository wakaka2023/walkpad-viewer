from collections import deque
from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np
from .activity_rules import runs


def band_edge(image):
    h,w=image.shape[:2]
    if h<100 or w<100:
        return 0
    lo=max(9,round(.023*h)); hi=round(.18*h)
    x=np.linspace(.015*w,.985*(w-1),min(w,240)).astype(int)
    sample=image[:hi+9,x].astype(np.int16)
    ref=np.rint(np.median(sample[:3],axis=(0,1))).astype(np.int16)
    flat=(np.max(np.abs(sample-ref),axis=2)<=13).mean(axis=1)
    if np.min(flat[:3])<.94:
        return 0
    for i in range(lo,hi+1):
        if (flat[i]<=.72 and np.mean(flat[i:i+7])<=.24 and np.median(flat[i-4:i])>=.9
                and np.mean(flat[:i]>=.67)>=.92):
            return i
    return 0


def stable_top_band(images, height):
    edges=[band_edge(image) for image in images]
    positive=[x for x in edges if x]
    if len(edges)<3 or len(positive)<max(3,int(np.ceil(.8*len(edges)))):
        return 0
    center=int(np.median(positive))
    limit=max(2,round(.005*height))
    if max(abs(v-center) for v in positive)>limit:
        return 0
    return center


def content_top(source,data):
    from .multi_track import oriented_image
    import av
    t=data['time_s']; height=int(data['height'])
    if len(t)<12:
        return 0
    samples=np.unique(np.linspace(0,len(t)-1,5,dtype=int))
    def images():
        with av.open(str(source)) as reader:
            stream=reader.streams.video[0]
            tick=float(stream.time_base)
            origin=float(data['first_pts_seconds'])
            for idx in samples:
                target=origin+t[idx]
                reader.seek(round(target/tick),stream=stream,backward=True)
                for frame in reader.decode(stream):
                    if frame.pts is not None and float(frame.pts*frame.time_base)>=target-max(tick,1/90000):
                        yield oriented_image(frame,int(data['rotate']))
                        break
    return stable_top_band(images(),height)


def cropped_track(data, rows, conf, content_top=0):
    if len(rows) < 12:
        return False
    b = data['boxes'][rows]
    k = data['keypoints'][rows,:,2]
    height = float(data['height'])
    content_h=height-content_top
    cut = b[:,1] <= content_top+.035*content_h
    lower = (np.nanmax(k[:,13:17],axis=1) >= conf)
    upper = (np.nanmax(k[:,[0,1,2,3,4,5,6]],axis=1) >= conf)
    close = b[:,3]-np.maximum(b[:,1],content_top) >= .44*content_h
    return bool((cut & close).mean() >= .78 and lower.mean() >= .68 and upper.mean() <= .08)


def deduplicate(data, conf, content_top=0):
    groups = {int(x): np.flatnonzero(data['track_id']==x) for x in np.unique(data['track_id'])}
    modes = {identity for identity,rows in groups.items() if cropped_track(data,rows,conf) or (content_top and cropped_track(data,rows,conf,content_top))}
    discarded = []
    h = float(data['height'])-content_top
    for primary in sorted(modes,key=lambda x:-len(groups[x])):
        if primary in discarded:
            continue
        main = groups[primary]
        for other,rows in groups.items():
            if other == primary or other in discarded or len(rows) >= len(main)/2:
                continue
            k = data['keypoints'][rows,:,2]
            if (np.nanmax(k[:,[0,1,2,3,4,5,6]],axis=1) >= conf).any():
                continue
            if ((data['boxes'][rows,1] <= content_top+.04*h) & ((data['boxes'][rows,3]-np.maximum(data['boxes'][rows,1],content_top)) >= .44*h)).mean() < .75:
                continue
            _, a,b = np.intersect1d(data['frame_index'][main],data['frame_index'][rows],return_indices=True)
            if len(a) < max(10,.55*len(rows)):
                continue
            A,B=data['boxes'][main[a]],data['boxes'][rows[b]]
            overlap = np.prod(np.maximum(0,np.minimum(A[:,2:],B[:,2:])-np.maximum(A[:,:2],B[:,:2])),axis=1)
            ios=overlap/np.minimum(np.prod(A[:,2:]-A[:,:2],axis=1),np.prod(B[:,2:]-B[:,:2],axis=1))
            left,right=data['keypoints'][main[a],13:17],data['keypoints'][rows[b],13:17]
            valid=(left[:,:,2]>=conf)&(right[:,:,2]>=conf)
            near=np.linalg.norm(left[:,:,:2]-right[:,:,:2],axis=2) <= .15*np.minimum(A[:,3]-A[:,1],B[:,3]-B[:,1])[:,None]
            paired=(valid & near).sum(axis=1) >= 2
            if np.quantile(ios,.2) < .60 or paired.sum() < max(4,.22*len(a)):
                continue
            outside = np.logical_or((right[:,:,:2] < A[:,None,:2]).any(axis=2),(right[:,:,:2] > A[:,None,2:]).any(axis=2))
            if np.sum(((right[:,:,2]>=conf)&outside).sum(axis=1)>=2) > .18*len(a):
                continue
            discarded.append(other)
    keep = ~np.isin(data['track_id'],discarded)
    cleaned = dict(data)
    for name in ('frame_index','track_id','boxes','confidence','keypoints','yolo_track_id'):
        if name in data:
            cleaned[name]=data[name][keep]
    return cleaned,modes.difference(discarded),discarded


def track_boxes(data, identity):
    rows=np.flatnonzero(data['track_id']==identity)
    idx=data['frame_index'][rows]
    coord=data['boxes'][rows]
    n=len(data['time_s'])
    boxes=np.full((n,4),np.nan)
    boxes[idx]=coord
    maxgap=max(1,round(.35/np.median(np.diff(data['time_s']))))
    for i in np.flatnonzero(np.diff(idx)>1):
        a,b=idx[i:i+2]
        if b-a <= maxgap:
            r=np.arange(1,b-a)[:,None]/(b-a)
            boxes[a+1:b]=(1-r)*coord[i]+r*coord[i+1]
    pose=np.full((n,17,3),np.nan)
    pose[idx]=data['keypoints'][rows]
    return boxes,pose


def background_mask(shape, boxes, content_top=0):
    mask=np.full(shape,255,np.uint8)
    mask[:max(shape[0]//8,content_top)]=0
    for b in boxes:
        if not np.isfinite(b).all():
            continue
        margin=.018*(b[3]-b[1]); (x1,y1),(x2,y2)=np.round(np.array([b[:2]-margin,b[2:]+margin])).astype(int)
        mask[max(0,y1):y2+1,max(0,x1):x2+1]=0
    return mask


def feature_tracks(previous, current, cv2, background, shoe_regions):
    fg=cv2.goodFeaturesToTrack(previous,maxCorners=1100,qualityLevel=.004,minDistance=4,blockSize=5)
    bg=cv2.goodFeaturesToTrack(previous,maxCorners=650,qualityLevel=.003,minDistance=4,blockSize=5,mask=background)
    fine=[]
    for center, radius in shoe_regions:
        mask = np.zeros(previous.shape,np.uint8)
        cv2.circle(mask,tuple(np.rint(center).astype(int)),int(np.ceil(radius)),255,-1)
        picks=cv2.goodFeaturesToTrack(previous,maxCorners=24,qualityLevel=.012,minDistance=2,blockSize=3,mask=mask)
        if picks is not None:
            fine.append(picks)
    groups=[v for v in (fg,bg,*fine) if v is not None]
    p=np.unique(np.concatenate(groups),axis=0) if groups else None
    if p is None:
        return np.empty((0,2)), np.empty((0,2))
    q,s,_=cv2.calcOpticalFlowPyrLK(previous,current,p,None,winSize=(19,19),maxLevel=3)
    if q is None:
        return np.empty((0,2)), np.empty((0,2))
    r,t,_=cv2.calcOpticalFlowPyrLK(current,previous,q,None,winSize=(19,19),maxLevel=3)
    if r is None:
        return np.empty((0,2)), np.empty((0,2))
    p,q,r=p[:,0],q[:,0],r[:,0]
    valid=s.ravel().astype(bool)&t.ravel().astype(bool)&(np.linalg.norm(r-p,axis=1)<=1.)
    return p[valid],q[valid]


def floor_registration(p, q, before, after, cv2):
    if len(p)<20:
        return
    h,w=before.shape
    x,y=np.clip(np.rint(p).astype(int),[0,0],[w-1,h-1]).T
    u,v=np.clip(np.rint(q).astype(int),[0,0],[w-1,h-1]).T
    outside=(before[y,x]>0)&(after[v,u]>0)
    if outside.sum()<20:
        return
    r,s=p[outside],q[outside]
    mat,good=cv2.estimateAffine2D(r,s,method=cv2.RANSAC,ransacReprojThreshold=1.5,maxIters=700)
    if mat is None or good.mean()<.58 or good.sum()<14:
        return
    good=good.ravel().astype(bool)
    residual=np.linalg.norm(s-r@mat[:,:2].T-mat[:,2],axis=1)
    noise=max(np.quantile(residual[good],.90),np.finfo(np.float32).eps*np.max(abs(p)))
    return mat,noise,outside


def flow_pairs(source, data, t, ids, others, poses, ankle_conf, factor, av, cv2):
    from .multi_track import oriented_image
    n=len(t)
    previous=None
    with av.open(str(source)) as reader:
        first = None
        for i,frame in enumerate(reader.decode(video=0)):
            pts = frame.pts * frame.time_base
            first = pts if first is None else first
            if i >= n or abs(float(pts-first)-t[i]) > 1/90000:
                raise ValueError('Source PTS differs from the close-view pose cache.')
            image=oriented_image(frame,int(data['rotate']))
            gray=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
            boxes=np.array([b[i]*factor for b in others.values()])
            mask=background_mask(gray.shape, boxes, int(np.ceil(data.get('cropped_content_top_px',0)*factor)))
            if previous is not None:
                prev,old=previous
                regions=[]
                for j in ids:
                    b=others[j][i-1]*factor
                    if not np.isfinite(b).all(): continue
                    unit=b[3]-b[1]
                    for ankle in poses[j][i-1,15:17]:
                        if np.isfinite(ankle).all() and ankle[2] >= ankle_conf[j]:
                            regions.append((ankle[:2]*factor+[0.,.055*unit],.087*unit))
                yield i,prev,gray,old,mask,regions
            previous=(gray,mask)
    if first is None or i+1 != n:
        raise ValueError('Source frame count differs from the close-view pose cache.')


def tracked_flow(pairs, cv2, workers):
    if workers == 1:
        for i,prev,gray,old,mask,regions in pairs:
            p,q=feature_tracks(prev,gray,cv2,old,regions)
            yield i,p,q,old,mask
        return
    pending=deque()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,prev,gray,old,mask,regions in pairs:
            pending.append((i,old,mask,pool.submit(feature_tracks,prev,gray,cv2,old,regions)))
            if len(pending) >= 2*workers:
                j,before,after,job=pending.popleft()
                p,q=job.result()
                yield j,p,q,before,after
        while pending:
            j,before,after,job=pending.popleft()
            p,q=job.result()
            yield j,p,q,before,after


def collect(source, data, modes, conf, ankle_conf=None):
    import av,cv2
    t=data['time_s']; n=len(t)
    ids=sorted(modes)
    ankle_conf={j: (ankle_conf or {}).get(j,conf) for j in ids}
    others={int(x):track_boxes(data,int(x))[0] for x in np.unique(data['track_id'])}
    measurements={j:np.full((n,10),np.nan) for j in ids}
    poses={j:track_boxes(data,j)[1] for j in ids}
    cv2.setRNGSeed(0)
    factor=min(1.,480/float(data['width']))
    available=len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    workers=min(4, max(1, available//max(1,cv2.getNumThreads())))
    pairs=flow_pairs(source,data,t,ids,others,poses,ankle_conf,factor,av,cv2)
    for i,p,q,old,mask in tracked_flow(pairs,cv2,workers):
        fit=floor_registration(p,q,old,mask,cv2)
        if fit is not None:
            mat,noise,outside=fit
            pred=p@mat[:,:2].T+mat[:,2]; err=np.linalg.norm(q-pred,axis=1)
            for j in ids:
                b=others[j][i-1]*factor
                if not np.isfinite(b).all():continue
                unit=b[3]-b[1]; dt=t[i]-t[i-1]
                floor=np.array([.5*(b[0]+b[2]),.83*b[3]+.17*b[1]])
                delta=floor@mat[:,:2].T+mat[:,2]-floor
                local=(p[:,0]>=b[0])&(p[:,0]<=b[2])&(p[:,1]>=b[1]+.50*unit)&(p[:,1]<=b[3])
                values=measurements[j][i]
                values[:3]=[*delta/(unit*dt),noise/(unit*dt)]
                if local.sum()>=8:
                    values[3:5]=np.quantile(err[local],[.75,.9])/(unit*dt)
                for side,ank in enumerate(poses[j][i-1,15:17]):
                    if not np.isfinite(ank).all() or ank[2]<ankle_conf[j]:
                        continue
                    center=ank[:2]*factor+np.array([0.,.055*unit]); good=np.linalg.norm(p-center,axis=1)<.095*unit
                    good &= (p[:,1]>=ank[1]*factor-.025*unit)
                    if good.sum()>=4:
                        values[5+side]=np.quantile(err[good],.5)/(unit*dt)
                ankle = poses[j][i-1,15:17] * np.array([factor,factor,1.])
                if np.isfinite(ankle).all() and (ankle[:,2]>=ankle_conf[j]).all():
                    values[7]=np.linalg.norm(ankle[0,:2]-ankle[1,:2])/unit
                values[8:]=[b[0]/mask.shape[1],b[3]/mask.shape[0]]
    return measurements


def own_stride(t, values):
    from scipy.signal import correlate, find_peaks
    from scipy.ndimage import gaussian_filter1d
    dt = float(np.median(np.diff(t)))
    count = int(round((t[-1]-t[0])/dt))+1
    limit=min(count//3,int(round(2.0/dt)))
    if limit<9:
        return None
    grid=np.linspace(t[0],t[-1],count)
    curves=[]
    for col in (4,8,9):
        x=values[:,col]; good=np.isfinite(x)
        if good.mean()<.72:
            continue
        v=np.interp(grid,t[good],x[good]); v -= gaussian_filter1d(v,max(1,.6/dt),mode='nearest')
        v -= v.mean()
        norm=np.dot(v,v)
        if norm<=np.finfo(float).eps:
            continue
        raw=correlate(v,v,method='fft')[len(v)-1:len(v)+limit]/norm
        curves.append(raw)
    if len(curves)<2:
        return None
    low=max(3,round(.19/dt))
    agreement=np.mean(np.maximum(0,curves),axis=0)
    peaks=find_peaks(agreement,prominence=.10)[0]
    peaks=peaks[(peaks>=low)&(agreement[peaks]>=.21)]
    if not len(peaks):
        return None
    top=np.max(agreement[peaks]); good=peaks[agreement[peaks]>=.68*top]
    return float(good[0]*dt)


def classify(t, signals):
    from scipy.ndimage import median_filter
    n=len(t); label=np.full(n,'',object); direct=np.zeros(n,bool)
    evidence=np.full(n,'',object); step=np.full(n,np.nan)
    background=signals[:,2]
    energy=signals[:,4]
    magnitude=np.linalg.norm(signals[:,:2],axis=1)
    valid=np.isfinite(energy)&np.isfinite(background)
    dt=float(np.median(np.diff(t)))
    if valid.sum()<24:
        return label,direct,evidence,step
    feet=signals[:,5:7]
    both=np.isfinite(feet).all(axis=1)
    sep=signals[:,7]
    distinct=sep > .5*np.nanmedian(sep[both]) if both.sum()>=8 else np.zeros(n,bool)
    planted=np.isfinite(feet)&(feet<=3.2*background[:,None])
    double=planted.all(axis=1)&distinct
    reference=valid & double & (magnitude < np.quantile(magnitude[valid],.38))
    if reference.sum()>=8:
        foot_noise=np.median(energy[reference])
        floor_noise=np.quantile(magnitude[reference],.88)
        scale_error=np.quantile(energy[reference],.9) - foot_noise
        leg_quiet=energy <= np.maximum(3.2*background,foot_noise+2.5*scale_error)
    else:
        floor_noise=np.quantile(background[valid],.85)*3.2
        leg_quiet=energy <= 3.2*background
    for a,b in runs(valid & leg_quiet & double):
        if b-a>=max(3,round(.13/dt)):
            label[a:b]='standing'; evidence[a:b]='cropped_ground_quiet'; direct[a:b]=True
    moving=valid & (magnitude>np.maximum(floor_noise,3.2*background))
    nonstill=valid & ~leg_quiet
    rough=median_filter(moving.astype(np.uint8),size=max(3,round(.32/dt)|1),mode='nearest').astype(bool)
    for a,b in runs(~rough):
        if a>0 and b<n and (b-a)*dt<=.55 and not (label[a:b]=='standing').any():
            rough[a:b]=True
    for a,b in runs(rough):
        if np.mean(nonstill[a:b]) < .34:
            continue
        period=own_stride(t[a:b],signals[a:b])
        if period is None:
            continue
        while a>0 and nonstill[a-1] and label[a-1]!='standing':
            a-=1
        while b<n and nonstill[b] and label[b]!='standing':
            b+=1
        active=np.flatnonzero(nonstill[a:b])+a
        if len(active)<2:
            continue
        a,b=active[0],active[-1]+1
        span=t[b-1]-t[a]
        if span<3*period:
            continue
        vec=signals[a:b,:2]; good=np.isfinite(vec).all(axis=1)
        denominator=np.sum(magnitude[a:b][good]); net=np.linalg.norm(np.sum(vec[good],axis=0))
        if denominator <= 0 or net/denominator<.62:
            continue
        center=(t >= t[a]+.65*period)&(t<=t[b-1]-.65*period)
        observed_double=double & center & moving
        anchors=np.flatnonzero(observed_double)
        anchor_times=t[anchors]; support_both=planted[a:b].sum(axis=0)
        stance_counts=np.array([sum(1 for l,r in runs(planted[a:b,k]) if r-l>=2) for k in (0,1)])
        extended=anchor_times.size>=2 and anchor_times[-1]-anchor_times[0]>=2*period and np.min(stance_counts)>=3
        chunks = np.arange(a,b,max(1,round(2.8*period/dt)))
        continuing=all(planted[l:min(l+round(2.8*period/dt),b)].any() for l in chunks
                       if (b-l)*dt>period)
        if not continuing or np.min(support_both)<3:
            extended=False
        if extended and b < n and reference.any():
            swing_level=np.quantile(energy[active],.30)
            strong=active[energy[active] >= swing_level]
            soft_level=np.quantile(energy[reference],.90)+.19*swing_level
            if len(strong):
                after=np.flatnonzero(valid[strong[-1]+1:b] & (energy[strong[-1]+1:b]<=soft_level))+strong[-1]+1
                for end in after:
                    r=np.searchsorted(t,t[end]+.35*period)
                    if (label[b:r]=='standing').any() and (np.nanmedian(sep[end:r]) >= .6*np.nanmedian(sep[reference])):
                        label[end:r]='standing'
                        direct[end:r]=double[end:r]&leg_quiet[end:r]
                        evidence[end:r]=np.where(direct[end:r],'cropped_ground_quiet','cropped_ground_quiet_context')
                        b=end
                        break
        label[a:b] = 'gait_walk' if extended else 'walking'
        evidence[a:b] = 'cropped_gait_repeated_ground_support' if extended else 'cropped_gait_type_unresolved'
        direct[a:b] = extended & double[a:b]
        step[a:b]=period
    for a,b in runs(label==''):
        if a==0 or b==n or not (label[a-1]==label[b]=='standing'):
            continue
        if not np.any(nonstill[a:b]) and (b-a)*dt<=.30:
            label[a:b]='standing'; evidence[a:b]='cropped_ground_quiet_context'
    for a,b in runs(label==''):
        if a==0 or b==n or {label[a-1],label[b]}!={'standing','gait_walk'}:
            continue
        walk_edge=a-1 if label[a-1]=='gait_walk' else b
        stride=step[walk_edge]
        if not np.isfinite(stride) or (b-a)*dt > .36*stride or valid[a:b].mean()<.75:
            continue
        l,r=np.searchsorted(t,[t[walk_edge]-stride,t[walk_edge]+stride])
        sample=energy[l:r][(label[l:r]=='gait_walk') & valid[l:r]]
        if len(sample) < 4 or not np.any(planted[a:b]):
            continue
        strong = valid[a:b] & (energy[a:b] > np.maximum(3.2*background[a:b],.30*np.median(sample)))
        marks=np.flatnonzero(strong)
        if label[a-1]=='standing':
            cut=a+marks[0] if len(marks) else b
            stand,walk=slice(a,cut),slice(cut,b)
        else:
            cut=a+marks[-1]+1 if len(marks) else a
            walk,stand=slice(a,cut),slice(cut,b)
        label[stand]='standing'; evidence[stand]='cropped_ground_quiet_context'
        label[walk]='gait_walk'; evidence[walk]='cropped_gait_repeated_ground_support'; step[walk]=stride
    if n>1 and label[0]=='' and label[1]=='standing':
        label[0]='standing'; evidence[0]='cropped_ground_quiet_context'
    return label,direct,evidence,step


def restore_image_supported_gaps(data, states, measurements):
    additions=[]
    t=data['time_s']
    for identity,(label,direct,evidence,period) in states.items():
        rows=np.flatnonzero(data['track_id']==identity)
        idx=data['frame_index'][rows]
        for i in np.flatnonzero(np.diff(idx)>1):
            a,b=idx[i:i+2]
            selection=np.arange(a+1,b)
            if not (label[a]==label[b]=='gait_walk' and (label[selection]=='gait_walk').all()):
                continue
            if t[b]-t[a] > .44*np.nanmedian(period[a:b+1]):
                continue
            if not np.isfinite(measurements[identity][selection,4]).all():
                continue
            qa,qb=rows[i:i+2]
            if data['yolo_track_id'][qa]!=data['yolo_track_id'][qb]:
                continue
            left,right=data['boxes'][[qa,qb]]
            if np.any(np.minimum(left[2:],right[2:]) <= np.maximum(left[:2],right[:2])):
                continue
            for k in selection:
                weight=(t[k]-t[a])/(t[b]-t[a]); box=left+(right-left)*weight
                others=data['boxes'][(data['frame_index']==k) & (data['track_id']!=identity)]
                if np.any(np.prod(np.maximum(0,np.minimum(others[:,2:],box[2:])-np.maximum(others[:,:2],box[:2])),axis=1)>0):
                    continue
                additions.append((k,identity,data['yolo_track_id'][qa],box))
    result=dict(data,cropped_position_inferred=np.zeros(len(data['track_id']),bool))
    if not additions:
        return result
    appended={
        'frame_index':[row[0] for row in additions], 'track_id':[row[1] for row in additions],
        'yolo_track_id':[row[2] for row in additions], 'boxes':[row[3] for row in additions],
        'confidence':[np.nan]*len(additions), 'keypoints':np.full((len(additions),17,3),np.nan),
        'cropped_position_inferred':np.ones(len(additions),bool),
    }
    order=np.argsort(np.r_[data['frame_index'],appended['frame_index']],kind='stable')
    for field,value in appended.items():
        result[field]=np.concatenate([result[field],np.asarray(value,dtype=result[field].dtype)],axis=0)[order]
    return result


def prepare(source,data,conf):
    groups=[np.flatnonzero(data['track_id']==j) for j in np.unique(data['track_id'])]
    height=int(data['height'])
    plausible=int(round(.18*height))
    probe=any(not cropped_track(data,g,conf) and cropped_track(data,g,conf,plausible) for g in groups)
    offset=content_top(source,data) if probe else 0
    data=dict(data,cropped_content_top_px=offset)
    data,modes,suppressed=deduplicate(data,conf,offset)
    if not modes:
        return data,{},suppressed
    banner_modes={j for j in modes if offset and not cropped_track(data,np.flatnonzero(data['track_id']==j),conf)}
    data['cropped_banner_track_ids']=sorted(banner_modes)
    measurements=collect(source,data,modes,conf,ankle_conf={j:.90*conf for j in banner_modes})
    states={j:classify(data['time_s'],value) for j,value in measurements.items()}
    return restore_image_supported_gaps(data,states,measurements),states,suppressed


def apply(rows, state, lo):
    labels,direct,evidence,period = state
    for i,row in enumerate(rows):
        k=lo+i
        label=labels[k]
        if not label or row['action_event_id'] or row['activity'] in ('running','squat','stairs_up','stairs_down'):
            continue
        if label == 'walking' and row['activity'] == 'gait_walk':
            continue
        basis=evidence[k]
        ambiguous=label=='walking'
        row.update(activity=label,activity_reason=basis,gait_evidence=basis,
                   action_hypotheses=('walking','running') if ambiguous else (),
                   ambiguous_gait=int(ambiguous),locomotion_partial=0,
                   gait_inferred=int(not direct[k]),
                   cropped_gait_stride_s=float(period[k]) if np.isfinite(period[k]) else None,
                   cropped_ground_observation=int(direct[k]))
