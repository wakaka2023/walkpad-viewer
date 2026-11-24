import numpy as np

def choose_geometry(data,args):
    k=np.asarray(data['keypoints'],float)
    conf=args.keypoint_conf
    legs={}
    for side,ids in [('left',[11,13,15]),('right',[12,14,16])]:
        p=k[:,ids,:2];c=k[:,ids,2]
        length=np.linalg.norm(p[:,0]-p[:,1],axis=1)+np.linalg.norm(p[:,1]-p[:,2],axis=1)
        valid=np.isfinite(k[:,ids]).all(axis=(1,2))&(c>=conf).all(axis=1)&(length>1)
        score=(int(valid.sum()),float(c[valid].mean()) if valid.any() else 0.)
        legs[side]={'ids':ids,'length':length,'valid':valid,'score':score}
    best=max(legs,key=lambda s:legs[s]['score'])
    available=np.concatenate([v['length'][v['valid']] for v in legs.values()])
    if not len(available):raise ValueError('Need at least one visible hip-knee-ankle chain; side view is supported')
    initial_scale=float(np.median(available))
    shoulder_good=np.isfinite(k[:,[5,6]]).all(axis=(1,2))&(k[:,[5,6],2]>=conf).all(axis=1)
    widths=np.linalg.norm(k[:,5,:2]-k[:,6,:2],axis=1)/initial_scale
    width=float(np.median(widths[shoulder_good])) if shoulder_good.any() else np.nan
    hint='side_or_oblique' if width<.25 else 'front_or_oblique' if width>.35 else 'oblique_or_uncertain'
    both=legs['left']['valid']&legs['right']['valid']
    forced=getattr(args,'side','auto')
    if forced not in ('auto','left','right'):raise ValueError('side must be auto, left or right')
    bilateral=(forced=='auto' and hint!='side_or_oblique' and both.sum()>=.85*legs[best]['valid'].sum())
    if bilateral:
        ids=[11,12];ankle_ids=[15,16];valid=both;mode='bilateral_visible'
        scale=initial_scale
    else:
        side=best if forced=='auto' else forced
        limb=legs[side];ids=[limb['ids'][0]];ankle_ids=[limb['ids'][2]];valid=limb['valid'];mode=side+'_visible'
        if not valid.any():raise ValueError('Selected leg has no visible hip-knee-ankle chain: '+side)
        scale=float(np.median(limb['length'][valid]))
    hip=k[:,ids,:2].mean(axis=1)

    ankles=k[:,ankle_ids,:2]
    if len(ankle_ids)==1:ankles=np.repeat(ankles,2,axis=1)
    meta={'view_hint':hint,'view_hint_is_uncalibrated':True,'geometry_mode':mode,
          'selected_hip_ids':ids,'selected_ankle_ids':ankle_ids,
          'visible_leg_fraction':{s:float(v['valid'].mean()) for s,v in legs.items()},
          'shoulder_width_over_projected_leg':width if np.isfinite(width) else None,
          'fixed_limb_selection_for_whole_video':True,'camera_view_forced':False}
    return {'hip':hip,'ankles':ankles,'valid':valid,'scale':scale,'ankle_ids':ankle_ids,'metadata':meta}
