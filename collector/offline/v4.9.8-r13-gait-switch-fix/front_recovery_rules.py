import numpy as np
from body_boundary_rules import body_features,neutral_hold,terminal_standing_stop

def cap_front_recoveries(f, events, frames):
    t=f['t'];previous_bottom=0
    for e in events:
        bottom=int(e['bottom_video_frame'])-1
        if e['body_boundary_status']!='body_rule_candidate':
            previous_bottom=bottom
            continue
        old_end=int(e['end_video_frame'])
        before=neutral_hold(f,previous_bottom,bottom)
        core=dict(e,end_video_frame=int(e['hip_core_end_video_frame']))
        end,info=terminal_standing_stop(f,core,before,old_end)
        e.update(recovery_stop_status=info['status'],recovery_previous_end_video_s=float(t[old_end]),
                 recovery_previous_end_video_frame=old_end)
        if end<old_end:
            e.update(end_video_frame=end,end_exclusive_video_frame=end+1,
                     end_video_s=float(t[end]),duration_s=float(t[end]-e['start_video_s']),
                     end_later_s=float(t[end]-e['hip_core_end_video_s']),end_driver='upright_body_hold',
                     end_sensitivity_first_frame=min(int(e.get('end_sensitivity_first_frame',end)),end),
                     end_sensitivity_last_frame=min(int(e.get('end_sensitivity_last_frame',end)),end),
                     recovery_confirmation_end_video_frame=info['confirmation_end']+1,
                     recovery_speed_limit_leg_s=info['speed_limit'])
            for i in range(end,old_end):
                if frames[i]['rep_id']==e['rep_id']:
                    frames[i].update(rep_id=0,squat_label=0,reason='confirmed_intercycle_upright_hold',movement_phase='')
        previous_bottom=bottom
    return events,frames

def refine_front_recoveries(data,args,events,frames):
    return cap_front_recoveries(body_features(data,args),events,frames)
