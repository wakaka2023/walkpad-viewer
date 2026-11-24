import numpy as np

from .action_rules import clean_mask


def relative_cut(values, quiet, usable, prior):
    good = usable & np.isfinite(values)
    baseline = values[good & quiet]
    active = values[good & ~quiet]
    if min(len(baseline), len(active)) < 12:
        return prior
    q50, q90 = np.quantile(baseline, [.5, .9])
    upper = np.quantile(active, .75)
    noise = max(q90-q50, .08*prior)
    if upper-q90 < 2.5*noise:
        return prior
    return float(np.clip(q90 + .32*(upper-q90), .82*prior, 1.18*prior))


def calibrate(f, config):
    t = f['time_s']
    usable = f['torso_visible'] & np.isfinite(f['camera_speed']) & (f['camera_speed'] < config['bend_max_camera_speed'])
    quiet = usable & (f['hip_speed'] < config['quiet_hip_speed'])
    quiet &= (f['foot_speed'] < config['quiet_foot_speed']) | (f['knee_motion'] < config['quiet_knee_speed'])
    quiet = clean_mask(t, quiet, usable, .8, 0.)
    out = dict(config)
    options = {'quiet_hip_speed': 'hip_speed', 'quiet_foot_speed': 'foot_speed', 'quiet_knee_speed': 'knee_motion',
               'walking_net_speed': 'hip_net_speed', 'walking_foot_speed': 'foot_speed',
               'foot_motion_speed': 'foot_motion', 'knee_motion_speed': 'knee_motion'}
    for key, field in options.items():
        if field in f:
            out[key] = relative_cut(f[field], quiet, usable, config[key])
    knees = f['knee_flexion'][quiet & f['both_legs_visible']]
    knees = knees[np.isfinite(knees) & (knees < .8*config['squat_min_knee_flexion_deg'])]
    if len(knees) >= max(12, .6/np.median(np.diff(t))):
        ref, upper = np.quantile(knees, [.30, .80])
        if upper-ref < .4*config['squat_min_knee_excursion_deg']:
            delta = config['squat_min_knee_flexion_deg'] - .65*config['squat_min_knee_excursion_deg']
            out['squat_min_knee_flexion_deg'] = float(np.clip(ref + delta,
                .90*config['squat_min_knee_flexion_deg'], 1.06*config['squat_min_knee_flexion_deg']))
    torso = f['local_torso_ratio'][quiet & (f['knee_flexion'] < .55*config['squat_min_knee_flexion_deg'])]
    torso = torso[np.isfinite(torso) & (torso >= 1-config['bend_min_excursion']*.5)]
    if len(torso) >= max(12, .6/np.median(np.diff(t))):
        ref = float(np.median(torso))
        out['bend_torso_ratio'] = float(np.clip(ref-(1-config['bend_torso_ratio']),
                                                config['bend_torso_ratio']-.03, config['bend_torso_ratio']+.03))
    return out, dict(quiet_reference_frames=int(quiet.sum()), **{k: v for k, v in out.items() if v != config[k]})
