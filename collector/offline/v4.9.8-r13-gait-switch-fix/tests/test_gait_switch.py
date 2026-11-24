import numpy as np
from gait_switch import apply, profile, reference, REASON


def rows(n):
    return [dict(activity='gait_walk', action_event_id=0) for _ in range(n)]


def fixture():
    t = np.arange(45) / 30
    data = np.tile([.02, .20, .07], (len(t), 1))
    data[15:] = [.13,.51,.26]
    refs = dict(walk=[[.02,.20,.07]]*4, run=[[.10,.42,.24]]*4)
    return t,data,np.full(len(t),.40),refs


def test_sustained_person_local_switch_not_based_on_absolute_rate():
    t,x,step,refs = fixture()
    for stretch in (.74, 1., 2.):
        r = rows(len(t))
        apply(t*stretch,r,x,step*stretch,profile(refs))
        assert all(q['activity']=='gait_walk' for q in r[:15])
        assert all(q['activity_reason']==REASON for q in r[15:])


def test_no_borrowing_no_short_glitch_or_unobserved_bridge():
    t,x,step,refs = fixture()
    r = rows(len(t))
    apply(t,r,x,step,profile(dict(walk=refs['walk'],run=[])))
    assert not any(q['activity']=='running' for q in r)
    x[27] = np.nan
    apply(t,r,x,step,profile(refs))
    assert not any(q['activity']=='running' for q in r)


def test_squat_event_leg_only_or_ambiguous_reference_cannot_seed_run():
    t,x,step,refs = fixture()
    r = rows(len(t))
    r[25].update(activity='squat',action_event_id=1)
    apply(t,r,x,step,profile(refs))
    assert r[25]['activity']=='squat' and not any(q['activity']=='running' for q in r)
    x[15:,0] = x[0,0]
    apply(t,r,x,step,profile(refs))
    assert not any(q['activity']=='running' for q in r)
    assert reference(dict(activity='running',gait_evidence='short_running_rebound',action_hypotheses=('running',)))==''
