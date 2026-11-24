from __future__ import annotations

import argparse
import html
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
VERSION = 'observable-video-foot-v4.4'
SIDES = ('left', 'right')
ACC = {s: [s+'_acc_'+a+'_g' for a in 'xyz'] for s in SIDES}
GYRO = {s: [s+'_gyro_'+a+'_dps' for a in 'xyz'] for s in SIDES}
METRICS = ('gyro_rms_dps', 'angular_acceleration_rms_dps2', 'acc_magnitude_sd_g')
GRADE = {3: 'GOOD', 2: 'MID', 1: 'POOR', 0: 'REVIEW'}


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def load_config(path=None):
    defaults=json.loads((HERE/'grading_config.json').read_text(encoding='utf-8'))
    if path:
        supplied=json.loads(Path(path).read_text(encoding='utf-8'))
        unknown=set(supplied)-set(defaults)
        if unknown:raise ValueError('Unknown grading settings: '+str(sorted(unknown)))
        defaults.update(supplied)
    cfg=defaults
    for key,value in cfg.items():
        if key=='profile_name' or value is None:continue
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<=0:
            raise ValueError('Invalid grading setting: '+key)
    for key in ('minimum_pose_coverage','minimum_imu_coverage','keypoint_confidence','joint_onset_fraction',
                'dissociation_minimum_depth_progress','indirect_duration_fraction','indirect_near_top_progress',
                'indirect_front_min_shin_ratio','indirect_front_max_folded_projection',
                'indirect_front_extra_fold_fraction','indirect_leg_match_tolerance',
                'indirect_matched_progress_tolerance','indirect_bottom_progress','indirect_bottom_comparison_fraction'):
        if cfg[key]>1:raise ValueError(key+' must not exceed 1')
    if int(cfg['minimum_imu_samples']) != cfg['minimum_imu_samples'] or cfg['minimum_imu_samples']<3:
        raise ValueError('minimum_imu_samples must be an integer >= 3')
    for low,high in [('return_error_warn_fraction','return_error_severe_fraction'),
                     ('ankle_motion_warn_leg_ratio','ankle_motion_severe_leg_ratio'),
                     ('depth_deficit_warn_thigh_ratio','depth_deficit_severe_thigh_ratio'),
                     ('trunk_shin_warn_deg','trunk_shin_severe_deg'),
                     ('knee_inward_warn_leg_ratio','knee_inward_severe_leg_ratio'),
                     ('trunk_lateral_warn_deg','trunk_lateral_severe_deg'),
                     ('hip_shoulder_lag_warn_leg_ratio','hip_shoulder_lag_severe_leg_ratio'),
                     ('front_dissociation_warn','front_dissociation_severe'),
                     ('side_dissociation_warn','side_dissociation_severe')]:
        if cfg[low]>=cfg[high]:raise ValueError('Threshold order: '+low)
    if not 0<cfg['torso_projection_severe_ratio']<cfg['torso_projection_warn_ratio']<1:
        raise ValueError('Torso projection ratios must increase in (0,1)')
    if cfg['imu_event_contrast_ratio'] <= 1:
        raise ValueError('imu_event_contrast_ratio must exceed 1')
    if int(cfg['imu_event_minimum_samples']) != cfg['imu_event_minimum_samples'] or cfg['imu_event_minimum_samples'] < 3:
        raise ValueError('imu_event_minimum_samples must be an integer >= 3')
    if cfg['indirect_side_length_ratio_min']>=1 or cfg['indirect_side_length_ratio_max']<=1:
        raise ValueError('Side projection length bounds must enclose 1')
    if cfg['indirect_min_duration_s']>cfg['indirect_duration_cap_s']:
        raise ValueError('Indirect persistence floor exceeds its cap')
    if cfg['indirect_near_top_progress']>=cfg['indirect_bottom_progress']:
        raise ValueError('Near-top progress must precede bottom progress')
    return cfg

def continuous_signals(table):
    required=['estimated_imu_time_s']+sum((ACC[s]+GYRO[s] for s in SIDES),[])
    x=table[required].to_numpy(dtype=float)
    if len(x)<3 or not np.isfinite(x).all() or np.any(np.diff(x[:,0])<=0):
        raise ValueError('Expected finite continuous six-axis foot signals and increasing seconds')
    t=x[:,0]
    if not np.allclose(np.diff(t),.02,atol=1e-8,rtol=0):
        raise ValueError('Expected the pipeline 50 Hz grid; do not treat milliseconds as seconds')
    signals={}
    for s in SIDES:
        w=table[GYRO[s]].to_numpy(float)
        a=table[ACC[s]].to_numpy(float)
        signals[s]={'gyro':w,'alpha':np.gradient(w,t,axis=0,edge_order=2),
                    'acc_magnitude':np.linalg.norm(a,axis=1)}
    return t,signals

def window_metrics(t,signals,start,stop):
    lo,hi=np.searchsorted(t,[start,stop],side='left')
    n=int(hi-lo)
    result={'imu_samples':n,'grid50_start_index_0based':int(lo),'grid50_stop_index_exclusive':int(hi)}
    result['imu_time_coverage']=float(max(0,min(stop,t[-1])-max(start,t[0]))/(stop-start)) if stop>start else 0.
    for s in SIDES:
        result[s+'_gyro_rms_dps']=float(np.sqrt(np.mean(np.sum(signals[s]['gyro'][lo:hi]**2,axis=1)))) if n else np.nan
        result[s+'_angular_acceleration_rms_dps2']=float(np.sqrt(np.mean(np.sum(signals[s]['alpha'][lo:hi]**2,axis=1)))) if n else np.nan
        result[s+'_acc_magnitude_sd_g']=float(np.std(signals[s]['acc_magnitude'][lo:hi],ddof=1)) if n>=2 else np.nan
        result[s+'_acc_magnitude_sd_mps2']=result[s+'_acc_magnitude_sd_g']*9.80665
    for name in METRICS:
        result[name]=float(np.hypot(result['left_'+name],result['right_'+name])/np.sqrt(2))
    result['acc_magnitude_sd_mps2']=result['acc_magnitude_sd_g']*9.80665
    left,right=result['left_gyro_rms_dps'],result['right_gyro_rms_dps']
    result['bilateral_gyro_rms_abs_diff_dps']=abs(left-right)

    result['bilateral_gyro_rms_relative_diff_pct']=200*abs(left-right)/(left+right) if left+right>1e-12 else np.nan
    return result

from individual_quality import pose_features as video_features, assign as assign_grades
from individual_quality import SOURCES, read_reviews, write_review_template

def add_row_grades(source,destination,grades,rep_column,sep=','):
    table=pd.read_csv(source,sep=sep,dtype=str,keep_default_na=False)
    mapping=grades.set_index('rep_id')
    codes={int(k):int(v) for k,v in mapping.candidate_grade_code.items()}
    labels={int(k):v for k,v in mapping.candidate_grade_zh.items()}
    ids=pd.to_numeric(table[rep_column],errors='raise').astype(int)
    table['candidate_grade_code']=ids.map(codes).fillna(0).astype(int)
    table['candidate_grade_zh']=ids.map(labels).fillna('NON_SQUAT')
    for name in ['grade_provisional','grade_confidence','grade_reasons']:
        values=mapping[name].to_dict()
        table[name]=ids.map(values).fillna(0 if name=='grade_provisional' else 'NON_SQUAT')
    table['grade_reviewed']=ids.map(mapping.grade_reviewed.to_dict()).fillna(0).astype(int)
    table.to_csv(destination,index=False,sep=sep,encoding='utf-8-sig')

    before=pd.read_csv(source,sep=sep,dtype=str,keep_default_na=False)
    after=pd.read_csv(destination,sep=sep,dtype=str,keep_default_na=False)
    pd.testing.assert_frame_equal(before,after[before.columns])
    return len(table)

def make_video_overlay(grades,pose_data=None):
    import cv2
    panels = {}
    records = {int(r['rep_id']): r for r in grades.to_dict('records')}
    colors = {3: (100, 220, 70), 2: (80, 210, 255), 1: (100, 100, 255), 0: (190, 190, 190)}
    def overlay(image, row, event):
        record = records.get(int(event['rep_id'])) if event else None
        if record is None:
            return image
        h, w = image.shape[:2]
        key = (int(event['rep_id']), w)
        if key not in panels:
            panel = np.full((115, w, 3), 20, dtype=np.uint8)
            code = int(record['candidate_grade_code'])
            basis = record.get('grading_basis', 'VIDEO_ONLY')
            counts = str(int(record.get('observable_criteria_count', 0)))+'/'+str(int(record.get('expected_observable_criteria_count', 0)))
            lines = [(25, 'GRADE: '+GRADE[code]+' | VISIBLE SCREEN', colors[code], .7)]
            if basis == 'VIDEO_PLUS_IMU':
                lines += [(53, f"RMS w: {record.get('gyro_rms_dps',np.nan):.2f} deg/s | alpha: {record.get('angular_acceleration_rms_dps2',np.nan):.1f} deg/s2", (220,220,220), .48),
                          (77, f"SD |a|: {record.get('acc_magnitude_sd_g',np.nan):.4f} g | L-R: {record.get('bilateral_gyro_rms_abs_diff_dps',np.nan):.2f} deg/s", (220,220,220), .48)]
            else:
                view = str(record.get('video_view_hint', 'uncertain')).split('_')[0].upper()
                reason = str(record.get('grade_reasons', '')).split(';')[0]
                if record.get('indirect_cue_count',0):
                    reason='COORDINATION RISK: '+str(int(record['indirect_cue_count']))+' indirect cue(s); cause unconfirmed'
                lines += [(53, view+' | Observable criteria: '+counts, (220,220,220), .48),
                          (77, reason, (220,220,220), .48)]
            lines += [(103, basis.replace('_', ' ')+' | 2-D MOTION SCREEN | criteria '+counts, (220,220,220), .48)]
            for y, text, color, size in lines:
                size = min(size, size*(w-25)/max(1,cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,size,1)[0][0]))
                cv2.putText(panel,text,(14,y),cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)
            panels[key] = panel
        if h > 354:
            panel=panels[key].copy()
            time=float(row.get('video_time_s',-1))
            cues=json.loads(record.get('indirect_cues_json','[]'))
            active=[c for c in cues if c['start_s']<=time<c['stop_s']]
            if active:
                cue=max(active,key=lambda c:c['value']/c['threshold'])
                text='ACTIVE CUE: '+cue['label']+' | REF F'+str(cue['reference_frame_1based'])
                panel[83:115]=20
                size=min(.48,.48*(w-25)/max(1,cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,.48,1)[0][0]))
                cv2.putText(panel,text,(14,103),cv2.FONT_HERSHEY_SIMPLEX,size,(220,120,255),1,cv2.LINE_AA)
                if pose_data is not None and row.get('video_frame'):
                    i=int(row['video_frame'])-1
                    ids=json.loads(record.get('indirect_body_ids_json','[[5,6],[11,12]]'))[:2]
                    groups=[np.asarray(pose_data['keypoints'][i,index]) for index in ids]
                    if all(np.isfinite(g).all() and (g[:,2]>=.5).all() for g in groups):
                        points=[g[:,:2].mean(axis=0)*[w/int(pose_data['width']),h/int(pose_data['height'])] for g in groups]
                        a,b=[tuple(np.rint(x).astype(int)) for x in points]
                        cv2.line(image,a,b,(220,100,255),5,cv2.LINE_AA)
                        for point in (a,b):cv2.circle(image,point,7,(220,100,255),-1)
            image[174:289,:] = panel
        return image
    return overlay

def write_review_html(path,grades,video_source='../video/review.mp4'):
    rows=[]
    for r in grades.to_dict('records'):
        code=int(r['candidate_grade_code'])
        def number(key):
            v=r.get(key,np.nan)
            return f'{v:.4g}' if np.isfinite(v) else '—'
        vals=[str(int(r['rep_id'])),GRADE[code],f"{r['start_video_s']:.3f}–{r['end_video_s']:.3f}",
              *[number(k) for k in METRICS],number('bilateral_gyro_rms_abs_diff_dps'),
              number('video_torso_projection_ratio'),number('video_trunk_shin_difference_deg'),
              str(int(r.get('fusion_evidence_level',0))),
              str(r.get('grading_basis','VIDEO_ONLY'))+'; '+str(int(r.get('observable_criteria_count',0)))+'/'+str(int(r.get('expected_observable_criteria_count',0)))+' observable criteria',str(r['grade_reasons'])]
        cells=''.join('<td>'+html.escape(x)+'</td>' for x in vals[:-1])
        cues=json.loads(r.get('indirect_cues_json','[]'))
        buttons=' '.join('<button onclick="event.stopPropagation();playRep('+str(c['start_s'])+','+str(c['stop_s'])+')">'+html.escape(c['label'])+'</button>' for c in cues)
        cells+='<td>'+str(len(cues))+' '+buttons+'</td><td>'+html.escape(vals[-1])+'</td>'
        rows.append(f'<tr class="c{code}" onclick="playRep({r["start_video_s"]},{r["end_video_s"]})">{cells}</tr>')
    text = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Squat grades</title>'
            '<style>body{font:15px system-ui;margin:24px;color:#203040}video{height:480px;max-width:100%;background:#111}'
            'table{border-collapse:collapse;width:100%}td,th{padding:9px;border:1px solid #ccc;text-align:left}'
            '.c3{background:#edf8ed}.c2{background:#fff5d9}.c1{background:#ffe6e6}tr{cursor:pointer}p{max-width:1100px}</style>'
            '<h1>View-specific observable squat motion screening</h1>'
            '<p>GOOD: no issue detected in six observable screening criteria; unmeasured anatomy is not cleared. MID: an indirect coordination cue, minor issue or insufficient actual pose observation. '
            'POOR: a major supported issue. These configurable engineering rules are not an expert-validated technique scale.</p>'
            '<p>Front: completion, ankle stability, knee tracking, lateral trunk motion and trunk-leg coordination. '
            'Side: completion, ankle stability, projected depth, hip-shoulder rise and trunk-leg coordination. '
            'Pelvic tilt, lumbar curvature, heel contact and loading are outside automatic scope. '
            'Missing IMU does not lower a video-only grade. Static lean and IMU intensity alone do not lower the grade.</p>'
            '<p>The sixth criterion checks phase-specific indirect coordination: early trunk fold, delayed trunk recovery, '
            'extra bottom fold with little leg motion, and ascent fold compared at matched leg posture. '
            'Front uses torso/shin projection ratios; side uses trunk lean and visible knee motion. '
            'Each indirect cue is capped at MID and does not diagnose pelvic tilt or lumbar curvature. Click a cue button to inspect its actual time span.</p>'
            '<video id="v" controls preload="metadata" src="'+html.escape(video_source,quote=True)+'"></video>'
            '<p>Click a row to play a repetition. When supplied, four bilateral foot metrics support time-coincident visible motion. '
            'Common start remains an alignment assumption.</p>'
            '<table><thead><tr><th>Rep</th><th>Grade</th><th>Seconds</th><th>Gyro RMS deg/s</th><th>Angular acceleration RMS deg/s2</th>'
            '<th>Acceleration magnitude SD g</th><th>Bilateral RMS difference deg/s</th><th>Torso projection ratio</th>'
            '<th>Trunk-shin angle difference deg</th><th>Joint evidence level</th><th>Evidence status</th><th>Indirect cues</th><th>Reasons</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table>'
            "<script>const v=document.getElementById('v');let stopAt=null;function playRep(a,b){stopAt=b;v.currentTime=a;v.play();}"
            "v.addEventListener('timeupdate',()=>{if(stopAt!==null&&v.currentTime>=stopAt){v.pause();stopAt=null;}});</script></html>")
    Path(path).write_text(text,encoding='utf-8')

def run(video_dir,data_dir,out,*,mapped=True,config_path=None,review_path=None):
    video_dir,out=Path(video_dir),Path(out)
    data_dir=Path(data_dir) if data_dir is not None else None
    if mapped and data_dir is None:
        raise ValueError('Mapped IMU grading requires a processed data directory.')
    out.mkdir(parents=True,exist_ok=True)
    cfg=load_config(config_path)
    save_json(out/'grading_config_used.json',cfg)
    reps=pd.read_csv(video_dir/'intervals.csv')
    pose_hash=hashlib.sha256((video_dir/'keypoints.npz').read_bytes()).hexdigest()
    reviews=read_reviews(review_path,reps,pose_hash)
    write_review_template(out/'video_review_template.csv',reps,pose_hash)
    vf=video_features(video_dir,reps,cfg)
    table=reps.merge(vf,on='rep_id',how='left',validate='one_to_one')
    metrics=[]
    if mapped and len(table):
        from motion_fusion import imu_video_evidence
        source=pd.read_csv(data_dir/'feet_50hz_unlabeled.csv')
        t,signals=continuous_signals(source)
        video_rows=table.set_index('rep_id').to_dict('index')
        for e in reps.itertuples(index=False):
            m=window_metrics(t,signals,e.start_video_s,e.end_video_s)
            m['rep_id']=int(e.rep_id)
            variations=[];asym_variations=[]
            for shift in [-cfg['sync_sensitivity_shift_s'],cfg['sync_sensitivity_shift_s']]:
                shifted=window_metrics(t,signals,e.start_video_s+shift,e.end_video_s+shift)
                if shifted['imu_time_coverage']>=cfg['minimum_imu_coverage']:
                    variations.extend(abs(shifted[key]-m[key])/max(m[key],floor) for key,floor in zip(METRICS,[2.,20.,.001]))
                    asym_variations.append(abs(shifted['bilateral_gyro_rms_relative_diff_pct']-m['bilateral_gyro_rms_relative_diff_pct']))
            m['sync_shift_sensitivity_test_s']=cfg['sync_sensitivity_shift_s']
            m['sync_shift_max_metric_relative_change']=max(variations) if variations else np.nan
            m['sync_shift_max_asymmetry_change_pp']=max(asym_variations) if asym_variations else np.nan
            m.update(imu_video_evidence(video_rows[int(e.rep_id)],t,signals,cfg))
            metrics.append(m)
        table=table.merge(pd.DataFrame(metrics),on='rep_id',validate='one_to_one')
        grades=assign_grades(table,cfg,reviews)
    else:

        for name in METRICS+('bilateral_gyro_rms_abs_diff_dps','bilateral_gyro_rms_relative_diff_pct'):
            table[name]=np.nan
        grades=assign_grades(table,cfg,reviews)
    grades['grade_reviewed']=(grades.get('manual_video_items_used',pd.Series('',index=grades.index)).fillna('')!='').astype(int)
    grades['grade_is_candidate']=1;grades['sync_confirmed']=0
    grades['wrist_used']=0;grades['device_euler_angles_used']=0
    grades['grade_scope']='view_specific_2d_motion_control_with_optional_foot_imu'
    grades.to_csv(out/'rep_metrics_and_grades.csv',index=False,encoding='utf-8-sig',float_format='%.10g')
    indirect_rows=[]
    for r in grades.to_dict('records'):
        for cue in json.loads(r.get('indirect_cues_json','[]')):
            indirect_rows.append(dict(rep_id=r['rep_id'],**cue))
    pd.DataFrame(indirect_rows,columns=list(dict.fromkeys(['rep_id','kind','label','view','start_s','stop_s',
        'frame_1based','reference_frame_1based','value','threshold','unit','level']+[k for r in indirect_rows for k in r]))).to_csv(
        out/'video_indirect_evidence.csv',index=False,encoding='utf-8-sig')
    evidence_rows=[]
    for r in grades.to_dict('records'):
        for detail in json.loads(r.get('imu_event_details_json','[]')):
            event=detail['video_event'];baseline=detail['time_shift_tests'][0]
            er={'rep_id':r['rep_id'],'event_kind':event['kind'],'start_video_s':event['start_s'],
                'end_video_s':event['stop_s'],'video_event_level':event['level'],
                'fusion_level':detail['fusion_level'],'imu_available':baseline['available'],
                'burst_robust_to_tested_shifts':detail['burst_robust_to_tested_shifts'],
                'asymmetry_robust_to_tested_shifts':detail['asymmetry_robust_to_tested_shifts']}
            er.update({'event_'+k:v for k,v in baseline.get('metrics',{}).items()})
            er.update({'same_rep_rest_'+k:v for k,v in baseline.get('same_rep_background_metrics',{}).items()})
            evidence_rows.append(er)
    evidence_columns=['rep_id','event_kind','start_video_s','end_video_s','video_event_level','fusion_level',
                      'imu_available','burst_robust_to_tested_shifts','asymmetry_robust_to_tested_shifts']
    pd.DataFrame(evidence_rows,columns=list(dict.fromkeys(evidence_columns+[k for e in evidence_rows for k in e]))).to_csv(
        out/'video_imu_event_evidence.csv',index=False,encoding='utf-8-sig')
    rows={}
    if mapped:
        rows['grid50']=add_row_grades(data_dir/'feet_50hz_candidate_labels.csv',data_dir/'feet_50hz_candidate_grades.csv',grades,'candidate_rep_id')
        rows['native']=add_row_grades(data_dir/'native_feet_preserved.tsv',data_dir/'native_feet_candidate_grades.tsv',grades,'candidate_rep_id',sep='\t')
    elif data_dir is not None:
        for name in ['feet_50hz_candidate_grades.csv','native_feet_candidate_grades.tsv']:
            p=data_dir/name
            if p.is_file():p.unlink()

    frame_table=pd.read_csv(video_dir/'frame_labels.csv',dtype=str,keep_default_na=False)
    frame_ids=pd.to_numeric(frame_table.rep_id,errors='raise').astype(int)
    lookup=grades.set_index('rep_id').candidate_grade_code.to_dict()
    frame_table['candidate_grade_code']=frame_ids.map(lookup).fillna(0).astype(int)
    frame_table['candidate_grade_zh']=frame_ids.map(grades.set_index('rep_id').candidate_grade_zh.to_dict()).fillna('NON_SQUAT')
    for name in ['grade_reviewed','grade_provisional','grade_confidence','grade_reasons']:
        frame_table[name]=frame_ids.map(grades.set_index('rep_id')[name].to_dict()).fillna(0 if name in ['grade_reviewed','grade_provisional'] else 'NON_SQUAT')
    frame_table.to_csv(video_dir/'frame_candidate_grades.csv',index=False,encoding='utf-8-sig')
    write_review_html(out/'review.html',grades,os.path.relpath(video_dir/'review.mp4',out).replace('\\','/'))
    report={'version':VERSION,'status':'complete','repetitions':len(grades),
            'candidate_grade_counts':{GRADE[k]:int((grades.candidate_grade_code==k).sum()) for k in [3,2,1]},
            'all_repetitions_have_one_of_three_grades':bool(grades.candidate_grade_code.isin([1,2,3]).all()),
            'used_imu_channels':sum((ACC[s]+GYRO[s] for s in SIDES),[]),
            'device_euler_angles_used':False,'wrist_used':False,'gyro_bias_calibrated':False,
            'gravity_removed_from_acceleration':False,'acceleration_sd_ddof':1,
            'angular_acceleration_method':'numpy.gradient continuous 8Hz-filtered 50Hz gyro against seconds, edge_order=2; then slice each half-open video interval',
            'combined_metric_definition':'root mean square of left/right per-foot RMS or SD (not average axes)',
            'bilateral_relative_difference_definition':'200*abs(RMS_L-RMS_R)/(RMS_L+RMS_R), range 0..200 percent',
            'sample_mapping':bool(mapped),'sync_confirmed':False,'unknown_packet_losses_recovered':False,
            'thresholds_are_configurable_engineering_rules':True,'thresholds_validated_against_expert_grades':False,
            'reference_scope':'each repetition independently; view-specific observable motion; no peer target',
            'standard_sources':SOURCES,'other_repetitions_used_for_grade':False,
            'posterior_pelvic_tilt_automatic_detection':False,'lumbar_collapse_automatic_detection':False,
            'fusion_method':'video-selected sustained events + all four local foot metrics; contrast against rest of the SAME repetition, require support at shifts 0 and +/- configured sensitivity',
            'four_metrics_used_in_fusion':list(METRICS)+['bilateral_gyro_rms_abs_diff_dps'],
            'torso_projection_alone_penalizes':False,'imu_intensity_alone_penalizes':False,
            'temporal_fusion_does_not_confirm_synchronization':True,
            'event_evidence_table':'video_imu_event_evidence.csv',
            'missing_critical_evidence_caps_good_at_mid':False,
            'unobservable_anatomy_required_for_good':False,'missing_imu_lowers_video_grade':False,
            'actual_pose_or_view_gaps_retain_review_status':True,
            'automatic_scope':'view_specific_2d_motion_control',
            'indirect_coordination_cues':True,'indirect_cue_maximum_grade_penalty':'MID',
            'indirect_cue_anatomical_diagnosis':False,'indirect_evidence_table':'video_indirect_evidence.csv',
            'indirect_features_version':'phase_matched_indirect_cues_v1',
            'indirect_cue_repetitions':int((grades.get('indirect_cue_count',pd.Series(dtype=float)).fillna(0)>0).sum()),
            'excluded_automatic_items':['pelvic_tilt','lumbar_curvature','heel_contact','load_distribution'],
            'grading_basis_counts':grades.get('grading_basis',pd.Series(dtype=str)).value_counts().to_dict(),
            'video_review_file':str(review_path) if review_path else None,
            'not_a_cross_subject_ranking':True,'not_anatomical_3d_angles_or_loading':True,
            'class_distribution_forced':False,'input_rows_preserved':rows,
            'config_file':'grading_config_used.json','per_rep_table':'rep_metrics_and_grades.csv'}
    save_json(out/'grading_audit.json',report)
    return report,grades

def render_graded_video(source,video_dir,grades,max_height=None):
    from pipeline_preview import render_preview
    source,video_dir=Path(source),Path(video_dir)
    metadata=json.loads((video_dir/'metadata.json').read_text(encoding='utf-8'))
    from video_to_intervals import sha256_file
    if metadata.get('source_sha256') != sha256_file(source):
        raise ValueError('Graded preview source hash does not match its annotation.')
    frames=pd.read_csv(video_dir/'frame_labels.csv').fillna('').to_dict('records')
    events=pd.read_csv(video_dir/'intervals.csv').fillna('').to_dict('records')
    activities=pd.read_csv(video_dir/'activities.csv').fillna('').to_dict('records')
    with np.load(video_dir/'keypoints.npz',allow_pickle=False) as cache:
        render_preview(source,video_dir/'review.mp4',cache,frames,events,activities,
                       max_height=max_height,frame_overlay=make_video_overlay(grades,cache))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video-dir',required=True,type=Path);p.add_argument('--data-dir',type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--config',type=Path);p.add_argument('--video-review',type=Path);p.add_argument('--unmapped',action='store_true')
    a=p.parse_args();report,_=run(a.video_dir,a.data_dir,a.output,mapped=not a.unmapped and a.data_dir is not None,config_path=a.config,review_path=a.video_review)
    print(json.dumps(report,ensure_ascii=False))
