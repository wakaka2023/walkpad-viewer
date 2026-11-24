import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'code/v13'))
from foot_io import read_native_feet
from preprocessing import preprocess_native, VERSION

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def label_times(t, frames):
    start = frames.video_time_s.to_numpy(float)
    stop = frames.frame_end_s.to_numpy(float)
    idx = np.searchsorted(start, t, side='right')-1
    good = (idx >= 0) & (idx < len(frames))
    safe = np.clip(idx, 0, len(frames)-1)
    good &= t < stop[safe]
    result = pd.DataFrame({'estimated_video_frame': np.where(good, safe+1, 0),
                           'candidate_squat_label': np.where(good, frames.squat_label.to_numpy()[safe], -1),
                           'candidate_rep_id': np.where(good, frames.rep_id.to_numpy()[safe], 0),
                           'candidate_activity': np.where(good, frames.activity.to_numpy()[safe], 'outside_video'),
                           'sync_confirmed': 0, 'boundary_reviewed': 0})
    return result

def run(data, video_dir, out, assume_common_start=False, left_device='F8', right_device='F9'):
    data, video_dir, out = Path(data), Path(video_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    before = digest(data)
    native, source_meta = read_native_feet(data, left_device, right_device)
    selected = source_meta['selected_feet']
    end = min(float(t[-1]) for t, x in native.values())
    t50 = np.arange(int(np.floor(end*50))+1)/50
    processed = pd.DataFrame({'estimated_imu_time_s': t50})
    audits = {}
    for side in ('left', 'right'):
        t, x = native[side]
        filtered, audit = preprocess_native(x, 1/(t[1]-t[0]))
        audits[side] = audit
        for j, axis in enumerate(['acc_x_g','acc_y_g','acc_z_g','gyro_x_dps','gyro_y_dps','gyro_z_dps']):
            processed[side+'_'+axis] = np.interp(t50, t, filtered[:,j])
    processed.to_csv(out/'feet_50hz_unlabeled.csv', index=False, encoding='utf-8-sig', float_format='%.10g')

    encoding = 'utf-8-sig'
    try:
        data.read_text(encoding=encoding)
    except UnicodeDecodeError:
        encoding = 'gb18030'
    raw = []
    with data.open(encoding=encoding, newline='') as handle:
        reader = csv.reader(handle, delimiter='\t')
        header = next(reader)
        for row in reader:
            if len(row)>1 and row[1] in selected.values():
                raw.append((reader.line_num, row))
    counters = {side:0 for side in native}
    source_lines = {side:[] for side in native}
    extra = ['source_line_1based','native_sample_index_0based','estimated_native_time_s']
    frames = pd.read_csv(video_dir/'frame_labels.csv') if assume_common_start else None
    if assume_common_start:
        extra += list(label_times(np.array([0.]),frames).columns)
    checks = []
    with (out/'native_feet_preserved.tsv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
        writer.writerow(header+extra)
        native_labels = {s:label_times(native[s][0],frames) for s in native} if assume_common_start else {}
        for line, row in raw:
            side = next(s for s, name in selected.items() if row[1] == name)
            index = counters[side]; counters[side] += 1
            source_lines[side].append(line)
            suffix = [line,index,float(native[side][0][index])]
            if assume_common_start:
                suffix += native_labels[side].iloc[index].tolist()
            writer.writerow(row+suffix)
            checks.append(row)
    reread = []
    with (out/'native_feet_preserved.tsv').open(encoding='utf-8-sig', newline='') as handle:
        reader=csv.reader(handle,delimiter='\t');next(reader)
        reread=[row[:len(header)] for row in reader]
    if checks != reread:
        raise RuntimeError('Raw-cell preservation check failed')
    if any(counters[s] != len(native[s][0]) for s in native):
        raise RuntimeError('Native sample counts and preserved raw rows disagree')

    interval_count = 0
    if assume_common_start:
        labels = label_times(t50,frames)
        candidate = pd.concat([processed,labels],axis=1)
        candidate.to_csv(out/'feet_50hz_candidate_labels.csv',index=False,encoding='utf-8-sig',float_format='%.10g')
        intervals = pd.read_csv(video_dir/'intervals.csv')
        interval_count = len(intervals)
        grid_counts = []
        for side in ('left','right','grid50'):
            t=native[side][0] if side!='grid50' else t50
            lo=np.searchsorted(t,intervals.start_video_s.to_numpy(float),side='left')
            hi=np.searchsorted(t,intervals.end_video_s.to_numpy(float),side='left')
            intervals[side+'_start_index_0based']=lo
            intervals[side+'_stop_index_exclusive']=hi
            intervals[side+'_samples']=hi-lo
            starts=intervals.start_video_s.to_numpy(float)
            stops=intervals.end_video_s.to_numpy(float)
            duration=stops-starts
            overlap=np.maximum(0.,np.minimum(stops,t[-1])-np.maximum(starts,t[0]))
            intervals[side+'_time_coverage_fraction']=np.divide(overlap,duration,out=np.zeros_like(duration),where=duration>0)
            if side!='grid50':
                lines=source_lines[side]
                intervals[side+'_first_source_line_1based']=[lines[a] if b>a else '' for a,b in zip(lo,hi)]
                intervals[side+'_last_source_line_1based']=[lines[b-1] if b>a else '' for a,b in zip(lo,hi)]
            if side=='grid50':
                grid_counts=[int(np.sum(candidate.candidate_rep_id.to_numpy()==rep)) for rep in intervals.rep_id]
                if grid_counts != (hi-lo).tolist():
                    raise RuntimeError('Interval and row labels disagree')
        intervals['sync_confirmed']=0
        intervals['mapping_basis']='assumed_common_start_estimated_clock_unknown_packet_loss'
        intervals.to_csv(out/'rep_boundaries_candidate.csv',index=False,encoding='utf-8-sig')
    after=digest(data)
    if before!=after:
        raise RuntimeError('Source file changed during processing')
    report = {'source':str(data.resolve()),'source_sha256':before,'source_unchanged':True,
              'wrist_used':False,'used_devices':selected,'native_samples':counters,
              'raw_cells_preserved':True,'raw_foot_rows':len(raw),'grid50_rows':len(t50),
              'grid50_end_s':float(t50[-1]),'grid50_interval_s':.02,
              'clock_definition':'Per-foot Huber receive-time slope; native_time = original_device_sample_index * fitted_interval. Does not locate losses.',
              'preprocessing_version':VERSION,'preprocessing_order':'native_Hampel -> native_8Hz_zero_phase_lowpass -> interpolate50 -> label_by_video_PTS',
              'start_assumption':bool(assume_common_start),'sync_confirmed':False,
              'packet_loss_positions_identifiable_from_receive_times':False,'packet_loss_positions_recovered':False,
              'candidate_only':bool(assume_common_start),'candidate_repetitions':interval_count,
              'label_convention':'[start, stop); indices 0-based, stop exclusive; video frame numbers 1-based inclusive',
              'label_interpolation':False,'resample_each_action_to_128':False,
              'source_v13_alignment':source_meta,'preprocessing_audits':audits}
    save_json(out/'processing_audit.json',report)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True,type=Path)
    parser.add_argument('--video-dir',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--left-device',default='F8')
    parser.add_argument('--right-device',default='F9')
    parser.add_argument('--assume-common-start',action='store_true',help='Emit conditional candidate labels. Does NOT confirm synchronization or repair losses.')
    args=parser.parse_args()
    result=run(args.data,args.video_dir,args.output,args.assume_common_start,args.left_device,args.right_device)
    print(json.dumps({k:result[k] for k in ['wrist_used','native_samples','grid50_rows','candidate_repetitions','sync_confirmed','source_unchanged','raw_cells_preserved']},ensure_ascii=False))
