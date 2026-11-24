import numpy as np


def number_outputs(data, tables, summaries):
    accepted = {row['track_id'] for row in summaries if row['accepted']}
    starts = {}
    for f, group, box in zip(data['frame_index'], data['track_id'], data['boxes']):
        group = int(group)
        if group in accepted and group not in starts:
            starts[group] = (int(f), float((box[0]+box[2])*.5), float((box[1]+box[3])*.5), group)
    if set(starts) != accepted:
        raise ValueError('An accepted analysis track has no observed box.')
    public = {group: j for j, group in enumerate(sorted(starts,key=starts.get),1)}
    for rows in tables.values():
        for row in rows:
            row['track_id'] = public[row['track_id']]
        rows.sort(key=lambda row: row['track_id'])
    for row in summaries:
        row['internal_track_id'] = row['track_id']
        row['track_id'] = public.get(row['internal_track_id'])
    summaries.sort(key=lambda row: (row['track_id'] is None, row['track_id'] or row['internal_track_id']))
    display = np.asarray([public.get(int(group),0) for group in data['track_id']], dtype=np.int64)
    return dict(data,track_id=display), public


def audit_id_map(raw_all, internal_by_raw, public, ignored, suppressed, *, non_person=(), flat_depictions=(), behind_surface=()):
    rows = []
    for raw in sorted(set(map(int,raw_all))):
        internal = internal_by_raw.get(raw)
        status = ('ignored_static_roi' if raw in ignored else
                  'rejected_flat_depiction' if raw in flat_depictions else
                  'rejected_behind_surface' if raw in behind_surface else
                  'rejected_person_evidence' if raw in non_person else
                  'cropped_duplicate' if internal in suppressed else
                  'accepted' if internal in public else 'rejected_short_or_insufficient_track')
        rows.append(dict(yolo_track_id=raw, track_id=public.get(internal), internal_track_id=internal, status=status))
    return rows
