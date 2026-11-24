import io
import json
import zipfile
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.exports import export_package
from backend.lab_attitude import LabAttitude
from backend.storage import FRAME_SCHEMA


STILL = (0, 0, 2049, 5, -8, 15)


def test_attitude_restarts_without_integrating_invalid_samples():
    fusion = LabAttitude('a')
    assert fusion.add((0, 0, 1000, 0, 0, 0), 0) is None
    for i in range(100):
        assert fusion.add(STILL, i * 20_000)[2:] == (None, 'a:0')
    assert fusion.add(STILL, 2_000_000)[2:] == (0, 'a:0')
    turn = STILL[:5] + (1015,)
    assert fusion.add(turn, 2_000_000) is None
    assert fusion.add(turn, 1_900_000) is None
    assert fusion.add(turn, None) is None
    assert fusion.add((0, 0, 0, 0, 0, 0), 2_010_000) is None
    assert fusion.add(turn, 2_020_000)[2:] == (pytest.approx(1.4), 'a:0')
    assert fusion.add(STILL, 2_270_000)[2:] == (pytest.approx(1.4), 'a:0')
    assert fusion.add(STILL, 2_520_001)[2:] == (None, 'a:1')
    assert fusion.add((32760,) + STILL[1:], 2_540_001) is None
    assert fusion.add(STILL, 2_560_001)[2:] == (None, 'a:2')


def make_record(path, *, long=False):
    path.mkdir()
    (path/'manifest.json').write_text(json.dumps({'session_id':path.name,'status':'complete'}), encoding='utf-8')
    epoch_ms = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc).timestamp() * 1000
    rows = []

    def add(device, segment, i, raw, kind='imu', profile='insole'):
        timestamp = None if i is None else epoch_ms + i * 20
        rows.append(dict(profile=profile,device_kind=kind,device_id=device,connection=segment,
                         foot='l' if kind == 'insole' else '',placement='rightAnkle' if kind == 'imu' else None,
                         device_calendar_ms=timestamp,source_frame_number_raw=i,receipt_monotonic_ms=1000+len(rows),
                         pressure_raw_values=[0]*70,imu_raw_values=raw,imu_physical_values=[42.]*6,
                         sensor_topology='v2',imu_bias_ready=True,imu_bias_dps=[1.,2.,3.],imu_bias_revision=1))

    for i in range(20_002 if long else 110):
        add('left', 'left1', i, STILL if i < 101 else STILL[:5] + (1015,), kind='insole')
        if i < 102:
            add('ankle', 'ankle1', i, STILL)
    add('ankle', 'ankle2', 103, (0, 205, 2038, 0, 0, 0))
    add('ankle', 'ankle2', None, STILL)
    add('wit', 'wit1', 0, STILL, profile='wt')
    pq.write_table(pa.Table.from_pylist(rows[:205],schema=FRAME_SCHEMA),path/'frames_000001.parquet')
    pq.write_table(pa.Table.from_pylist(rows[205:],schema=FRAME_SCHEMA),path/'frames_000002.parquet')
    return path


def test_http_lab_package_v26(tmp_path):
    with TestClient(create_app(tmp_path,demo=True)) as client:
        make_record(tmp_path/'records'/'20260102_030405_1234567890')
        response = client.get('/api/sessions/20260102_030405_1234567890/export?format=package')
    assert response.status_code == 200, response.text
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        manifest = json.loads(z.read('manifest.json'))
        assert manifest['schemaVersion'] == 'lab_measurement_package.v2.6'
        assert manifest['videos'] == [] and manifest['includeVideos'] is False
        assert manifest['frameAlignmentMode'] == 'full_capture'
        window = manifest['commonAlignedWindow']
        assert len(window['includedSourceIds']) == 2
        assert window['startMonoNs'] <= window['endMonoNs']
        assert window['durationUs'] == window['endEffectiveElapsedUs']-window['startEffectiveElapsedUs']
        assert window['durationUs'] >= 0
        timeline = manifest['sessionTimeline']
        # 单一来源各自零点推断的周期：起止在会话单调钟上有确切值，不写空
        assert timeline['periods'][0]['startMonoNs'] == timeline['sessionStartMonoNs']
        assert timeline['periods'][0]['endMonoNs'] > timeline['periods'][0]['startMonoNs']
        assert len(manifest['sources']) == 2
        tables = {}
        by_id = {s['sourceId']:s['deviceFamily'] for s in manifest['sources']}
        for output in manifest['outputs']:
            if output['path'].startswith('frames/'):
                table = pq.read_table(io.BytesIO(z.read(output['path'])))
                assert table.column_names == list(manifest['physicalSchemas']['frame'])
                assert table.column_names[14:19] == ['pitchDeg','sideTiltDeg','relativeHeadingDeg','referenceId','pressureAdc']
                assert len(table.column_names) == 30
                tables[by_id[output['sourceId']]] = table.to_pylist()
    left, ankle = tables['d56'], tables['walkpadImu']
    assert left[0]['deviceTimeRaw'] == '20260102030405000'
    assert left[0]['accelerationG']['z'] == pytest.approx(2049 * .000488)
    assert left[0]['angularVelocityDps'] == pytest.approx({'x':-.56,'y':-.35,'z':1.05})
    assert left[0]['imuCalibration']['angularVelocityCompensatedDps'] == pytest.approx({'x':-1.56,'y':-2.35,'z':-1.95})
    assert len(left[0]['pressureAdc']) == 56 and ankle[0]['pressureAdc'] is None
    assert left[99]['relativeHeadingDeg'] is None and left[100]['relativeHeadingDeg'] == 0
    assert left[101]['relativeHeadingDeg'] == pytest.approx(1.4)
    assert left[101]['referenceId'] == left[99]['referenceId'] != ankle[100]['referenceId']
    assert ankle[101]['relativeHeadingDeg'] == 0
    assert ankle[102]['sideTiltDeg'] < 0 and ankle[102]['relativeHeadingDeg'] is None
    assert ankle[102]['referenceId'] != ankle[101]['referenceId']
    assert all(ankle[103][column] is None for column in ('pitchDeg','sideTiltDeg','relativeHeadingDeg','referenceId'))


def test_attitude_survives_package_part_boundary(tmp_path):
    target = export_package(make_record(tmp_path/'capture',long=True))
    with zipfile.ZipFile(target) as archive:
        paths = [p for p in archive.namelist() if p.startswith('frames/d56_')]
        assert len(paths) == 2
        columns = ['index','referenceId','relativeHeadingDeg']
        prev = pq.read_table(io.BytesIO(archive.read(paths[0])),columns=columns).to_pylist()[-1]
        curr = pq.read_table(io.BytesIO(archive.read(paths[1])),columns=columns).to_pylist()[0]
    assert (prev['index'],curr['index']) == (19999,20000)
    assert prev['referenceId'] == curr['referenceId']
    assert (curr['relativeHeadingDeg']-prev['relativeHeadingDeg']) % 360 == pytest.approx(1.4)


def make_paused_record(path):
    path.mkdir()
    started = 5_000.
    periods = [
        dict(period_id='p1',period_no=1,start_monotonic_ms=started,end_monotonic_ms=started+1_000,
             start_session_elapsed_us=0,end_session_elapsed_us=1_000_000,
             start_effective_elapsed_us=0,end_effective_elapsed_us=1_000_000,end_reason='pause'),
        dict(period_id='p2',period_no=2,start_monotonic_ms=started+2_000,end_monotonic_ms=started+4_000,
             start_session_elapsed_us=2_000_000,end_session_elapsed_us=4_000_000,
             start_effective_elapsed_us=1_000_000,end_effective_elapsed_us=3_000_000,end_reason='stop'),
    ]
    (path/'manifest.json').write_text(json.dumps(dict(session_id=path.name,status='complete',
        started_monotonic_ms=started,clock_domain='python.monotonic_ns',periods=periods)),encoding='utf-8')
    epoch_ms = 1767323045000
    rows = []
    for i,receipt in enumerate((5_100.,5_900.,6_500.,7_500.)):
        rows.append(dict(profile='insole',device_kind='insole',device_id='left',connection='left1',foot='l',
                         device_calendar_ms=epoch_ms+i*20,source_frame_number_raw=i,
                         receipt_monotonic_ms=receipt,pressure_raw_values=[0]*70,imu_raw_values=list(STILL),
                         imu_physical_values=[42.]*6,sensor_topology='v2',imu_bias_ready=True,
                         imu_bias_dps=[1.,2.,3.],imu_bias_revision=1))
    pq.write_table(pa.Table.from_pylist(rows,schema=FRAME_SCHEMA),path/'frames_000001.parquet')
    return path,periods


def test_paused_interval_stays_out_of_effective_time(tmp_path):
    target = export_package(make_paused_record(tmp_path/'capture')[0])
    with zipfile.ZipFile(target) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        path = [p for p in archive.namelist() if p.startswith('frames/d56_')][0]
        rows = pq.read_table(io.BytesIO(archive.read(path)),columns=['periodId','effectiveElapsedUs','validity']).to_pylist()
    assert [(r['periodId'],r['effectiveElapsedUs']) for r in rows[:2]] == [('p1',100_000),('p1',900_000)]
    assert rows[2]['periodId'] is None and rows[2]['effectiveElapsedUs'] is None
    assert rows[2]['validity'] == 'outside_period'
    assert (rows[3]['periodId'],rows[3]['effectiveElapsedUs']) == ('p2',1_500_000)
    timeline = manifest['sessionTimeline']
    assert timeline['effectiveTimeRule'] == 'exclude_pauses_half_open_periods'
    assert timeline['effectiveDurationUs'] == 3_000_000
    assert [(p['periodId'],p['endReason']) for p in timeline['periods']] == [('p1','pause'),('p2','stop')]


def test_recorder_pause_closes_and_reopens_period(tmp_path):
    from backend.storage import Recorder
    recorder = Recorder(tmp_path)
    session = recorder.start({})
    assert recorder.state()['periods'] == 1 and recorder.state()['recording']
    recorder.pause()
    assert recorder.state()['paused'] and not recorder.state()['recording']
    recorder.resume()
    assert recorder.state()['periods'] == 2 and recorder.state()['recording']
    recorder.stop()
    manifest = json.loads((tmp_path/session/'manifest.json').read_text(encoding='utf-8'))
    assert [p['end_reason'] for p in manifest['periods']] == ['pause','stop']
    assert manifest['effective_time_rule'] == 'exclude_paused_intervals'
    assert manifest['status'] == 'complete'


def test_video_entry_declares_unverified_alignment(tmp_path):
    path = make_record(tmp_path/'capture')
    video_id = '0123456789abcdef0123456789abcdef'
    video = path/('video_'+video_id+'.mp4')
    video.write_bytes(b'\x00\x00\x00\x18ftypmp42'+b'\x00'*64)
    meta = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    meta['videoAnchors'] = [dict(videoId=video_id,file=video.name,chunks=2,
                                 firstMonotonicMs=1_002.,lastMonotonicMs=1_100.)]
    (path/'manifest.json').write_text(json.dumps(meta),encoding='utf-8')
    with zipfile.ZipFile(export_package(path)) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        entry = manifest['videos'][0]
        assert archive.getinfo(entry['path']).file_size == video.stat().st_size
    assert entry['alignmentStatus'] == 'unverified' and entry['timestampSource'] == 'unverified'
    assert entry['alignmentReasonCode'] == 'no_video_timing_table'
    # 没有逐帧对时表就不写首帧时刻，也不参与公共窗口
    assert entry['frameTimingPath'] is None and entry['firstFrameCaptureMonoNs'] is None
    assert entry['firstFrameEffectiveElapsedUs'] is None and entry['lastFrameEffectiveElapsedUs'] is None
    assert entry['periodId'] == 'p1'
    assert entry['firstChunkReceivedMonoNs'] == 1_002_000_000
    assert entry['lastChunkReceivedMonoNs'] == 1_100_000_000
    assert manifest['frameAlignmentMode'] == 'full_capture'
    assert entry['assetId'] not in manifest['commonAlignedWindow']['includedSourceIds']
