import csv
import hashlib
import json
import os
import re
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .lab_attitude import ACC_SCALE, GYRO_SCALE, LabAttitude, measured_axes
from .protocol import WT_HEADER, slot_map
from .version import VERSION


def batches(path):
    for file in sorted(path.glob('frames_*.parquet')):
        yield from pq.ParquetFile(file).iter_batches(batch_size=2000)


def export_txt(path):
    target = path/'wt901.txt'
    with target.open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.writer(f,delimiter='\t',lineterminator='\r\n')
        writer.writerow(WT_HEADER)
        latest = {}
        for batch in batches(path):
            for row in batch.to_pylist():
                if row['profile'] != 'wt':
                    continue
                state = latest.setdefault((row['device_id'],row['connection']),{})
                raw = bytes.fromhex(row['raw_frame_hex'])
                import struct
                kind = row['frame_kind']
                if kind == 0x61:
                    values = list(struct.unpack_from('<9h',raw,2))
                    state['acc'] = [x/2048 for x in values[:3]]
                    state['gyro'] = [x*2000/32768 for x in values[3:6]]
                    state['ang'] = [x*180/32768 for x in values[6:9]]
                elif kind in (0x51,0x52,0x53,0x54):
                    key,scale = {0x51:('acc',1/2048),0x52:('gyro',2000/32768),0x53:('ang',180/32768),0x54:('mag',8000/32768)}[kind]
                    state[key] = [x*scale for x in struct.unpack_from('<3h',raw,2)]
                    state['temperature'] = row['gyro_temperature_celsius']
                if kind == 0x59:
                    state['quat'] = [x/32768 for x in struct.unpack_from('<4h',raw,2)]
                stamp = datetime.fromtimestamp(row['receipt_epoch_ms']/1000).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                writer.writerow([stamp,row['device_id']]+state.get('acc',['']*3)+state.get('gyro',['']*3)+
                                state.get('ang',['']*3)+state.get('mag',['']*3)+state.get('quat',['']*4)+
                                [state.get('temperature',''),'','','',''])
    return target


def export_standard(path):
    target = path/'standard_measurement_frames.parquet'
    specs = json.loads((__import__('pathlib').Path(__file__).parent/'standard_schema.json').read_text(encoding='utf-8'))
    scalar = {'str':pa.string(),'dbl':pa.float64(),'i32':pa.int32(),'u32':pa.uint32(),'i64':pa.int64(),'u64':pa.uint64()}
    fields = []
    for name,kind in specs:
        key = kind.rstrip('?')
        dtype = pa.list_(scalar[key[5:]]) if key.startswith('list.') else scalar[key]
        fields.append(pa.field(name,dtype,nullable=True))
    schema = pa.schema(fields,metadata={b'walkpad.schema_version':b'standard_measurement_frames.parquet.v1',
                b'walkpad.writer_name':b'walkpad_imu_tools_v2',b'walkpad.writer_version':VERSION.encode(),
                b'walkpad.time_sync':b'unverified; effective time unavailable without independent clock synchronization',
                b'walkpad.extra_columns':b'placement_code,gyro_temperature_celsius,device_date,device_time,dropped_frames_before'})
    writer = pq.ParquetWriter(target.with_suffix('.pending'),schema,compression='snappy')
    metadata = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    origin = metadata.get('started_monotonic_ms',0)
    previous = {}
    ordinal = {}
    try:
        for batch in batches(path):
            rows = []
            for r in batch.to_pylist():
                if r['profile'] != 'insole':
                    continue
                key = (r['device_id'],r['connection'])
                prev = previous.get(key)
                index = ordinal.get(key,0)+1
                ordinal[key] = index
                raw = r['pressure_raw_values']
                slots = slot_map(r['foot'],r['sensor_topology'])
                clock = r['device_calendar_ms']
                observed = max(0,int((r['receipt_monotonic_ms']-origin)*1000))
                side = {'l':'left','r':'right'}.get(r.get('foot'),'unspecified')
                code = {'leftInstep':'left_instep','rightInstep':'right_instep','leftWrist':'left_wrist','rightWrist':'right_wrist',
                        'leftAnkle':'left_ankle','rightAnkle':'right_ankle','leftShank':'left_shank','rightShank':'right_shank',
                        'leftThigh':'left_thigh','rightThigh':'right_thigh','pelvis':'pelvis'}
                placement = code.get(r.get('placement'),side+'_instep' if side != 'unspecified' else 'unspecified')
                # 独立 IMU 没有压力通道：留空并标 absent，不写 70 个 0 冒充实测零压
                pressureless = r.get('device_kind') == 'imu'
                frame = dict(schema_id='standard_measurement_frame',schema_version='1.0.0',dataset_id=r['session_id'],
                    frame_identity=r['raw_packet_sha256']+':'+str(r['event_id'])+':'+str(r['frame_index']),product_code='lab',session_id=r['session_id'],
                    processing_input_snapshot_id=r['session_id'],processing_attempt_id='capture-v2',alignment_revision_id='unaligned-v2',
                    capture_stream_id=r['device_id'],assigned_side=side,capture_period_id=r['session_id'],stream_segment_id=r['connection'],connection_epoch=0,
                    source_frame_ordinal=index,source_frame_number_raw=r['source_frame_number_raw'],source_frame_number_unwrapped=None,
                    measurement_contract_id='d56-measurement-v3',raw_envelope_contract_version='ble_notify_raw.v3',
                    sensor_topology_id='d56-14x5-56-legacy-v1' if r['sensor_topology'] == 'v1' else 'd56-14x5-56-side-mirrored-candidate-v3',
                    pressure_conversion_id='d56-adc-identity-v2',alignment_timestamp_definition_id='unverified-cross-clock-v2',
                    source_time_domain='device_local_calendar' if clock is not None else 'unknown',
                    source_time_value=int(clock) if clock is not None else 0,source_time_unit_code='ms',
                    device_time_raw=None,device_time_unit_code='unknown',observed_session_elapsed_us=observed,
                    aligned_session_elapsed_us=None,effective_time_status='unavailable',effective_elapsed_us=None,
                    pressure_raw_bits=[],pressure_raw_values=[] if pressureless else raw,pressure_values=[] if pressureless else [raw[i] for i in slots],
                    pressure_raw_unit_code='raw',pressure_value_unit_code='raw',
                    sensor_layout_version=r['sensor_topology'],conversion_version='d56-adc-identity-v2',
                    imu_raw_bits=[],imu_raw_values=r['imu_raw_values'],imu_physical_values=r['imu_physical_values'],
                    axis_order_code='ax_ay_az_gx_gy_gz',accel_unit_code='g',gyro_unit_code='deg_per_s',imu_unit_status='confirmed',
                    source_asset_ids=[r['device_id']],source_chunk_indices=[0],source_record_indices=[r['event_id']],
                    source_native_sequences=[r['source_frame_number_raw']],source_native_monotonic_ns=[],source_session_elapsed_us=[observed],
                    raw_packet_sha256=r['raw_packet_sha256'],capture_disposition='effective',parse_status='parsed',frame_status='valid',
                    pressure_status='absent' if pressureless else 'available',imu_status='available',previous_source_frame_ordinal=index-1 if prev else None,
                    source_frame_ordinal_delta=1 if prev else None,effective_elapsed_delta_us=None,
                    continuity_status='boundary_start' if prev is None else 'unverified',
                    quality_flag_codes=['cross_clock_alignment_unverified','sequence_gap_not_equated_to_imu_loss']+(['pressure_channel_absent'] if pressureless else []),
                    placement_code=placement,gyro_temperature_celsius=r['gyro_temperature_celsius'],device_date='',device_time='',dropped_frames_before=None)
                rows.append(frame)
                previous[key] = r
            if rows:
                writer.write_table(pa.Table.from_pylist(rows,schema=schema))
    finally:
        writer.close()
        os.replace(target.with_suffix('.pending'),target)
    return target


# ---------------------------------------------------------------- 测量包
# Lab v2.6 测量包：manifest.json 的 sources/outputs
# 指向 frames/<设备目录>/frame-00001.parquet，周期与片段写在 timeline/footer.parquet。
# 列名、struct 六轴、目录命名都按 walkpad_measurement_core 的定义，改名会让 app 侧读包失败。

# 与 labpad laboratory_measurement_columns.dart 一致：缺测列也按同一类型写出。
LAB_V2_COLUMNS = {
    'frame': {
        'sourceId':'VARCHAR','segmentId':'VARCHAR','index':'BIGINT','frameType':'VARCHAR',
        'deviceTimeRaw':'VARCHAR','deviceTimeDefinition':'VARCHAR','deviceSequence':'BIGINT',
        'timeUs':'BIGINT','deviceTimeUs':'BIGINT','nativeReceiveNs':'BIGINT','connectionEpoch':'BIGINT',
        'periodId':'VARCHAR','effectiveElapsedUs':'BIGINT','validity':'VARCHAR',
        'pitchDeg':'DOUBLE','sideTiltDeg':'DOUBLE','relativeHeadingDeg':'DOUBLE','referenceId':'VARCHAR',
        'pressureAdc':'INTEGER[]',
        'accelerationRaw':'STRUCT(x INTEGER,y INTEGER,z INTEGER)',
        'accelerationG':'STRUCT(x DOUBLE,y DOUBLE,z DOUBLE)',
        'angularVelocityRaw':'STRUCT(x INTEGER,y INTEGER,z INTEGER)',
        'angularVelocityDps':'STRUCT(x DOUBLE,y DOUBLE,z DOUBLE)',
        'imuCalibration':'STRUCT(status VARCHAR,initializationId VARCHAR,biasRevision BIGINT,'
                         'accelerationG STRUCT(x DOUBLE,y DOUBLE,z DOUBLE),'
                         'angularVelocityCompensatedDps STRUCT(x DOUBLE,y DOUBLE,z DOUBLE),'
                         'gyroBiasDps STRUCT(x DOUBLE,y DOUBLE,z DOUBLE),accelerationCompensated BOOLEAN)',
        'temperatureRaw':'INTEGER','temperatureCelsius':'DOUBLE',
        'magneticRaw':'STRUCT(x INTEGER,y INTEGER,z INTEGER)',
        'magneticMicrotesla':'STRUCT(x DOUBLE,y DOUBLE,z DOUBLE)',
        'quaternionRaw':'STRUCT(w INTEGER,x INTEGER,y INTEGER,z INTEGER)',
        'quaternion':'STRUCT(w DOUBLE,x DOUBLE,y DOUBLE,z DOUBLE)',
    },
    'notify': {
        'notifyIndex':'BIGINT','sourceId':'VARCHAR','segmentId':'VARCHAR','assetId':'VARCHAR',
        'recordIndex':'BIGINT','connectionId':'VARCHAR','connectionEpoch':'BIGINT','nativeSequence':'BIGINT',
        'nativeReceiveNs':'BIGINT','payloadBytes':'INTEGER','characteristic':'VARCHAR',
    },
    'link': {
        'sourceId':'VARCHAR','segmentId':'VARCHAR','index':'BIGINT','assetId':'VARCHAR','recordIndex':'BIGINT',
        'connectionEpoch':'BIGINT','nativeReceiveNs':'BIGINT','payloadStart':'INTEGER','payloadEnd':'INTEGER',
    },
    'issue': {
        'sourceId':'VARCHAR','segmentId':'VARCHAR','frameIndex':'BIGINT','code':'VARCHAR','recordIndex':'BIGINT',
        'assetId':'VARCHAR','nativeSequence':'BIGINT','previousSequence':'BIGINT','deviceSequence':'BIGINT',
        'count':'BIGINT','component':'VARCHAR','scope':'VARCHAR',
        'rawLocations':'STRUCT(assetId VARCHAR,recordIndex BIGINT,payloadStart INTEGER,payloadEnd INTEGER)[]',
        'discardedBytes':'BIGINT','observedSessionUs':'BIGINT',
    },
}

LAB_UNITS = {
    'pitchDeg':'degrees about body X','sideTiltDeg':'degrees about body Y',
    'relativeHeadingDeg':'relative yaw in degrees [-180,180]; null until tracking',
    'referenceId':'attitude scope and generation; compare headings only within the same source and reference',
    'pressureAdc':'ADC count','accelerationRaw':'device count','accelerationG':'g including gravity',
    'angularVelocityRaw':'device count','angularVelocityDps':'deg/s',
    'imuCalibration.angularVelocityCompensatedDps':'deg/s','imuCalibration.gyroBiasDps':'deg/s in output axes',
    'imuCalibration.accelerationG':'g including gravity; not compensated',
    'temperatureRaw':'device count','temperatureCelsius':'degC',
    'magneticRaw':'device count','magneticMicrotesla':'uT',
    'quaternionRaw':'device count','quaternion':'dimensionless wxyz',
}

LAB_PLACEMENTS = {'leftInsole','rightInsole','leftInstep','rightInstep','leftWrist','rightWrist',
                  'leftAnkle','rightAnkle','leftShank','rightShank','leftThigh','rightThigh','pelvis','unspecified'}
# 鞋垫装在鞋内，app 侧侧别只有 left/right -> 鞋垫位置；独立 IMU 保留脚背等佩戴位置
INSOLE_PLACEMENT = {'leftInstep':'leftInsole','rightInstep':'rightInsole'}
TIME_DEFINITION = 'd56-synchronized-local-calendar-session-v2'
MEASUREMENT_CONTRACT = 'd56-measurement-v3'
CONVERSION_PROFILE = 'd56-imu-acc488ug-gyro70mdps-y-negx-z-v2'


def _tri(dtype):
    return pa.struct([('x',dtype),('y',dtype),('z',dtype)])


def _quad(dtype):
    return pa.struct([('w',dtype),('x',dtype),('y',dtype),('z',dtype)])


PKG_FRAME_SCHEMA = pa.schema([
    ('sourceId',pa.string()),('segmentId',pa.string()),('index',pa.int64()),('frameType',pa.string()),
    ('deviceTimeRaw',pa.string()),('deviceTimeDefinition',pa.string()),('deviceSequence',pa.int64()),
    ('timeUs',pa.int64()),('deviceTimeUs',pa.int64()),('nativeReceiveNs',pa.int64()),('connectionEpoch',pa.int64()),
    ('periodId',pa.string()),('effectiveElapsedUs',pa.int64()),('validity',pa.string()),
    ('pitchDeg',pa.float64()),('sideTiltDeg',pa.float64()),('relativeHeadingDeg',pa.float64()),('referenceId',pa.string()),
    ('pressureAdc',pa.list_(pa.int32())),
    ('accelerationRaw',_tri(pa.int32())),('accelerationG',_tri(pa.float64())),
    ('angularVelocityRaw',_tri(pa.int32())),('angularVelocityDps',_tri(pa.float64())),
    ('imuCalibration',pa.struct([('status',pa.string()),('initializationId',pa.string()),('biasRevision',pa.int64()),
                                 ('accelerationG',_tri(pa.float64())),('angularVelocityCompensatedDps',_tri(pa.float64())),
                                 ('gyroBiasDps',_tri(pa.float64())),('accelerationCompensated',pa.bool_())])),
    ('temperatureRaw',pa.int32()),('temperatureCelsius',pa.float64()),
    ('magneticRaw',_tri(pa.int32())),('magneticMicrotesla',_tri(pa.float64())),
    ('quaternionRaw',_quad(pa.int32())),('quaternion',_quad(pa.float64())),
])


def _tri_of(values):
    if not values or len(values) < 3 or any(v is None for v in values[:3]):
        return None
    return {'x':values[0],'y':values[1],'z':values[2]}


def _device_stamp(ms):
    if ms is None:
        return None
    seconds, millis = divmod(round(ms),1000)
    return datetime.fromtimestamp(seconds,timezone.utc).strftime('%Y%m%d%H%M%S')+'%03d' % millis


def _lab_placement(value, kind, foot=None):
    if kind == 'insole':
        # 旧记录没有位置列，只能按脚别还原成鞋垫位置
        mapped = INSOLE_PLACEMENT.get(value) or value
        if mapped in ('leftInsole','rightInsole'):
            return mapped
        return {'l':'leftInsole','r':'rightInsole'}.get(foot,'unspecified')
    if value in LAB_PLACEMENTS:
        return value
    return {'l':'leftInstep','r':'rightInstep'}.get(foot,'unspecified')


def _lab_directory(family, name, address, used):
    """按 app 规则拼设备目录：<类型>_<蓝牙名>_<首2位>-<末4位>，冲突时加长前缀。"""
    text = name or '未记录'
    safe = ''.join(ch if ch not in '/\\:*?"<>|' and ord(ch) >= 32 else '_' for ch in text[:40])
    hexaddr = (address or '').replace(':','').replace('-','').upper()
    valid = re.fullmatch(r'[0-9A-F]{12}|[0-9A-F]{32}',hexaddr) is not None
    lead = 2

    def candidate():
        if not valid:
            return '%s_%s_Device' % (family,safe)
        return '%s_%s_%s-%s' % (family,safe,hexaddr[:lead],hexaddr[-4:])

    directory = candidate()
    while directory.lower() in used and valid and lead < len(hexaddr)-4:
        lead += 2
        directory = candidate()
    base, number = directory, 2
    while directory.lower() in used:
        directory = '%s_%d' % (base,number)
        number += 1
    used.add(directory.lower())
    return directory


def _package_source(row, known, used):
    key = row['device_id']
    address = known.get('address') or (key.split(':',1)[1] if ':' in key else '')
    name = known.get('name') or address or key
    kind = row.get('device_kind') or 'insole'
    family = 'walkpadImu' if kind == 'imu' else 'd56'
    return dict(sourceId=uuid.uuid4().hex.upper()[:26],device=row['device_id'],kind=kind,family=family,
                placement=_lab_placement(row.get('placement') or known.get('placement'),kind,row.get('foot')),
                name=name,address=address,
                directory=_lab_directory(family,name,address,used),
                pressure=kind != 'imu',
                topology='d56-14x5-56-legacy-v1' if row.get('sensor_topology') == 'v1' else 'd56-14x5-56-side-mirrored-candidate-v3',
                rows=0,part=0,pending=[],index={},attitude={},
                first_receive_ms=None,last_receive_ms=None)


def _package_row(r, src, zero, periods):
    cal = r['device_calendar_ms']
    # 设备日历是唯一可比的采样时间；拿不到就留空，不拿接收时刻顶替
    if cal is not None and zero is not None:
        device_us = int(round(cal*1000))
        elapsed = device_us - zero
    else:
        elapsed = None
        device_us = None
    raw = r['imu_raw_values'] or []
    # Packaged physical values use the Lab scaling; live channels and raw records stay untouched.
    acc_raw, gyro_raw = _tri_of(raw[:3]), _tri_of(raw[3:6])
    acc = measured_axes(raw[:3],ACC_SCALE) if acc_raw is not None else None
    gyro = measured_axes(raw[3:6],GYRO_SCALE) if gyro_raw is not None else None
    pressure = r['pressure_raw_values'] or []
    if src['pressure'] and pressure:
        pressure = [pressure[i] for i in slot_map(r['foot'],r['sensor_topology'])]
    temp = r['gyro_temperature_celsius']
    # 偏置与角速度同在 (y,-x,z) 输出坐标，直接相减即可，不需要再做轴变换
    bias = r.get('imu_bias_dps')
    ready = bool(r.get('imu_bias_ready')) and bias is not None and len(bias) == 3
    compensated = [value - offset for value, offset in zip(gyro,bias)] if ready and gyro is not None else None
    revision = int(r.get('imu_bias_revision') or 0)
    segment = r['connection']
    attitude = src['attitude'].get(segment)
    if attitude is None:
        attitude = src['attitude'][segment] = LabAttitude(f'{src["sourceId"]}/{segment}')
    pitch, side, heading, reference = attitude.add(raw,device_us) or (None,None,None,None)
    index = src['index'].get(segment,0)
    src['index'][segment] = index+1
    src['rows'] += 1
    period_id, effective = _locate_period(periods,r['receipt_monotonic_ms'])
    # 公共窗口按接收单调钟取交集；落在暂停空隙的帧不算有效覆盖
    if period_id is not None:
        mono = r['receipt_monotonic_ms']
        src['first_receive_ms'] = mono if src['first_receive_ms'] is None else min(src['first_receive_ms'],mono)
        src['last_receive_ms'] = mono if src['last_receive_ms'] is None else max(src['last_receive_ms'],mono)
    return dict(sourceId=src['sourceId'],segmentId=segment,index=index,
                frameType='insole' if src['pressure'] else 'inertial',
                deviceTimeRaw=_device_stamp(cal),deviceTimeDefinition=TIME_DEFINITION,
                deviceSequence=None if r['source_frame_number_raw'] is None else int(r['source_frame_number_raw']),
                timeUs=elapsed,deviceTimeUs=device_us,
                nativeReceiveNs=int(round(r['receipt_monotonic_ms']*1e6)),connectionEpoch=0,
                periodId=period_id,effectiveElapsedUs=effective,
                validity='valid' if elapsed is not None and period_id else ('unknown' if elapsed is None else 'outside_period'),
                pitchDeg=pitch,sideTiltDeg=side,relativeHeadingDeg=heading,referenceId=reference,
                pressureAdc=[int(x) for x in pressure] if src['pressure'] else None,
                accelerationRaw=acc_raw,accelerationG=_tri_of(acc),
                angularVelocityRaw=gyro_raw,angularVelocityDps=_tri_of(gyro),
                # 偏置来自采集时的在线静止估计，逐帧冻结；没有偏置的帧不拿零值冒充
                imuCalibration=dict(status='ready' if ready else 'uncalibrated',
                                    initializationId='%s-bias-%d' % (src['sourceId'],revision) if ready else None,
                                    biasRevision=revision if ready else 0,
                                    accelerationG=_tri_of(acc),
                                    angularVelocityCompensatedDps=_tri_of(compensated) if compensated else None,
                                    gyroBiasDps=_tri_of(bias) if ready else None,
                                    accelerationCompensated=False),
                temperatureRaw=None if temp is None else int(round(temp*100)),
                temperatureCelsius=None if temp is None else float(temp),
                magneticRaw=None,magneticMicrotesla=None,quaternionRaw=None,quaternion=None)


def _write_part(root, src, rows, outputs):
    src['part'] += 1
    rel = 'frames/%s/frame-%05d.parquet' % (src['directory'],src['part'])
    target = root/rel
    target.parent.mkdir(parents=True,exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows,schema=PKG_FRAME_SCHEMA),target,
                   compression='zstd',row_group_size=8192)
    outputs.append(dict(path=rel,sourceId=src['sourceId'],part=src['part'],rows=len(rows),
                        bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest()))


def _flush_source(root, src, outputs):
    # 空块不写文件：outputs 里每个 part 都真实存在
    while len(src['pending']) >= 20000:
        _write_part(root,src,src['pending'][:20000],outputs)
        src['pending'] = src['pending'][20000:]
    if src['pending']:
        _write_part(root,src,src['pending'],outputs)
        src['pending'] = []


def _sequence_delta(previous, current):
    delta = (current-previous) & 0xffffffff
    if delta == 1:
        return None
    if delta == 0:
        return 'sequence_duplicate'
    return 'sequence_gap' if delta < 0x80000000 else 'sequence_backward_or_reset'


def _export_periods(meta, path):
    """录制期写下的周期边界；老记录没有周期就退化成整段一个周期，不假装有暂停。"""
    recorded = [dict(p) for p in (meta.get('periods') or [])
                if isinstance(p, dict) and p.get('start_monotonic_ms') is not None]
    if recorded:
        return recorded
    low = high = None
    for batch in batches(path):
        for value in batch.column('receipt_monotonic_ms').to_pylist():
            if value is None:
                continue
            low = value if low is None else min(low, value)
            high = value if high is None else max(high, value)
    if low is None:
        return []
    span = int(round(high-low+1))
    return [dict(period_id='p1', period_no=1, start_monotonic_ms=low, end_monotonic_ms=high+1,
                 start_session_elapsed_us=0, end_session_elapsed_us=span,
                 start_effective_elapsed_us=0, end_effective_elapsed_us=span,
                 end_reason='inferred_single_period')]


def _locate_period(periods, receipt_ms):
    """按接收单调钟定位帧所属周期；落在暂停空隙里的帧不给周期，也不计入有效时间轴。"""
    if receipt_ms is None:
        return None, None
    for period in periods:
        start = period['start_monotonic_ms']
        end = period.get('end_monotonic_ms')
        if receipt_ms < start or (end is not None and receipt_ms >= end):
            continue
        offset = int(round((receipt_ms-start)*1000))
        return period['period_id'], int(period.get('start_effective_elapsed_us') or 0)+offset
    return None, None


def _effective_at_mono(periods, mono_ms):
    """接收单调钟映射到有效时间轴；落在暂停空隙里的时刻没有有效位置。"""
    if mono_ms is None:
        return None
    for period in periods:
        start = period['start_monotonic_ms']
        end = period.get('end_monotonic_ms')
        if mono_ms < start or (end is not None and mono_ms >= end):
            continue
        return int(period.get('start_effective_elapsed_us') or 0)+int(round((mono_ms-start)*1000))
    return None


def _common_window(periods, sources):
    """各来源有效帧覆盖在主机接收单调钟上的交集。

    接收钟同机同域，可作公共时间轴；它不含蓝牙传输时延修正，所以只声明覆盖重叠，
    不声明任何硬件同步。
    """
    spans = [(s['sourceId'],s['first_receive_ms'],s['last_receive_ms']) for s in sources
             if s['first_receive_ms'] is not None]
    low = max(s[1] for s in spans) if spans else None
    high = min(s[2] for s in spans) if spans else None
    start_eff = _effective_at_mono(periods,low)
    end_eff = _effective_at_mono(periods,high)
    overlap = start_eff is not None and end_eff is not None and start_eff <= end_eff
    return dict(includedSourceIds=[s[0] for s in spans],
                startMonoNs=None if not overlap else int(round(low*1e6)),
                endMonoNs=None if not overlap else int(round(high*1e6)),
                startEffectiveElapsedUs=start_eff if overlap else None,
                endEffectiveElapsedUs=end_eff if overlap else None,
                durationUs=end_eff-start_eff if overlap else None)


def _period_at_elapsed(periods, elapsed_us):
    """段边界落在会话轴上，按起始时刻取所属周期；落在暂停空隙里的段不给周期。"""
    for period in periods:
        start = int(period.get('start_session_elapsed_us') or 0)
        end = period.get('end_session_elapsed_us')
        if elapsed_us < start or (end is not None and elapsed_us >= end):
            continue
        return period['period_id']
    return None


def _package_segments(path, session_zero):
    """第一遍取每台设备的日历零点，第二遍统计片段边界、序号连续性与时间相邻对。

    片段边界落在会话单调钟上（receipt），设备日历只用于逐设备零点与相邻对。
    """
    zeros = {}
    for batch in batches(path):
        for r in batch.to_pylist():
            cal = r['device_calendar_ms']
            if r['profile'] != 'insole' or cal is None:
                continue
            key = r['device_id']
            us = int(round(cal*1000))
            zeros[key] = us if key not in zeros else min(zeros[key],us)
    segments, order = {}, []
    for batch in batches(path):
        for r in batch.to_pylist():
            if r['profile'] != 'insole':
                continue
            key = (r['device_id'],r['connection'])
            seg = segments.get(key)
            if seg is None:
                seg = segments[key] = dict(sourceId=r['device_id'],segmentId=r['connection'],
                                           comparisons=0,timeComparisons=0,anomalies={},
                                           previous=None,lastTime=None,start=None,end=None)
                order.append(key)
            cal = r['device_calendar_ms']
            if cal is not None and zeros.get(r['device_id']) is not None:
                device_elapsed = int(round(cal*1000)) - zeros[r['device_id']]
                if seg['lastTime'] is not None:
                    seg['timeComparisons'] += 1
                seg['lastTime'] = device_elapsed
            receipt = r['receipt_monotonic_ms']
            if receipt is not None and session_zero is not None:
                elapsed = int(round((receipt-session_zero)*1000))
                seg['start'] = elapsed if seg['start'] is None else min(seg['start'],elapsed)
                seg['end'] = elapsed+1 if seg['end'] is None else max(seg['end'],elapsed+1)
            seq = r['source_frame_number_raw']
            if seq is not None:
                previous = seg['previous']
                seg['previous'] = int(seq)
                if previous is not None:
                    seg['comparisons'] += 1
                    code = _sequence_delta(previous,int(seq))
                    if code:
                        seg['anomalies'][code] = seg['anomalies'].get(code,0)+1
    return zeros, [segments[key] for key in order]


def _footer_parquet(target, sections):
    rows = {'section':[],'body':[]}
    for name, value in sections.items():
        rows['section'].append(name)
        rows['body'].append(json.dumps(value,ensure_ascii=False,separators=(',',':')))
    schema = pa.schema([('section',pa.string()),('body',pa.string())])
    pq.write_table(pa.table(rows,schema=schema),target,compression='zstd')


def export_package(path):
    path = Path(path)
    target = path/'capture_package.zip'
    staging = path/'.package.pending'
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    meta = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    known = {(d.get('id') or ''):d for d in meta.get('devices') or []}
    periods = _export_periods(meta,path)
    session_zero = meta.get('started_monotonic_ms')
    if session_zero is None and periods:
        session_zero = periods[0]['start_monotonic_ms']
    # 每个来源以首帧设备日历为零点：采集端没有手机-设备对时，不能假装有共同基准
    zeros, segments = _package_segments(path,session_zero)
    sources, outputs, used = {}, [], set()
    try:
        for batch in batches(path):
            for r in batch.to_pylist():
                if r['profile'] != 'insole':
                    continue
                src = sources.get(r['device_id'])
                if src is None:
                    src = sources[r['device_id']] = _package_source(r,known.get(r['device_id']) or {},used)
                src['pending'].append(_package_row(r,src,zeros.get(r['device_id']),periods))
                if len(src['pending']) >= 20000:
                    _flush_source(staging,src,outputs)
        for src in sources.values():
            _flush_source(staging,src,outputs)
        anchors = {a.get('file'):a for a in (meta.get('videoAnchors') or []) if a.get('file')}
        videos = []
        for i, video in enumerate(sorted(p for p in path.iterdir() if p.is_file() and p.name.startswith('video_')),1):
            (staging/'videos').mkdir(exist_ok=True)
            rel = 'videos/clip-%03d%s' % (i,video.suffix)
            shutil.copyfile(video,staging/rel)
            digest = hashlib.sha256(video.read_bytes()).hexdigest()
            anchor = anchors.get(video.name) or {}
            first = anchor.get('firstMonotonicMs')
            last = anchor.get('lastMonotonicMs')
            # 没有逐帧对时表：录像只留下首末块到达的主机时刻，因此不声明已对齐
            video_period = _locate_period(periods,first)[0] if first is not None else None
            videos.append(dict(assetId=video.stem,periodId=video_period,segmentId=None,path=rel,
                               bytes=video.stat().st_size,sha256=digest,
                               alignmentStatus='unverified',timestampSource='unverified',
                               alignmentReasonCode='no_video_timing_table',frameTimingPath=None,
                               firstFrameVideoPtsUs=None,firstFrameCaptureMonoNs=None,
                               firstFrameEffectiveElapsedUs=None,
                               lastFrameVideoPtsUs=None,lastFrameCaptureMonoNs=None,
                               lastFrameEffectiveElapsedUs=None,
                               frameCount=0,droppedFrameCount=0,uncertaintyUs=None,
                               firstChunkReceivedMonoNs=None if first is None else int(round(first*1e6)),
                               lastChunkReceivedMonoNs=None if last is None else int(round(last*1e6))))
            outputs.append(dict(path=rel,assetId=video.stem,periodId=video_period,
                                bytes=video.stat().st_size,sha256=digest))
        name_of = {src['device']:src for src in sources.values()}
        period_rows = [dict(period_id=p['period_id'],period_no=p.get('period_no') or i+1,
                            start_session_elapsed_us=int(p.get('start_session_elapsed_us') or 0),
                            end_session_elapsed_us=int(p.get('end_session_elapsed_us') or 0),
                            start_effective_elapsed_us=int(p.get('start_effective_elapsed_us') or 0),
                            end_effective_elapsed_us=int(p.get('end_effective_elapsed_us') or 0))
                       for i,p in enumerate(periods)]
        # 每个来源的第一段是录制开始，其余都是重连
        starts, first_seen = {}, set()
        for seg in segments:
            if seg['sourceId'] in first_seen:
                starts[seg['segmentId']] = ('reconnect','reconnect')
            else:
                first_seen.add(seg['sourceId'])
                starts[seg['segmentId']] = ('record_start','recordStart')
        segment_facts = [dict(segment_id=seg['segmentId'],stream_id=name_of[seg['sourceId']]['sourceId'],
                              period_id=_period_at_elapsed(periods,seg['start'] or 0),
                              start_session_elapsed_us=seg['start'] or 0,
                              end_session_elapsed_us=seg['end'] or 0,
                              start_reason_code=starts[seg['segmentId']][0])
                         for seg in segments if seg['sourceId'] in name_of]
        mappings = []
        for src in sources.values():
            for seg in segments:
                if seg['sourceId'] != src['device']:
                    continue
                mappings.append(dict(
                    sourceId=src['sourceId'],segmentId=seg['segmentId'],
                    startReason=starts[seg['segmentId']][1],
                    timeBasis='deviceClock',timeDefinitionId=TIME_DEFINITION,timeUncertaintyUs=None,
                    mapping=dict(sourceZeroUs=zeros.get(seg['sourceId']),syncAttemptId=None,
                                 deviceClockDomain='device_local_calendar_ordinal_us',
                                 formula='deviceTimeUs - sourceZeroUs',
                                 sessionStartCalendarUs=zeros.get(seg['sourceId']),
                                 appliedOffsetUs=0,offsetBasis='legacy_zero_fallback',
                                 estimatedUncertaintyUs=None)))
        footer = staging/'timeline'/'footer.parquet'
        footer.parent.mkdir(parents=True,exist_ok=True)
        _footer_parquet(footer,{'periods':period_rows,'segments':segment_facts,'eventOrder':[],
                                'mappings':mappings,**({'videos':videos} if videos else {})})
        outputs.append(dict(path='timeline/footer.parquet',bytes=footer.stat().st_size,
                            sha256=hashlib.sha256(footer.read_bytes()).hexdigest()))
        guide = [
            '设备目录对照表',
            'frames/ 存放标准帧，evidence/ 存放对应证据。sourceId 是帧中使用的数据源编号。',
            '测量包按 Lab 换算原始六轴；在线偏置补偿结果见同帧 imuCalibration.angularVelocityCompensatedDps（°/s）。',
            '姿态由各连接段的原始六轴和设备采样时间重算；航向未跟踪时为空，只在同一 sourceId、同一 referenceId 内比较。',
            'imuCalibration.gyroBiasDps 是该帧实际采用的三轴零偏（输出坐标 °/s）；biasRevision 在重连、偏置跳变或复位时递增。',
            'imuCalibration.accelerationG 仅透传加速度（g），accelerationCompensated=false；不可用结果为 null。',
            '偏置由采集时的在线静止估计得到，工作台不保存静置原件：initializationId 是偏置会话标识（<sourceId>-bias-<biasRevision>），不对应 calibration/ 原件，manifest.calibrationOriginals 为空；偏置未就绪的帧 status 为 uncalibrated，补偿与偏置列为 null。',
            'manifest.sessionTimeline 是有效时间轴（排除暂停）；commonAlignedWindow 是各来源有效帧覆盖在主机接收单调钟上的交集，未扣除蓝牙传输时延，只声明覆盖重叠，不构成硬件同步。',
            '视频没有逐帧对时表：videos[].alignmentStatus 恒为 unverified，只记首末块到达的主机时刻，periodId 指首块到达时所在周期，不能据此把画面帧与采集帧对齐。',
            '',
        ]
        for src in sources.values():
            guide.append('目录：%s\n设备类型：%s\n蓝牙名称：%s\n完整设备标识（MAC 或 iOS 外设标识符）：%s\nsourceId：%s\n设备绑定编号：未记录\n'
                         % (src['directory'],src['family'],src['name'],src['address'] or '未记录',src['sourceId']))
        (staging/'设备目录对照.txt').write_text('\n'.join(guide),encoding='utf-8')
        outputs.append(dict(path='设备目录对照.txt',bytes=(staging/'设备目录对照.txt').stat().st_size,
                            sha256=hashlib.sha256((staging/'设备目录对照.txt').read_bytes()).hexdigest()))
        common = _common_window(periods,sources.values())
        manifest = dict(
            schemaVersion='lab_measurement_package.v2.6',recordId=uuid.uuid4().hex.upper()[:26],
            title=meta.get('session_id'),inputManifestId=meta.get('session_id'),
            normalizationVersion='lab-normalization.1',
            sources=[dict(sourceId=s['sourceId'],deviceFamily=s['family'],placement=s['placement'],
                          protocolVersion=MEASUREMENT_CONTRACT,conversionProfileId=CONVERSION_PROFILE,
                          frameTypes=['insole' if s['pressure'] else 'inertial'],
                          pressureLayoutId=s['topology'] if s['pressure'] else None,
                          pressureChannelCount=56 if s['pressure'] else None,
                          rawAxesId='d56-register-xyz',outputAxesId='d56-y-negx-z',quaternionConventionId=None,
                          directory=s['directory'],bluetoothName=s['name'],deviceType=s['family'],
                          deviceIdentifier=s['address'] or None,deviceBindingId=None) for s in sources.values()],
            inputs=[],outputs=outputs,physicalSchemas=LAB_V2_COLUMNS,calibrationOriginals=[],
            includeOriginals=False,includeVideos=bool(videos),
            videos=videos,commonAlignedWindow=common,
            frameAlignmentMode='video_aligned_window' if any(v['alignmentStatus'] == 'aligned' for v in videos)
            else 'full_capture',
            sessionTimeline=dict(
                clockDomain=meta.get('clock_domain') or 'python.monotonic_ns',
                sessionStartMonoNs=None if session_zero is None else int(round(session_zero*1e6)),
                sessionEndMonoNs=None if not periods or periods[-1].get('end_monotonic_ms') is None
                else int(round(periods[-1]['end_monotonic_ms']*1e6)),
                sessionStartUtcIso=None if meta.get('started_epoch_ms') is None
                else datetime.fromtimestamp(meta['started_epoch_ms']/1000,timezone.utc).isoformat(),
                effectiveTimeRule='exclude_pauses_half_open_periods',
                effectiveDurationUs=int(period_rows[-1]['end_effective_elapsed_us']) if period_rows else 0,
                periods=[dict(periodId=p['period_id'],periodNo=p['period_no'],
                              startMonoNs=None if periods[i].get('start_monotonic_ms') is None
                              else int(round(periods[i]['start_monotonic_ms']*1e6)),
                              endMonoNs=None if periods[i].get('end_monotonic_ms') is None
                              else int(round(periods[i]['end_monotonic_ms']*1e6)),
                              startSessionElapsedUs=p['start_session_elapsed_us'],
                              endSessionElapsedUs=p['end_session_elapsed_us'],
                              startEffectiveElapsedUs=p['start_effective_elapsed_us'],
                              endEffectiveElapsedUs=p['end_effective_elapsed_us'],
                              endReason=periods[i].get('end_reason')) for i,p in enumerate(period_rows)]),
            units=LAB_UNITS,
            frameKey=['sourceId','segmentId','index'],
            recordIndexBasis='header=0; notifyIndex=0 for first notify',
            intervalRule='startInclusive,endExclusive; period and source segment must both contain timeUs',
            coverageRule='comparisons > 0 means evaluated adjacent pairs; 0 means insufficient pairs; anomaly counts only describe observed data',
            metadataTypes={'pressureChannelCount':'nullable int32','sourceId':'string','segmentId':'string',
                           'timeUncertaintyUs':'nullable int64 us'},
            deviceTimeRawFormat='D56 yyyyMMddHHmmssSSS; local calendar, no timezone',
            deviceTimeUnit='us',nativeReceiveUnit='ns',recordTimeUnit='us',
            checks=[dict(sourceId=name_of[seg['sourceId']]['sourceId'],segmentId=seg['segmentId'],
                         sequenceComparisons=seg['comparisons'],sequenceSupported=True,
                         timeComparisons=seg['timeComparisons'],nativeSequenceComparisons=0,
                         anomalies=seg['anomalies'],sourceDecodeCheckExecuted=False,
                         sourceDecodeErrorCount=None)
                    for seg in segments if seg['sourceId'] in name_of])
        (staging/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=1),encoding='utf-8')
        with zipfile.ZipFile(target.with_suffix('.zip.pending'),'w',zipfile.ZIP_DEFLATED) as z:
            for file in sorted(staging.rglob('*')):
                if file.is_file():
                    z.write(file,str(file.relative_to(staging)))
        os.replace(target.with_suffix('.zip.pending'),target)
    finally:
        shutil.rmtree(staging,ignore_errors=True)
    return target
