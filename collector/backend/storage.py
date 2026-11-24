import concurrent.futures
import io
import json
import os
import queue
import threading
import time
import uuid
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

FRAME_SCHEMA = pa.schema([
    ('schema_version',pa.string()),('session_id',pa.string()),('device_id',pa.string()),
    ('profile',pa.string()),('device_kind',pa.string()),('foot',pa.string()),('placement',pa.string()),('placement_label',pa.string()),
    ('connection',pa.string()),('source',pa.string()),
    ('event_id',pa.int64()),('frame_index',pa.int32()),('receipt_monotonic_ms',pa.float64()),
    ('receipt_epoch_ms',pa.float64()),('source_receipt_ms',pa.float64()),('source_time_domain',pa.string()),
    ('device_calendar_ms',pa.float64()),('source_frame_number_raw',pa.uint32()),
    ('imu_raw_values',pa.list_(pa.int32())),('imu_physical_values',pa.list_(pa.float64())),
    ('pressure_raw_values',pa.list_(pa.int32())),('pressure_sparse_values',pa.list_(pa.int32())),
    ('pressure_observed_slots',pa.list_(pa.int32())),('pressure_output_format',pa.int32()),
    ('sensor_topology',pa.string()),('gyro_temperature_celsius',pa.float64()),
    ('frame_kind',pa.int32()),('raw_packet_sha256',pa.string()),('raw_frame_hex',pa.string()),
    # 在线零偏的三轴结果必须逐帧落盘，否则导出无法说明该帧用的是哪个偏置
    ('imu_bias_dps',pa.list_(pa.float64())),('imu_bias_ready',pa.bool_()),
    ('imu_bias_revision',pa.int32()),('imu_bias_gate',pa.string()),
])


def atomic_json(path, value):
    temp = path.with_suffix('.pending')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp,path)


class Recorder:
    def __init__(self, root, capacity=20000, chunk_rows=2000):
        self.root = Path(root)
        self.root.mkdir(parents=True,exist_ok=True)
        self.queue = queue.Queue(maxsize=capacity)
        self.chunk_rows = chunk_rows
        self.thread = None
        self.error = None
        self.current = None
        self.metadata = {}
        self.session_id = None
        self.state_name = 'stopped'
        self.periods = []
        self.session_start_monotonic_ms = None
        self.effective_us = 0
        self.written = 0
        self.frames = 0
        self.chunks = 0
        self._lock = threading.RLock()
        self._video_lock = threading.Lock()
        self.videos = {}
        self.recovered = []
        for p in self.root.glob('*/manifest.json'):
            try:
                m = json.loads(p.read_text(encoding='utf-8'))
                if m.get('status') == 'recording':
                    self.recovered.append(m['session_id'])
            except (OSError,ValueError,KeyError):
                continue

    @property
    def accepting(self):
        # 暂停期仍然收包落盘：数据不能静默丢弃，只是不计入有效采集时间
        return self.state_name in ('recording','paused')

    @accepting.setter
    def accepting(self, value):
        # 外部置 False 表示本次记录不可继续；period 的收尾由 stop 与导出兜底处理
        if not value:
            self.state_name = 'stopped'

    def start(self, metadata):
        with self._lock:
            if self.accepting or self.thread and self.thread.is_alive():
                raise ValueError('已有记录任务，请先停止')
            self.session_id = time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:10]
            self.current = self.root/self.session_id
            self.current.mkdir()
            self.queue = queue.Queue(maxsize=self.queue.maxsize)
            self.error = None
            self.written = self.frames = self.chunks = 0
            self.videos = {}
            self.periods = []
            self.effective_us = 0
            self.session_start_monotonic_ms = time.monotonic()*1000
            self.periods.append(self._open_period(self.session_start_monotonic_ms))
            self.metadata = dict(metadata,session_id=self.session_id,schema_version='walkpad.capture.v2',
                                 started_epoch_ms=time.time()*1000,started_monotonic_ms=self.session_start_monotonic_ms,status='recording',
                                 clock_domain='python.monotonic_ns',
                                 effective_time_rule='exclude_paused_intervals',
                                 periods=self.periods,
                                 time_contract='device calendars and source monotonic domains are distinct; no hardware synchronization asserted',
                                 pressure_contract='70 held slots plus independent sparse snapshot and observed slot indices; unit only confirmed by device readback')
            atomic_json(self.current/'manifest.json',self.metadata)
            self.state_name = 'recording'
            self.thread = threading.Thread(target=self._run,name='capture-writer',daemon=True)
            self.thread.start()
            return self.session_id

    def _open_period(self, now):
        start = self.session_start_monotonic_ms
        return dict(period_id='p%d' % (len(self.periods)+1),period_no=len(self.periods)+1,
                    start_monotonic_ms=now,
                    start_session_elapsed_us=int(round((now-start)*1000)),
                    start_effective_elapsed_us=self.effective_us,
                    end_monotonic_ms=None,end_session_elapsed_us=None,end_effective_elapsed_us=None,end_reason=None)

    def _close_period(self, now, reason):
        period = self.periods[-1]
        if period['end_monotonic_ms'] is not None:
            return
        period['end_monotonic_ms'] = now
        period['end_session_elapsed_us'] = int(round((now-self.session_start_monotonic_ms)*1000))
        # 有效时间轴累加的是真实录制时长，暂停区间不进累计
        self.effective_us += int(round((now-period['start_monotonic_ms'])*1000))
        period['end_effective_elapsed_us'] = self.effective_us
        period['end_reason'] = reason

    def _persist_meta(self):
        with self._lock:
            if self.current is None or not self.metadata:
                return
            self.metadata['periods'] = self.periods
            self.metadata['effective_us'] = self.effective_us
            self.metadata['videoAnchors'] = [dict(videoId=key,file=state['file'],chunks=state['next'],
                                                  firstMonotonicMs=state.get('first_monotonic_ms'),
                                                  lastMonotonicMs=state.get('last_monotonic_ms'))
                                             for key,state in self.videos.items()]
            atomic_json(self.current/'manifest.json',self.metadata)

    def pause(self):
        with self._lock:
            if self.state_name != 'recording':
                raise ValueError('当前没有进行中的记录')
            self._close_period(time.monotonic()*1000,'pause')
            self.state_name = 'paused'
            self._persist_meta()
            return self.state()

    def resume(self):
        with self._lock:
            if self.state_name != 'paused':
                raise ValueError('记录未处于暂停状态')
            self.periods.append(self._open_period(time.monotonic()*1000))
            self.state_name = 'recording'
            self._persist_meta()
            return self.state()

    def put(self, kind, value):
        with self._lock:
            if not self.accepting:
                return False
            try:
                self.queue.put_nowait((kind,value))
                return True
            except queue.Full:
                self.error = '记录队列已满，停止接受记录；已保存部分保留，禁止静默丢弃'
                self.accepting = False
                return False

    def checkpoint(self):
        if not self.accepting:
            raise RuntimeError(self.error or '记录未运行，不能确认持久化')
        future = concurrent.futures.Future()
        if not self.put('barrier',future):
            raise RuntimeError(self.error or '记录未运行')
        return future

    def flush(self):
        self.checkpoint().result(timeout=10)

    def stop(self):
        with self._lock:
            if self.state_name == 'recording':
                self._close_period(time.monotonic()*1000,'stop')
            self.state_name = 'stopped'
            try:
                self._persist_meta()
            except OSError:
                pass
        if self.thread:
            self.thread.join(timeout=30)
            if self.thread.is_alive():
                raise RuntimeError('写盘仍在进行，不能宣告记录完成')
        if self.error:
            return self.state()
        return self.state()

    def _run(self):
        frames = []
        last_flush = time.monotonic()
        raw = decisions = None
        try:
            raw = (self.current/'raw_events.jsonl').open('a',encoding='utf-8')
            decisions = (self.current/'decisions.jsonl').open('a',encoding='utf-8')
            while self.state_name != 'stopped' or not self.queue.empty():
                try:
                    kind, value = self.queue.get(timeout=.1)
                except queue.Empty:
                    kind = None
                if kind == 'raw':
                    raw.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
                    self.written += 1
                elif kind == 'frame':
                    frames.append(value)
                    self.frames += 1
                elif kind == 'decision':
                    decisions.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
                elif kind == 'barrier':
                    raw.flush(); decisions.flush()
                    os.fsync(raw.fileno()); os.fsync(decisions.fileno())
                    value.set_result(True)
                if len(frames) >= self.chunk_rows or time.monotonic()-last_flush >= 1:
                    raw.flush(); decisions.flush()
                    os.fsync(raw.fileno()); os.fsync(decisions.fileno())
                    self._parquet(frames)
                    frames.clear()
                    last_flush = time.monotonic()
            raw.flush(); decisions.flush()
            os.fsync(raw.fileno()); os.fsync(decisions.fileno())
            self._parquet(frames)
        except Exception as exc:
            self.error = '写盘失败：'+str(exc)
            self.accepting = False
        finally:
            for f in (raw,decisions):
                if f:
                    f.close()
            self.metadata.update(status='failed' if self.error else 'complete',error=self.error,
                                 stopped_epoch_ms=time.time()*1000,events=self.written,frames=self.frames,chunks=self.chunks)
            try:
                atomic_json(self.current/'manifest.json',self.metadata)
            except OSError as exc:
                self.error = '记录清单写入失败：'+str(exc)

    def _parquet(self, rows):
        if not rows:
            return
        target = self.current/('frames_%06d.parquet' % self.chunks)
        temp = target.with_suffix('.pending')
        schema = FRAME_SCHEMA.with_metadata({b'walkpad.schema':b'capture_frames.v3',b'walkpad.time_sync':b'not_hardware_synchronized'})
        table = pa.Table.from_pylist(rows,schema=schema)
        pq.write_table(table,temp,compression='zstd')
        with temp.open('r+b') as f:
            os.fsync(f.fileno())
        os.replace(temp,target)
        self.chunks += 1

    def state(self):
        return dict(recording=self.state_name == 'recording',paused=self.state_name == 'paused',session_id=self.session_id,path=str(self.current or ''),
                    events=self.written,frames=self.frames,chunks=self.chunks,queued=self.queue.qsize(),
                    periods=len(self.periods),effective_us=self.effective_us,
                    error=self.error,recovered=self.recovered,draining=bool(self.state_name == 'stopped' and self.thread and self.thread.is_alive()))

    def sessions(self):
        result = []
        for p in sorted(self.root.glob('*/manifest.json'),reverse=True):
            try:
                m = json.loads(p.read_text(encoding='utf-8'))
                m['interrupted'] = m.get('status') == 'recording' and m.get('session_id') != (self.session_id if self.accepting else None)
                result.append(m)
            except (OSError,ValueError):
                pass
        return result[:100]

    def directory(self, session):
        if not session or Path(session).name != session or '/' in session or '\\' in session:
            raise ValueError('invalid session')
        path = (self.root/session).resolve()
        if path.parent != self.root.resolve() or not (path/'manifest.json').is_file():
            raise ValueError('记录不存在')
        return path

    def export(self, session):
        path = self.directory(session)
        if session == self.session_id and (self.accepting or self.thread and self.thread.is_alive()):
            raise ValueError('先停止记录并等待写盘完成，再导出')
        target = path/'capture.zip'
        with zipfile.ZipFile(target.with_suffix('.zip.pending'),'w',zipfile.ZIP_DEFLATED) as z:
            for p in path.iterdir():
                if p.is_file() and p.suffix not in ('.zip','.pending'):
                    z.write(p,p.name)
        os.replace(target.with_suffix('.zip.pending'),target)
        return target

    def append_video(self, session, video_id, index, mime, data):
        path = self.directory(session)
        if len(video_id) != 32 or any(c not in '0123456789abcdef' for c in video_id):
            raise ValueError('invalid video id')
        if not 0 <= index <= 100000 or len(data) > 24*1024*1024:
            raise ValueError('invalid video chunk')
        with self._video_lock:
            state = self.videos.get(video_id)
            if state is None:
                if index != 0:
                    raise ValueError('录像块必须从0开始')
                # 只记首块到达的主机单调钟：这是录像唯一的对时证据，不能当成首帧拍摄时刻
                state = self.videos[video_id] = dict(session=session,next=0,last=-1,file=None,
                                                    first_monotonic_ms=time.monotonic()*1000,
                                                    last_monotonic_ms=None)
            if state['session'] != session:
                raise ValueError('video session mismatch')
            if index == state['last']:
                return {'next':state['next']}
            if index != state['next']:
                raise ValueError('录像块不连续，停止录制以免生成损坏文件')
            ext = 'mp4' if 'mp4' in mime else 'webm'
            target = path/('video_'+video_id+'.'+ext)
            with target.open('ab') as f:
                f.write(data); f.flush(); os.fsync(f.fileno())
            state['last'] = index; state['next'] += 1
            state['file'] = target.name; state['last_monotonic_ms'] = time.monotonic()*1000
            if index == 0:
                try:
                    self._persist_meta()
                except OSError:
                    pass
            return {'next':state['next'],'file':target.name}
