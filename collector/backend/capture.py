import asyncio
import base64
import collections
import json
import math
import re
import statistics
import time
import uuid
from pathlib import Path

from bleak import BleakClient, BleakScanner
from bleak.exc import BleakError

from .algorithm import Algorithm
from .protocol import Decoder, command_packet, crc_ccitt, crc_measurement, demo_packet, slot_map
from .storage import Recorder, atomic_json
from .lab_attitude import LabAttitude
from .version import VERSION

INSOLE_NOTIFY = '00002a37-0000-1000-8000-00805f9b34fb'
INSOLE_WRITE = '0000ffe1-0000-1000-8000-00805f9b34fb'
WT_NOTIFY = '0000ffe4-0000-1000-8000-00805f9a34fb'
WT_WRITE = '0000ffe9-0000-1000-8000-00805f9a34fb'
WT_SERVICE = '0000ffe5-0000-1000-8000-00805f9a34fb'

# 位置 -> (显示名, 脚别)；只有左右脚背带脚别进左右脚算法，其余位置照常采集与零偏
PLACEMENTS = {'leftInstep':('左脚背','l'),'rightInstep':('右脚背','r'),
              'leftWrist':('左手腕',None),'rightWrist':('右手腕',None),
              'leftAnkle':('左踝',None),'rightAnkle':('右踝',None),
              'leftShank':('左小腿',None),'rightShank':('右小腿',None),
              'leftThigh':('左大腿',None),'rightThigh':('右大腿',None),
              'pelvis':('骨盆',None)}
FOOT_PLACEMENT = {'l':'leftInstep','r':'rightInstep'}
# 0x1111 紧凑广播：data[0]=布局版本，data[1] 高四位产品(1 鞋垫/3 独立IMU)、低四位脚位，data[2:4] 配对号
ADVERTISEMENT_COMPANY_ID = 0x1111
PRODUCT_KINDS = {1:'insole',3:'imu'}
PRODUCT_SIDES = {1:'l',2:'r'}
SCAN_SECONDS = 5
SCAN_CONFIDENCE = {None:0,'name':1,'service':2,'manufacturer':3}


def advertisement_identity(advertisement):
    """按已登记布局读广播身份；未登记布局返回 (None,None,None)，不拿名称当身份。"""
    data = (getattr(advertisement,'manufacturer_data',None) or {}).get(ADVERTISEMENT_COMPANY_ID)
    if not data or len(data) not in (4,8) or data[0] != 1:
        return None,None,None
    if len(data) == 8 and data[:4] != data[4:]:
        return None,None,None
    kind = PRODUCT_KINDS.get(data[1] >> 4)
    foot = PRODUCT_SIDES.get(data[1] & 15) if kind == 'insole' else None
    return (kind,foot,data[2:4].hex()) if kind else (None,None,None)


def scan_identity(name, adv):
    kind,foot,pair = advertisement_identity(adv)
    if kind:
        return 'insole',kind,foot,pair,'manufacturer'
    if WT_SERVICE in (value.lower() for value in adv.service_uuids):
        return 'wt','wt',None,None,'service'
    upper = name.upper()
    if re.match(r'^B?WT\d',upper):
        return 'wt',None,None,None,'name'
    if re.match(r'^(?:D56(?!\d)|W0\d|IMU(?:$|[-_ ]|\d))',upper):
        return 'insole',None,None,None,'name'
    return None,None,None,None,None


class Capture:
    def __init__(self, root, demo=False):
        self.root = Path(root)
        self.demo_allowed = demo
        self.recorder = Recorder(self.root/'records')
        self.algorithm = Algorithm()
        self.algorithm.on_decision = lambda result:self.recorder.put('decision',result)
        self.devices = {}
        self.found = {}
        self.ingress = asyncio.Queue(maxsize=10000)
        self.sequence = 0
        self.cursor = 0
        self.history = collections.deque(maxlen=24000)
        self.logs = collections.deque(maxlen=100)
        self.tasks = []
        self.demo_task = None
        self.replay_task = None
        self.phone_task = None
        self.source = 'direct'
        self.phone = {'enabled':False}
        self.error = None
        self.debug = False
        self.loop = None
        self.running = True
        self.connect_lock = asyncio.Lock()
        self.saving = False
        self.config_path = self.root/'runtime/devices.json'
        try:
            saved = json.loads(self.config_path.read_text(encoding='utf-8'))
        except (OSError,ValueError):
            saved = []
        for d in saved:
            self._device(d['id'],d['name'],d['profile'],d.get('foot'),d.get('address',''),
                         placement=d.get('placement'),label=d.get('placement_label'),kind=d.get('kind'))

    async def start(self):
        self.loop = asyncio.get_running_loop()
        self.tasks.append(asyncio.create_task(self._consume()))

    def log(self, text, level='info'):
        self.logs.append({'text':str(text),'level':level,'time':time.time()*1000})
        if self.debug:
            print('[capture]',level,text,flush=True)

    def _device(self, key, name, profile, foot, address='', source='direct', placement=None, label=None, kind=None):
        placement = placement or FOOT_PLACEMENT.get(foot)
        if not label:
            label = PLACEMENTS[placement][0] if placement in PLACEMENTS else placement
        # 独立 IMU 与鞋垫共用同一套测量协议，区别只在没有压力通道；
        # 已登记的设备沿用原类型，避免重连时丢成默认鞋垫
        kind = kind or (self.devices[key]['kind'] if key in self.devices else None) or ('wt' if profile == 'wt' else 'insole')
        has_pressure = kind == 'insole'
        # 零偏按设备记账：脚背位置用 l/r（步态引擎按脚别取用），其余位置用设备键
        bias_key = foot or 'd:'+key
        if key in self.devices:
            d = self.devices[key]
            if (d['profile'],d['source'],d['placement']) != (profile,source,placement):
                d.update(attitude_filter=None,attitude=None,attitude_receipt=None)
            d.update(name=name,profile=profile,foot=foot,address=address,source=source,placement=placement,placement_label=label,bias_key=bias_key,kind=kind,has_pressure=has_pressure)
            return d
        d = dict(id=key,name=name,profile=profile,foot=foot,address=address,source=source,kind=kind,has_pressure=has_pressure,
                 placement=placement,placement_label=label,bias_key=bias_key,
                 connected=False,reconnecting=False,connection='',client=None,decoder=Decoder(profile),
                 frames=0,bad=0,last=None,samples=collections.deque(maxlen=4000),recent=collections.deque(maxlen=1000),
                 manual=False,topology='v2',format=None,format_status='未回读，单位未知',command_seq=0,
                 command_lock=asyncio.Lock(),control_buffer=b'',ack=None,command_status='',write_char=None,
                 generation=0,command_future=None,
                 attitude_filter=None,attitude=None,attitude_receipt=None,attitude_connection=None)
        self.devices[key] = d
        return d

    def resolve_placement(self, value, label=None):
        """返回 (placement 键, 显示名, foot)；自定义位置按当前设备表分配空号。"""
        value = '' if value is None else str(value)
        value = FOOT_PLACEMENT.get(value) or value
        if value in PLACEMENTS:
            known = PLACEMENTS[value]
            return value, known[0], known[1]
        text = str(label or '').strip()
        if value == 'custom' or re.fullmatch(r'custom\d{1,2}', value):
            if not text:
                raise ValueError('自定义位置需要填写名称')
            if len(text) > 20:
                raise ValueError('位置名称最多 20 个字符')
            existing = next((d for d in self.devices.values() if d['placement_label'] == text), None)
            if existing is not None:
                if existing['connected']:
                    raise ValueError('该位置名称已被占用')
                # 换个设备接同一位置：沿用原槽位，旧设备条目保留为已断开
                return existing['placement'], text, None
            if value == 'custom':
                used = {d['placement'] for d in self.devices.values()}
                index = 1
                while 'custom%d' % index in used:
                    index += 1
                value = 'custom%d' % index
            return value, text, None
        raise ValueError('请选择设备位置')

    def persist_devices(self):
        self.config_path.parent.mkdir(exist_ok=True)
        atomic_json(self.config_path,[{k:d[k] for k in ('id','name','profile','foot','address','placement','placement_label','kind')} for d in self.devices.values() if d['source'] == 'direct'])

    async def scan(self):
        if self.source != 'direct':
            raise ValueError('先切换到电脑直连来源')
        radio = {}

        def detected(device, adv):
            address = device.address
            previous = radio.get(address)
            name = (adv.local_name or '').strip() or (previous[1] if previous else '')
            identity = scan_identity(name or (device.name or '').strip(),adv)
            samples = previous[3] if previous else collections.deque(maxlen=7)
            samples.append(adv.rssi)
            # 后到的空广播不能抹掉同一次扫描已观察到的厂商/服务标识。
            if previous and SCAN_CONFIDENCE[previous[2][4]] > SCAN_CONFIDENCE[identity[4]] and previous[2][4] != 'name':
                identity = previous[2]
            radio[address] = (device,name,identity,samples)

        async with self.connect_lock:
            try:
                async with BleakScanner(detection_callback=detected):
                    await asyncio.sleep(SCAN_SECONDS)
            except (BleakError,OSError) as exc:
                self.log('蓝牙扫描失败：'+str(exc),'warning')
                raise ValueError('无法扫描蓝牙，请确认电脑蓝牙已开启、适配器可用，再重试') from exc
            registered = {d['address']:d for d in self.devices.values() if d['source'] == 'direct'}
            found = {}
            for address,(device,name,identity,samples) in radio.items():
                saved = registered.get(address)
                profile,kind,foot,pair,evidence = identity
                if saved and evidence not in ('manufacturer','service'):
                    profile,kind,evidence = saved['profile'],saved['kind'],'saved'
                name = name or (device.name or '').strip() or (saved['name'] if saved else '') or '未广播名称'
                row = dict(address=address,name=name,rssi=round(statistics.median(samples)),profile=profile,
                           kind=kind,foot=foot,pair_id=pair,evidence=evidence,connected=bool(saved and saved['connected']))
                found[address] = (device,row)
            self.found = found
        return sorted((row for _,row in found.values()),key=lambda row:(-row['rssi'],row['address']))

    async def connect(self, address, profile, placement=None, key=None, label=None, kind=None):
        if self.source != 'direct':
            raise ValueError('非电脑直连模式不能连接电脑蓝牙')
        if profile not in ('wt','insole'):
            raise ValueError('invalid profile')
        if kind not in (None,'insole','imu','wt'):
            raise ValueError('invalid kind')
        async with self.connect_lock:
            device,scan = self.found.get(address,(None,{}))
            if scan.get('evidence') in ('manufacturer','service'):
                kind = scan['kind']
            kind = kind or ('wt' if profile == 'wt' else 'insole')
            profile = 'wt' if kind == 'wt' else 'insole'
            placement, text, foot = self.resolve_placement(placement or scan.get('foot'),label)
            if kind == 'imu':
                foot = None
            for other in self.devices.values():
                if other['connected'] and other['source'] == 'direct' and other['address'] == address:
                    if other['profile'] != profile or other['placement'] != placement:
                        raise ValueError('该设备已连接；更换协议或位置前先停止记录并断开')
                    return self.public(other)
                if other['connected'] and other['placement'] == placement and other['address'] != address:
                    raise ValueError('该位置已有设备连接')
            if not device:
                device = await BleakScanner.find_device_by_address(address,timeout=8)
            if not device:
                raise ValueError('未扫描到设备，请确认上电且未被其他程序占用')
            d = None
            generation = None
            def gone(client):
                if d is not None and d['connected'] and generation == d['generation']:
                    self.loop.call_soon_threadsafe(lambda:asyncio.create_task(self._lost(d,generation)))
            client = BleakClient(device,disconnected_callback=gone,timeout=20)
            try:
                await client.connect()
                notify = client.services.get_characteristic(INSOLE_NOTIFY if profile == 'insole' else WT_NOTIFY)
                char = client.services.get_characteristic(INSOLE_WRITE if profile == 'insole' else WT_WRITE)
                if char and not set(char.properties).intersection(('write','write-without-response')):
                    char = None
                # 标准心率设备也有 2A37，不能单凭它确认自研协议。
                if not notify or not set(notify.properties).intersection(('notify','indicate')) or (profile == 'insole' and not char):
                    protocol_name = '自研鞋垫 / 独立 IMU' if profile == 'insole' else '维特 BLE'
                    raise ValueError('设备未提供 '+protocol_name+' 所需的蓝牙通道，请检查设备和所选协议')
                name = scan.get('name')
                if not name or name == '未广播名称':
                    name = device.name or address
                d = self._device(key or profile+':'+address,name,profile,foot,address,placement=placement,label=text,kind=kind)
                d['manual'] = False
                d['generation'] += 1
                generation = d['generation']
                d['connection'] = uuid.uuid4().hex
                d['client'] = client
                d['decoder'] = Decoder(profile)
                d['control_buffer'] = b''
                d['format'] = None
                d['format_status'] = '未回读，单位未知'
                d['write_char'] = char
                d['connected'] = True; d['reconnecting'] = False
                self.emit(d,'connect',dict(name=d['name'],address=address))
                def receive(sender, data):
                    receipt = time.monotonic()*1000
                    epoch = time.time()*1000
                    if generation == d['generation']:
                        self.loop.call_soon_threadsafe(self.submit,d['id'],bytes(data),receipt,epoch,None,None,False,generation)
                await client.start_notify(notify,receive)
                if profile == 'insole' and char and any(x in char.properties for x in ('notify','indicate')):
                    def reply(sender,data):
                        if generation == d['generation']:
                            self.loop.call_soon_threadsafe(self.submit,d['id'],bytes(data),time.monotonic()*1000,time.time()*1000,None,None,True,generation)
                    await client.start_notify(char,reply)
                self.persist_devices()
                self.log('已连接 '+d['name'])
            except Exception:
                if d is not None:
                    d['connected'] = False
                    d['client'] = d['write_char'] = None
                if client.is_connected:
                    await client.disconnect()
                raise
            return self.public(d)

    async def _lost(self, d, generation):
        if generation != d['generation']:
            return
        d['connected'] = False
        d.update(attitude_filter=None,attitude=None,attitude_receipt=None)
        self.emit(d,'disconnect',{})
        self.algorithm.offer(dict(control='boundary',foot=d['foot'],bias_key=d['bias_key'],clock_ms=time.monotonic()*1000,reason='device_boundary'))
        d['decoder'] = Decoder(d['profile'])
        self.log('设备断开：'+d['name'],'warning')
        if d['manual'] or not self.running or d['source'] != 'direct':
            return
        d['reconnecting'] = True
        for delay in (1,2,4,8,15):
            await asyncio.sleep(delay)
            if d['manual'] or not self.running or d['generation'] != generation or self.source != 'direct':
                break
            try:
                await self.connect(d['address'],d['profile'],d['placement'],d['id'],d['placement_label'],kind=d['kind'])
                return
            except Exception as exc:
                self.log('重连未成功：'+str(exc),'warning')
                generation = d['generation']
        d['reconnecting'] = False

    async def disconnect(self, key):
        d = self.devices[key]
        d['manual'] = True
        if d['client'] and d['client'].is_connected:
            await d['client'].disconnect()
        elif d['connected']:
            d['connected'] = False
            self.emit(d,'disconnect',{})
        return self.public(d)

    def forget(self, key):
        d = self.devices.get(key)
        if not d:
            raise ValueError('设备不存在')
        if d['source'] != 'direct':
            raise ValueError('只能移除电脑直连设备')
        if d['connected'] or d['reconnecting']:
            raise ValueError('先断开该设备再移除')
        d['manual'] = True
        d['generation'] += 1
        self.found.pop(d['address'],None)
        self.devices.pop(key)
        self.persist_devices()
        self.log('已移除设备：'+d['name'])
        return self.snapshot()

    def emit(self,d,kind,data,receipt=None,epoch=None,source_receipt=None,source_domain=None):
        self.sequence += 1
        receipt = time.monotonic()*1000 if receipt is None else receipt
        event = dict(v=2,id=self.sequence,type=kind,device_id=d['id'],profile=d['profile'],device_kind=d['kind'],foot=d['foot'],
                     placement=d['placement'],placement_label=d['placement_label'],
                     connection=d['connection'],source=d['source'],receipt_monotonic_ms=receipt,
                     receipt_epoch_ms=time.time()*1000 if epoch is None else epoch,
                     source_receipt_ms=receipt if source_receipt is None else source_receipt,
                     source_time_domain=source_domain or 'desktop_monotonic',data=data)
        self.recorder.put('raw',event)
        return event

    def submit(self,key,data,receipt,epoch,source_receipt=None,source_domain=None,control=False,generation=None):
        if generation is not None and generation != self.devices[key]['generation']:
            return
        try:
            self.ingress.put_nowait((key,data,receipt,epoch,source_receipt,source_domain,control,generation))
        except asyncio.QueueFull:
            self.error = '蓝牙接收队列已满，当前记录完整性不可保证'
            self.log(self.error,'error')
            self.algorithm.offer(dict(control='boundary',foot=self.devices[key]['foot'],bias_key=self.devices[key]['bias_key'],clock_ms=receipt,reason='ingress_overflow'))
            self.recorder.error = self.error
            self.recorder.accepting = False

    async def _consume(self):
        while self.running:
            item = await self.ingress.get()
            try:
                await self._process(*item)
            except Exception as exc:
                self.error = str(exc)
                self.log('解析失败：'+str(exc),'error')
            finally:
                self.ingress.task_done()

    async def _process(self,key,data,receipt,epoch,source_receipt,source_domain,control,generation=None):
        d = self.devices[key]
        if generation is not None and generation != d['generation']:
            return
        e = self.emit(d,'notify',dict(bytes=base64.b64encode(data).decode(),channel='control' if control else 'measurement'),receipt,epoch,source_receipt,source_domain)
        if d['profile'] == 'insole':
            self._control_reply(d,data)
        if control:
            return
        frames = d['decoder'].feed(data)
        d['bad'] = d['decoder'].bad
        connection = d['connection']
        prepared = None
        if frames and d['profile'] == 'insole':
            try:
                prepared = await asyncio.wait_for(asyncio.wrap_future(self.algorithm.control('prepare_frames',
                    bias_key=d['bias_key'],foot=d['foot'],connection=connection,frames=frames,
                    clock_ms=e['source_receipt_ms'],source_receipt_ms=e['source_receipt_ms'])), 2.)
            except Exception as exc:
                prepared = {'states':[], 'events':[], 'error':'bias_unavailable'}
                self.log('在线零偏不可用：'+str(exc),'warning')
            if connection != d['connection'] or not d['connected']:
                return
            if generation is not None and generation != d['generation']:
                return
        algo_frames = []
        for index, frame in enumerate(frames):
            # 每帧独立判定：上一帧的偏置状态不能顺延到本帧
            bias_state = None
            d['frames'] += 1; d['recent'].append(receipt)
            frame = dict(frame,device_id=key,foot=d['foot'],bias_key=d['bias_key'],profile=d['profile'],connection=d['connection'],
                         placement=d['placement'],placement_label=d['placement_label'],
                         source=d['source'],receipt_ms=receipt,epoch_ms=epoch)
            if prepared and len(prepared['states']) == len(frames):
                bias_state = prepared['states'][index]
                if bias_state and bias_state.get('ready'):
                    frame['bias_used'] = list(bias_state['gyro_bias_dps'])
            self._update_attitude(d, frame)
            self.cursor += 1
            frame['cursor'] = self.cursor
            d['last'] = frame
            if frame.get('values') is not None:
                d['samples'].append(frame)
            if frame.get('values') is not None or 'attitude' in frame:
                self.history.append(frame)
            if d['has_pressure']:
                mapping = slot_map(d['foot'],d['topology'])
                frame['pressure'] = [frame['pressure_sparse'][slot] for slot in mapping]
            self.recorder.put('frame',dict(schema_version='capture_frames.v3',session_id=self.recorder.session_id,
                device_id=key,profile=d['profile'],device_kind=d['kind'],foot=d['foot'],placement=d['placement'],placement_label=d['placement_label'],
                connection=d['connection'],source=d['source'],
                event_id=e['id'],frame_index=index,receipt_monotonic_ms=receipt,receipt_epoch_ms=epoch,
                source_receipt_ms=e['source_receipt_ms'],source_time_domain=e['source_time_domain'],
                device_calendar_ms=frame.get('device_ms'),source_frame_number_raw=frame.get('seq'),
                imu_raw_values=frame.get('raw6',[]),imu_physical_values=frame.get('values') or [],
                pressure_raw_values=frame.get('pressure_slots',[]),pressure_sparse_values=frame.get('pressure_sparse',[]),
                pressure_observed_slots=frame.get('pressure_observed_slots',[]),pressure_output_format=d['format'],
                sensor_topology=d['topology'],gyro_temperature_celsius=frame.get('temperature'),
                imu_bias_dps=None if not frame.get('bias_used') else list(frame['bias_used']),
                imu_bias_ready=bool(frame.get('bias_used')),
                imu_bias_revision=int((bias_state or {}).get('generation') or 0),
                imu_bias_gate=(bias_state or {}).get('gate_version'),
                frame_kind=frame['kind'],raw_packet_sha256=frame['raw_packet_sha256'],raw_frame_hex=frame['raw_frame_hex']))
            if d['has_pressure'] or frame.get('inference_eligible'):
                algo_frames.append(frame)
        if algo_frames:
            self.algorithm.offer(dict(profile=d['profile'],foot=d['foot'],bias_key=d['bias_key'],connection=d['connection'],frames=algo_frames,
                clock_ms=e['source_receipt_ms'],source_receipt_ms=e['source_receipt_ms'],source_time_domain=e['source_time_domain'],prepared=prepared))

    def _control_reply(self,d,data):
        b = (d['control_buffer']+bytes(data))[-4096:]
        i = 0
        while i+2 < len(b):
            if b[i] != 0x5a:
                i += 1
                continue
            kind = b[i+2]
            if b[i+1:i+3] == b'\x00\x09':
                if len(b)-i < 30:
                    break
                size = 46+2*b[i+29]
                if size > 186:
                    i += 1
                    continue
                if len(b)-i < size:
                    break
                i += size if crc_measurement(b[i:i+size]) else 1
                continue
            future = d['command_future']
            waiting = future is not None and not future.done() and kind == d.get('command_type')
            if kind not in (5,13,16) or (b[i+1] != 0 and not waiting):
                i += 1
                continue
            if len(b)-i < 19:
                break
            size = int.from_bytes(b[i+15:i+17],'little')
            if size not in {5:(0,),13:(42,78),16:(1,)}[kind]:
                i += 1
                continue
            legacy = d.get('command_link') == 'v1' and kind in (5,13)
            length = 83 if legacy else 19+size
            if len(b)-i < length:
                break
            frame = b[i:i+length]
            if legacy and crc_ccitt(frame[:81]) != int.from_bytes(frame[81:83],'little'):
                i += 1
                continue
            payload = frame[17:17+size]
            d['ack'] = {'type':kind,'status':payload[0] if payload else 0,'size':size}
            if payload and payload[0] == 0 and kind == 13:
                fmt = payload[3]
                if 0 <= fmt <= 4:
                    d['format'] = fmt; d['format_status'] = '设备回读确认'
            if waiting:
                future.set_result(d['ack'])
            i += length
        d['control_buffer'] = b[i:]

    async def device_command(self,key,action,value=None,link='v2'):
        d = self.devices[key]
        if d['source'] != 'direct' or not d['connected'] or not d['write_char']:
            raise ValueError('只有电脑直连且具备写入通道的设备可发命令')
        async with d['command_lock']:
            if d['profile'] == 'wt':
                regs = {'info':0x66,'battery':0x64,'version':0x2e}
                if action not in regs:
                    raise ValueError('unsupported WT command')
                packet = bytes([0xff,0xaa,0x27,regs[action],0])
            else:
                if action not in ('read_state','set_format'):
                    raise ValueError('unsupported command')
                if action == 'set_format' and (not isinstance(value,int) or not 0 <= value <= 4):
                    raise ValueError('invalid format')
                kind = 13 if action == 'read_state' else 16
                payload = b'' if action == 'read_state' else bytes([value])
                d['command_seq'] += 1
                packet = command_packet(kind,payload,d['command_seq'],time.time()*1000,link)
                d['command_type'] = kind
                d['command_link'] = link
                d['control_buffer'] = b''
                if action == 'set_format':
                    d['format'] = None
                    d['format_status'] = '格式写入待确认，单位未知'
            char = d['write_char']
            response = 'write-without-response' not in char.properties
            if not response and len(packet) > char.max_write_without_response_size:
                raise ValueError('当前链路单包容量不足，未发出命令；不会自行拆包或改发其他协议')
            d['command_future'] = asyncio.get_running_loop().create_future()
            try:
                await d['client'].write_gatt_char(char,packet,response=response)
                self.emit(d,'command',{'action':action,'value':value,'bytes':packet.hex(),'link':link})
                d['command_status'] = '已写出，等待设备回执'
                if d['profile'] == 'insole':
                    try:
                        ack = await asyncio.wait_for(d['command_future'],1.5)
                        d['command_status'] = '设备回执状态 '+str(ack['status'])
                        if action == 'set_format' and ack['status'] == 0:
                            d['format'] = value
                            d['format_status'] = '设备设置回执确认'
                    except asyncio.TimeoutError:
                        d['command_status'] = '已写出但无回执，不能确认设备已应用'
                else:
                    d['command_status'] = '已写出信息查询，原始回包已记录'
                return {'status':d['command_status']}
            finally:
                d['command_future'] = None

    async def source_change(self,source):
        if source not in ('direct','phone','demo','replay'):
            raise ValueError('unknown source')
        if self.recorder.accepting:
            raise ValueError('先停止记录再切换来源')
        if any(d['connected'] and d['source'] == 'direct' for d in self.devices.values()):
            raise ValueError('先断开电脑直连设备再切换来源')
        await self.stop_virtual()
        self.source = source
        await asyncio.wrap_future(self.algorithm.control('configure',enabled=False))
        return self.snapshot()

    async def stop_virtual(self):
        for name in ('demo_task','replay_task','phone_task'):
            task = getattr(self,name)
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                setattr(self,name,None)
        for d in self.devices.values():
            if d['source'] != 'direct':
                d['connected'] = False

    async def demo(self,profile):
        if not self.demo_allowed:
            raise ValueError('模拟采集仅在显式测试模式可用')
        if profile not in ('wt','insole'):
            raise ValueError('invalid profile')
        await self.source_change('demo')
        for foot in ('l','r'):
            d = self._device('demo:'+foot,'模拟'+('左脚' if foot == 'l' else '右脚'),profile,foot,source='demo')
            d['connected'] = True; d['connection'] = uuid.uuid4().hex; d['decoder'] = Decoder(profile)
            self.emit(d,'connect',{'name':d['name']})
        async def produce():
            seq = 0
            start = self.loop.time()
            receipt_origin = time.monotonic()*1000
            epoch_origin = time.time()*1000
            while True:
                seq += 1
                # 模拟时轴必须独立于 Windows 补发调度；真实 BLE 仍使用实际回调时刻。
                receipt = receipt_origin+seq*20
                epoch = epoch_origin+seq*20
                for foot in ('l','r'):
                    self.submit('demo:'+foot,demo_packet(profile,seq,foot,epoch),receipt,epoch)
                await asyncio.sleep(max(0,start+seq*.02-self.loop.time()))
        self.demo_task = asyncio.create_task(produce())
        self.log('模拟数据运行中，不代表真实设备性能','warning')
        return self.snapshot()

    async def record_start(self):
        await self.ingress.join()
        if not any(d['connected'] for d in self.devices.values()) and self.source not in ('direct','phone'):
            raise ValueError('无采集来源')
        session = self.recorder.start({'source':self.source,'devices':[self.public(d) for d in self.devices.values() if d['connected']]})
        for d in self.devices.values():
            if d['connected']:
                self.emit(d,'connect',{'name':d['name'],'address':d['address'],'record_boundary':True})
        return {'session_id':session,**self.recorder.state()}

    async def record_stop(self):
        await self.ingress.join()
        return await asyncio.to_thread(self.recorder.stop)

    def _boundary(self,reason):
        # 暂停/恢复都切断零偏与姿态的连续积分，不能让跨缺口的帧被当成相邻帧
        for d in self.devices.values():
            if d['connected']:
                self.algorithm.offer(dict(control='boundary',foot=d['foot'],bias_key=d['bias_key'],
                                          clock_ms=time.monotonic()*1000,reason=reason))

    async def record_pause(self):
        await self.ingress.join()
        result = await asyncio.to_thread(self.recorder.pause)
        self._boundary('record_pause')
        return result

    async def record_resume(self):
        result = await asyncio.to_thread(self.recorder.resume)
        self._boundary('record_resume')
        return result

    async def replay(self,session,speed=1.):
        if not .1 <= speed <= 10:
            raise ValueError('回放速度必须0.1到10倍')
        path = self.recorder.directory(session)/'raw_events.jsonl'
        await self.source_change('replay')
        replay_id = uuid.uuid4().hex
        async def play():
            previous = None
            start = time.monotonic()*1000
            origin = None
            with path.open(encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    if event['type'] not in ('connect','disconnect','notify'):
                        continue
                    stamp = event['receipt_monotonic_ms']
                    if origin is None:
                        origin = stamp
                    if previous is not None:
                        await asyncio.sleep(max(0,(stamp-previous)/1000/speed))
                    previous = stamp
                    key = 'replay:'+event['device_id']
                    d = self._device(key,event['device_id'],event['profile'],event['foot'],source='replay',
                                     placement=event.get('placement'),label=event.get('placement_label'),kind=event.get('device_kind'))
                    d['connection'] = f'replay:{replay_id}:{event["connection"]}'
                    d['connected'] = event['type'] != 'disconnect'
                    if event['type'] == 'connect':
                        d['decoder'] = Decoder(d['profile'])
                    if event['type'] == 'notify':
                        receipt = time.monotonic()*1000
                        self.submit(key,base64.b64decode(event['data']['bytes']),receipt,time.time()*1000,
                                    event.get('source_receipt_ms',stamp),'replay_original_monotonic',event['data'].get('channel') == 'control')
            await self.ingress.join()
            for d in self.devices.values():
                if d['source'] == 'replay':
                    d['connected'] = False
            self.log('回放结束')
        async def guarded_play():
            try:
                await play()
            except Exception as exc:
                self.error = '回放失败：'+str(exc)
                self.log(self.error,'error')
        self.replay_task = asyncio.create_task(guarded_play())
        return {'ok':True}

    @staticmethod
    def _update_attitude(d, frame):
        if d['profile'] == 'insole':
            fusion = d['attitude_filter']
            if fusion is None or fusion.scope != d['connection']:
                fusion = d['attitude_filter'] = LabAttitude(d['connection'])
            stamp = frame.get('device_ms')
            result = fusion.add(frame['raw6'], None if stamp is None else round(stamp * 1000))
            attitude = dict(degrees=result[:3],reference_id=result[3]) if result else None
        elif frame['kind'] in (0x53, 0x61):
            attitude = dict(degrees=frame['angles'],reference_id=d['connection'])
        else:
            return
        frame['attitude'] = d['attitude'] = attitude
        d['attitude_receipt'] = frame['receipt_ms']
        d['attitude_connection'] = d['connection']

    def public(self,d):
        now = time.monotonic()*1000
        attitude = d['attitude'] if (d['connected'] and d['attitude_receipt'] is not None
                                      and d['attitude_connection'] == d['connection']
                                      and now - d['attitude_receipt'] <= 500) else None
        recent = d['recent']
        while recent and recent[0] < now-1000:
            recent.popleft()
        return {k:d[k] for k in ('id','name','profile','kind','has_pressure','foot','bias_key','address','placement','placement_label','source','connected','reconnecting','connection','frames','bad','last','topology','format','format_status','command_status')} | {'rate':len(recent),'attitude':attitude}

    def snapshot(self):
        return dict(version=VERSION,source=self.source,devices=[self.public(d) for d in self.devices.values()],
                    recorder=self.recorder.state(),algorithm=self.algorithm.snapshot(),phone=self.phone,
                    error=self.error,logs=list(self.logs),cursor=self.cursor,demo_allowed=self.demo_allowed,
                    warning='设备日历、手机接收与电脑接收属于不同时间域；未证明硬件同步或真实单程延迟')

    def stream(self,after):
        first = self.history[0]['cursor'] if self.history else self.cursor+1
        gap = after > 0 and after < first-1
        return dict(type='update',state=self.snapshot(),gap=gap,
                    frames=[f for f in self.history if f['cursor'] > after][-4000:],cursor=self.cursor)

    async def close(self):
        await self.stop_virtual()
        for d in self.devices.values():
            d['manual'] = True
            if d['client'] and d['client'].is_connected:
                await d['client'].disconnect()
        await self.ingress.join()
        self.running = False
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks,return_exceptions=True)
        await asyncio.to_thread(self.recorder.stop)
        await asyncio.to_thread(self.algorithm.close)
