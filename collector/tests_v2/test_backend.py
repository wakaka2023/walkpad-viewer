import asyncio
import io
import base64
import json
import struct
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend import capture as capture_module
from backend.capture import Capture
from backend.exports import export_package, export_standard, export_txt
from backend.phone import Phone, validate
from backend.protocol import Decoder, RAW_V1, RAW_V2, command_packet, demo_packet, slot_map
from backend.storage import Recorder


def test_ble_scan_classifies_evidence_and_replaces_results(tmp_path,monkeypatch):
    c = Capture(tmp_path)
    c._device('old','Old WT','wt',None,'saved',placement='rightWrist')

    def advertisement(address,name='',rssi=-65,data=b'',uuids=(),cached=None):
        return SimpleNamespace(address=address,name=cached),SimpleNamespace(local_name=name,rssi=rssi,service_uuids=uuids,manufacturer_data={0x1111:data})

    packets = [advertisement('unknown','ESLinker',-41),advertisement('unnamed',rssi=-49),
               advertisement('insole',data=b'\x01\x12\x34\x56',rssi=-60),
               advertisement('insole','D56-01',-62,cached='WT old name'),advertisement('insole',rssi=-31),
               advertisement('imu','IMU',data=b'\x01\x32\x12\x34'),
               advertisement('wt_service',uuids=(capture_module.WT_SERVICE.upper(),)),
               advertisement('wt_name','WT901BLE68',cached='Stale name'),
               advertisement('old_insole','W01234'),advertisement('saved'),
               advertisement('not_wt','WTHome'),advertisement('not_imu','iMusic'),
               advertisement('invalid','',data=b'\x01\x91\x00\x00')]

    class Scanner:
        def __init__(self,detection_callback): self.detected = detection_callback
        async def __aenter__(self):
            for packet in packets: self.detected(*packet)
            return self
        async def __aexit__(self,*exc): pass

    monkeypatch.setattr(capture_module,'BleakScanner',Scanner)
    monkeypatch.setattr(capture_module,'SCAN_SECONDS',0)
    rows = asyncio.run(c.scan())
    result = {item['address']:item for item in rows}
    assert len(result) == len(rows) == 11
    assert rows[0]['address'] == 'unknown'
    assert result['unknown']['profile'] is result['unknown']['kind'] is None
    assert result['unnamed']['name'] == '未广播名称'
    assert result['insole'] == dict(address='insole',name='D56-01',rssi=-60,profile='insole',kind='insole',foot='r',pair_id='3456',evidence='manufacturer',connected=False)
    assert result['imu']['kind'] == 'imu' and result['imu']['foot'] is None
    assert result['wt_service']['kind'] == 'wt' and result['wt_service']['evidence'] == 'service'
    assert result['wt_name']['name'] == 'WT901BLE68' and result['wt_name']['kind'] is None and result['wt_name']['profile'] == 'wt'
    assert result['old_insole']['profile'] == 'insole' and result['old_insole']['kind'] is None
    assert result['saved']['kind'] == 'wt' and result['saved']['evidence'] == 'saved'
    assert all(result[key]['profile'] is None for key in ('not_wt','not_imu','invalid'))
    packets.clear()
    assert asyncio.run(c.scan()) == [] and not c.found


def test_ble_connect_checks_gatt_before_registering(tmp_path,monkeypatch):
    seen = []
    characteristics = {capture_module.INSOLE_NOTIFY:SimpleNamespace(properties=['notify'])}

    class Client:
        def __init__(self,device,disconnected_callback,timeout):
            self.callback = disconnected_callback
            self.is_connected = False
            self.services = SimpleNamespace(get_characteristic=characteristics.get)
        async def connect(self): self.is_connected = True
        async def disconnect(self):
            self.is_connected = False
            self.callback(self)
        async def start_notify(self,char,callback): seen.append(char)

    monkeypatch.setattr(capture_module,'BleakClient',Client)

    async def run():
        c = Capture(tmp_path)
        raw_device = SimpleNamespace(name='Raw name')
        c.found['device'] = (raw_device,dict(name='Chosen device',kind=None,evidence=None))
        with pytest.raises(ValueError,match='所需的蓝牙通道'):
            await c.connect('device','insole','leftAnkle',kind='imu')
        assert not c.devices and not seen
        characteristics[capture_module.INSOLE_WRITE] = SimpleNamespace(properties=['write-without-response'])
        d = await c.connect('device','insole','leftAnkle',kind='imu')
        assert d['connected'] and d['kind'] == 'imu' and d['name'] == 'Chosen device'
        assert seen == [characteristics[capture_module.INSOLE_NOTIFY]]
        saved = Capture(tmp_path).devices[d['id']]
        assert saved['placement'] == 'leftAnkle' and saved['kind'] == 'imu'

    asyncio.run(run())


def test_live_lab_attitude_uses_device_time_and_resets_reference(tmp_path):
    async def run():
        c = Capture(tmp_path)
        await c.start()
        try:
            d = c._device('ankle','Ankle','insole',None,placement='leftAnkle',kind='imu')
            d.update(connected=True,connection='first')
            epoch = time.time()*1000
            receipt = time.monotonic()*1000
            for i in range(101):
                await c._process('ankle',demo_packet('insole',i,'l',epoch+i*20),receipt,epoch,None,None,False)
            frames = c.stream(0)['frames']
            assert frames[99]['attitude']['degrees'][2] is None
            first = frames[100]['attitude']
            assert first['degrees'] == pytest.approx((0,0,0))
            assert 'velocity' not in frames[-1]
            packet = demo_packet('insole',101,'l',epoch+2270)
            await c._process('ankle',packet,time.monotonic()*1000,epoch,None,None,False)
            after_gap = c.public(d)['attitude']
            assert after_gap['degrees'][2] is None and after_gap['reference_id'] != first['reference_id']
            d['connection'] = 'second'
            assert c.public(d)['attitude'] is None
            await c._process('ankle',packet,time.monotonic()*1000,epoch,None,None,False)
            assert c.public(d)['attitude']['reference_id'] == 'second:0'
            d['connected'] = False
            assert c.public(d)['attitude'] is None
        finally:
            await c.close()
    asyncio.run(run())


def test_live_wt_streams_only_fresh_angle_frames(tmp_path):
    async def run():
        c = Capture(tmp_path)
        d = c._device('wt','WT','wt',None,placement='leftWrist')
        d.update(connected=True,connection='wt1')
        packed = b'\x55\x53' + struct.pack('<4h',4096,-8192,16384,0)
        angle = packed + bytes([sum(packed) % 256])
        receipt, epoch = time.monotonic()*1000, time.time()*1000
        await c._process('wt',angle,receipt,epoch,None,None,False)
        expected = dict(degrees=[22.5,-45.,90.],reference_id='wt1')
        assert c.public(d)['attitude'] == expected
        assert c.stream(0)['frames'][0]['attitude'] == expected
        packet = b'\x55\x59' + bytes(18)
        await c._process('wt',packet,receipt,epoch,None,None,False)
        assert 'attitude' not in d['last'] and c.public(d)['attitude'] == expected
        d['attitude_receipt'] -= 501
        assert c.public(d)['attitude'] is None
        await c._process('wt',demo_packet('wt',0,'l',epoch),time.monotonic()*1000,epoch,None,None,False)
        assert c.public(d)['attitude']['degrees'] == [0.,0.,0.]
    asyncio.run(run())


@pytest.mark.parametrize('profile',['insole','wt'])
def test_split_and_combined(profile):
    data = demo_packet(profile,2,'l',time.time()*1000)
    for cut in range(1,len(data)):
        decoder = Decoder(profile)
        assert decoder.feed(data[:cut]) == []
        assert len(decoder.feed(data[cut:])) == 1
    assert len(Decoder(profile).feed(data+data)) == 2


def test_insole_crc_bitmap_and_raw():
    packet = demo_packet('insole',3,'l',time.time()*1000)
    frame = Decoder('insole').feed(packet)[0]
    assert frame['values'][2] == 1
    assert len(frame['pressure_slots']) == len(frame['pressure_sparse']) == 70
    assert len(frame['pressure_observed_slots']) == 2
    assert frame['seq'] == 3
    bad = bytearray(packet); bad[23] ^= 1
    decoder = Decoder('insole')
    assert decoder.feed(bad) == [] and decoder.bad > 0
    assert frame['device_ms'] is not None


@pytest.mark.parametrize('foot',['l','r'])
@pytest.mark.parametrize('topology',['v1','v2'])
def test_mapping(foot,topology):
    mapping = slot_map(foot,topology)
    assert len(mapping) == len(set(mapping)) == 56
    assert min(mapping) >= 0 and max(mapping) < 70
    assert slot_map('l','v2') == RAW_V2
    assert slot_map('r','v1') == RAW_V1


def test_commands():
    packet = command_packet(16,b'\4',5,123,'v2')
    assert len(packet) == 144 and packet[:2] == b'\x5a\x10'
    assert packet[14] == 4 and struct.unpack_from('<I',packet,10)[0] == 1
    assert len(command_packet(13,link='v1')) == 80


def test_record_and_export(tmp_path):
    async def run():
        c = Capture(tmp_path,demo=True)
        await c.start()
        await c.demo('insole')
        await c.record_start()
        await asyncio.sleep(1.4)
        state = await c.record_stop()
        assert not state['error'],state
        assert state['frames'] >= 100
        assert state['queued'] == 0 and state['chunks'] > 0
        path = c.recorder.current
        chunks = sorted(path.glob('frames_*.parquet'))
        assert sum(pq.ParquetFile(p).metadata.num_rows for p in chunks) == state['frames']
        standard = pq.read_table(export_standard(path))
        assert standard.num_rows == state['frames']
        expected = [name for name,kind in json.loads((Path(__file__).parents[1]/'backend/standard_schema.json').read_text())]
        assert standard.column_names == expected
        assert standard['effective_elapsed_us'].null_count == standard.num_rows
        elapsed = standard['observed_session_elapsed_us'].to_pylist()
        assert min(elapsed) >= 0 and max(elapsed) < 10_000_000
        assert len(standard['pressure_values'][0].as_py()) == 56
        assert len(standard['pressure_raw_values'][0].as_py()) == 70
        archive = c.recorder.export(state['session_id'])
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            assert 'raw_events.jsonl' in z.namelist()
        await c.close()
    asyncio.run(run())


def test_wt_export(tmp_path):
    async def run():
        c = Capture(tmp_path,demo=True)
        await c.start(); await c.demo('wt'); await c.record_start()
        await asyncio.sleep(.3)
        state = await c.record_stop()
        assert not state['error']
        lines = export_txt(c.recorder.current).read_text(encoding='utf-8-sig').splitlines()
        assert len(lines) > 4
        assert all(len(line.split('\t')) == 23 for line in lines)
        await c.close()
    asyncio.run(run())


def test_recovery_and_traversal(tmp_path):
    path = tmp_path/'interrupted'; path.mkdir()
    (path/'manifest.json').write_text(json.dumps({'session_id':'interrupted','status':'recording'}))
    (path/'raw_events.jsonl').write_text('{}\n')
    recorder = Recorder(tmp_path)
    assert recorder.recovered == ['interrupted']
    assert recorder.sessions()[0]['interrupted']
    for bad in ('../x','..','/tmp','a\\b'):
        with pytest.raises(ValueError):
            recorder.directory(bad)


def test_video_idempotency(tmp_path):
    recorder = Recorder(tmp_path)
    sid = recorder.start({'source':'test'})
    video_id = 'a'*32
    assert recorder.append_video(sid,video_id,0,'video/webm',b'abc')['next'] == 1
    assert recorder.append_video(sid,video_id,0,'video/webm',b'abc')['next'] == 1
    assert recorder.append_video(sid,video_id,1,'video/webm',b'def')['next'] == 2
    recorder.stop()
    assert (recorder.current/('video_'+video_id+'.webm')).read_bytes() == b'abcdef'
    with pytest.raises(ValueError):
        recorder.append_video(sid,video_id,4,'video/webm',b'bad')


def test_video_container_follows_mime(tmp_path):
    recorder = Recorder(tmp_path)
    sid = recorder.start({'source':'test'})
    mp4_id = 'b'*32
    assert recorder.append_video(sid,mp4_id,0,'video/mp4;codecs=avc1',b'abc')['next'] == 1
    assert recorder.append_video(sid,mp4_id,1,'video/mp4;codecs=avc1',b'def')['next'] == 2
    recorder.stop()
    assert (recorder.current/('video_'+mp4_id+'.mp4')).read_bytes() == b'abcdef'
    anchor = json.loads((recorder.current/'manifest.json').read_text(encoding='utf-8'))['videoAnchors'][0]
    assert anchor['file'] == 'video_'+mp4_id+'.mp4' and anchor['chunks'] == 2
    assert anchor['firstMonotonicMs'] <= anchor['lastMonotonicMs']


def test_api_tokens_and_websocket_refresh(tmp_path):
    with TestClient(create_app(tmp_path,demo=True)) as client:
        cfg = client.get('/api/config').json()
        token = cfg['control_token']
        headers = {'X-Walkpad-Token':token}
        assert client.post('/api/control',json={'action':'demo'}).status_code == 403
        assert client.post('/api/control',headers=headers|{'Origin':'https://evil.example'},json={'action':'demo'}).status_code == 403
        assert client.get('/api/status',headers={'Host':'evil.example'}).status_code == 403
        def command(action,**args):
            response = client.post('/api/control',headers=headers,json={'action':action,**args})
            assert response.status_code == 200,response.text
            return response.json()
        command('demo',profile='insole')
        start = command('record_start')
        with client.websocket_connect('/ws?token='+token) as ws:
            first = ws.receive_json()
            before = sum(d['frames'] for d in first['state']['devices'])
        time.sleep(.25)
        with client.websocket_connect('/ws?token='+token) as ws:
            after = ws.receive_json()
            assert after['state']['recorder']['recording']
            assert after['state']['recorder']['session_id'] == start['session_id']
            assert sum(d['frames'] for d in after['state']['devices']) > before
        command('record_stop')
        response = client.get('/api/sessions/'+start['session_id']+'/export')
        assert response.status_code == 200 and response.content[:2] == b'PK'


def envelope(seq=1,profile='wt',now=1000.):
    packet = demo_packet(profile,seq,'l',time.time()*1000)
    return {'v':1,'session':'testphone123','profile':profile,'phone_now_ms':now,'events':[
        {'v':1,'session':'testphone123','profile':profile,'id':seq,'phone_ms':now,'type':'notify',
         'data':{'foot':'l','connection':'conn1','receipt_ms':now,'bytes':base64.b64encode(packet).decode()}}]}


def test_phone_validation():
    validate(envelope())
    invalid = envelope(); invalid['events'][0]['data']['receipt_ms'] = 1001
    with pytest.raises(ValueError):validate(invalid)
    invalid = envelope(); invalid['phone_now_ms'] = float('nan')
    with pytest.raises(ValueError):validate(invalid)
    invalid = envelope(); invalid['ack_mode'] = 'ignore_disk'
    with pytest.raises(ValueError,match='acknowledgment'):validate(invalid)


def test_phone_dedup_and_separate_clock(tmp_path):
    async def run():
        c = Capture(tmp_path)
        await c.start(); await c.source_change('phone')
        phone = Phone(c)
        body = envelope()
        first = await phone.ingest(body)
        frames = c.devices['phone:l']['frames']
        second = await phone.ingest(body)
        assert first['ack_id'] == second['ack_id'] == 1
        assert c.devices['phone:l']['frames'] == frames
        assert c.phone['oneway_ms'] is None
        assert c.recorder.accepting
        assert c.history[-1]['receipt_ms'] > 1000
        raw = (c.recorder.current/'raw_events.jsonl').read_text(encoding='utf-8')
        assert 'phone_batch' in raw and 'notify' in raw
        phone.last_receive = time.monotonic()*1000-2000
        await asyncio.sleep(.3)
        assert c.phone['stale'] and not c.devices['phone:l']['connected']
        await c.record_stop()
        phone.paused = True
        with pytest.raises(ValueError,match='停止记录'):
            await phone.ingest(envelope(2,now=1001))
        await c.record_start(); phone.paused = False
        assert (await phone.ingest(envelope(2,now=1001)))['ack_id'] == 2
        await c.record_stop(); await phone.release(); await phone.close(); await c.close()
    asyncio.run(run())


def test_algorithm_and_bias_controls(tmp_path):
    async def run():
        c = Capture(tmp_path,demo=True)
        await c.start(); await c.demo('wt')
        result = await asyncio.wrap_future(c.algorithm.control('configure',profile='wt',enabled=True))
        assert result['model']['source'] == 'wt'
        await asyncio.sleep(3.5)
        state = c.algorithm.snapshot()
        assert state['error'] is None,state
        assert state['result'].get('published_host_ms') is not None,state
        assert state['result']['valid'],state
        await c.stop_virtual()
        await asyncio.wrap_future(c.algorithm.control('configure',profile='insole',enabled=False))
        result = await asyncio.wrap_future(c.algorithm.control('bias',foot='l',action='manual',clock_ms=time.monotonic()*1000))
        assert result['bias']['l']['mode'] == 'manual'
        await c.close()
    asyncio.run(run())


def test_bias_control_takes_device_key(tmp_path):
    with TestClient(create_app(tmp_path,demo=True)) as client:
        h = {'X-Walkpad-Token':client.get('/api/config').json()['control_token']}
        client.post('/api/control',json={'action':'demo','profile':'insole'},headers=h)
        time.sleep(1.5)
        keys = [(d['placement_label'],d['bias_key']) for d in client.get('/api/status').json()['devices']]
        assert keys == [('左脚背','l'),('右脚背','r')],keys
        response = client.post('/api/control',json={'action':'bias','bias_key':'l','bias_action':'manual'},headers=h)
        assert response.status_code == 200,response.text
        assert response.json()['bias']['l']['mode'] == 'manual'
        assert client.post('/api/control',json={'action':'bias','bias_key':'nope','bias_action':'manual'},headers=h).status_code == 400


def test_insole_bias_runs_at_non_foot_placement(tmp_path):
    async def run():
        c = Capture(tmp_path,demo=True)
        await c.start()
        await asyncio.wrap_future(c.algorithm.control('configure',profile='insole',enabled=False))
        d = c._device('wrist','腕','insole',None,'addr',placement='leftWrist',label='左手腕')
        d['connected'] = True; d['connection'] = 'conn1'; d['decoder'] = Decoder('insole')
        assert c.public(d)['bias_key'] == 'd:wrist'
        base = time.monotonic()*1000; wall = time.time()*1000
        for seq in range(1,120):
            c.submit('wrist',demo_packet('insole',seq,None,wall+seq*20),base+seq*20,wall+seq*20)
        await c.ingress.join()
        for _ in range(50):
            if c.algorithm.snapshot()['bias'].get('d:wrist'):
                break
            await asyncio.sleep(.1)
        state = c.algorithm.snapshot()['bias'].get('d:wrist')
        assert state and state['foot'] is None,state
        result = await asyncio.wrap_future(c.algorithm.control('bias',bias_key='d:wrist',foot=None,action='manual',clock_ms=base+120*20))
        assert result['bias']['d:wrist']['mode'] == 'manual'
        await c.close()
    asyncio.run(run())


def test_phone_native_http_and_pause(tmp_path):
    import socket
    import httpx
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        port = sock.getsockname()[1]
    app = create_app(tmp_path,phone_port=port)
    with TestClient(app) as client:
        token = client.get('/api/config').json()['control_token']
        headers = {'X-Walkpad-Token':token}
        def control(action,**args):
            return client.post('/api/control',headers=headers,json=dict(action=action,**args))
        assert control('source',source='phone').status_code == 200
        phone = app.state.phone
        url = 'http://127.0.0.1:%d/api/ingest' % port
        native = {'Authorization':'Bearer '+phone.token}
        assert httpx.post(url,json=envelope()).status_code == 403
        assert httpx.post(url,json=envelope(),headers=native|{'Origin':'http://evil.example'}).status_code == 403
        response = httpx.post(url,json=envelope(),headers=native)
        assert response.status_code == 200,response.text
        assert response.json()['ack_id'] == 1
        live = envelope(2,now=1020); live['ack_mode'] = 'received'
        response = httpx.post(url,json=live,headers=native)
        assert response.status_code == 200 and response.json()['received_id'] == 2
        assert response.json()['ack_id'] <= response.json()['received_id']
        raw = (app.state.capture.recorder.current/'raw_events.jsonl').read_text(encoding='utf-8')
        assert 'phone_batch' in raw and 'notify' in raw
        assert control('record_stop').status_code == 200
        assert httpx.post(url,json=envelope(2,now=1001),headers=native).status_code == 400
        assert control('source',source='direct').status_code == 400
        assert control('record_start').status_code == 200
        assert httpx.post(url,json=envelope(3,now=1040),headers=native).json()['ack_id'] == 3
        assert control('record_stop').status_code == 200
        assert control('phone_release').status_code == 200
        assert control('source',source='direct').status_code == 200


def test_phone_no_ack_on_storage_failure(tmp_path):
    async def run():
        c = Capture(tmp_path)
        phone = Phone(c)
        await c.start(); await c.source_change('phone')
        assert (await phone.ingest(envelope()))['ack_id'] == 1
        original = c.recorder.flush
        def fail():
            raise RuntimeError('disk unavailable')
        c.recorder.flush = fail
        with pytest.raises(RuntimeError,match='disk unavailable'):
            await phone.ingest(envelope(2,now=1001))
        assert phone.last_id == 1
        c.recorder.flush = original
        await phone.close(); await c.close()
    asyncio.run(run())


def test_phone_received_ack_does_not_wait_for_disk(tmp_path):
    import threading
    async def run():
        c = Capture(tmp_path)
        phone = Phone(c)
        await c.start(); await c.source_change('phone')
        await phone.ingest(envelope())
        entered = threading.Event(); release = threading.Event()
        checkpoint = c.recorder.checkpoint
        import concurrent.futures
        def slow_checkpoint():
            barrier = checkpoint()
            future = concurrent.futures.Future()
            def finish():
                entered.set()
                try:
                    if not release.wait(5): raise RuntimeError('test disk barrier timed out')
                    barrier.result(timeout=5); future.set_result(True)
                except Exception as exc: future.set_exception(exc)
            threading.Thread(target=finish,daemon=True).start()
            return future
        c.recorder.checkpoint = slow_checkpoint
        try:
            body = envelope(2,now=1020); body['ack_mode'] = 'received'
            response = await asyncio.wait_for(phone.ingest(body),1)
            assert response['received_id'] == 2
            assert response['ack_id'] == response['durable_id'] == 1
            await asyncio.to_thread(entered.wait,1)
            assert entered.is_set() and not release.is_set()
            retry = await phone.ingest(body)
            assert retry['received_id'] == 2 and c.devices['phone:l']['frames'] == 2
            body = envelope(3,now=1040); body['ack_mode'] = 'received'
            assert (await phone.ingest(body))['received_id'] == 3
            assert c.devices['phone:l']['frames'] == 3
            phone.paused = True
            await c.record_stop()
            release.set()
            await phone.flush_task
            c.recorder.checkpoint = checkpoint
            await c.record_start(); phone.paused = False
            final = envelope(4,now=1060)
            response = await phone.ingest(final)
            assert response['ack_id'] == response['received_id'] == 4
        finally:
            release.set(); c.recorder.checkpoint = checkpoint
            await phone.close(); await c.close()
    asyncio.run(run())


def test_phone_gap_inside_batch_resets_fragment_decoder(tmp_path):
    async def run():
        c = Capture(tmp_path); phone = Phone(c)
        await c.start(); await c.source_change('phone')
        boundaries = []
        original = c.algorithm.offer
        def offer(item):
            if item.get('control') == 'boundary': boundaries.append(item)
            original(item)
        c.algorithm.offer = offer
        packet = demo_packet('wt',1,'l',time.time()*1000)
        body = envelope()
        first = body['events'][0]
        first['data']['bytes'] = base64.b64encode(packet[:8]).decode()
        import copy
        second = copy.deepcopy(first)
        second['id'] = 3
        second['data']['bytes'] = base64.b64encode(packet[8:]).decode()
        body['events'].append(second)
        try:
            assert (await phone.ingest(body))['ack_id'] == 3
            assert c.devices['phone:l']['frames'] == 0
            assert any(x['reason'] == 'network_gap' for x in boundaries)
            assert (await phone.ingest(envelope(4,now=1020)))['ack_id'] == 4
            assert c.devices['phone:l']['frames'] == 1
        finally:
            await phone.close(); await c.close()
    asyncio.run(run())


def test_phone_background_disk_failure_never_advances_durable_ack(tmp_path):
    async def run():
        c = Capture(tmp_path); phone = Phone(c)
        await c.start(); await c.source_change('phone')
        await phone.ingest(envelope())
        import concurrent.futures
        def fail():
            future = concurrent.futures.Future()
            future.set_exception(RuntimeError('background disk unavailable'))
            return future
        c.recorder.checkpoint = fail
        body = envelope(2,now=1020); body['ack_mode'] = 'received'
        try:
            assert (await phone.ingest(body))['ack_id'] == 1
            with pytest.raises(RuntimeError,match='background disk unavailable'):
                await phone.flush_task
            assert phone.last_id == 1 and not c.recorder.accepting
            with pytest.raises(ValueError,match='background disk unavailable'):
                await phone.ingest(envelope(3,now=1040))
        finally:
            await phone.close(); await c.close()
    asyncio.run(run())


def test_connected_device_cannot_change_identity(tmp_path):
    async def run():
        c = Capture(tmp_path)
        d = c._device('test','test','wt','l','address1')
        d['connected'] = True
        assert (await c.connect('address1','wt','l'))['id'] == 'test'
        with pytest.raises(ValueError,match='更换协议或位置'):
            await c.connect('address1','wt','r')
        assert d['foot'] == 'l'
        await c.close()
    asyncio.run(run())


def test_placement_assigns_foot_and_allocates_custom(tmp_path):
    async def run():
        c = Capture(tmp_path)
        assert c.resolve_placement('leftInstep') == ('leftInstep','左脚背','l')
        assert c.resolve_placement('l') == ('leftInstep','左脚背','l')
        assert c.resolve_placement('leftWrist') == ('leftWrist','左手腕',None)
        assert c.resolve_placement('custom','腰部') == ('custom1','腰部',None)
        old = c._device('a','a','wt',None,'addr',placement='custom1',label='腰部')
        assert c.resolve_placement('custom','小腿') == ('custom2','小腿',None)
        # 换设备沿用同一位置时复用槽位；该槽位在线时不能重复占用
        assert c.resolve_placement('custom','腰部') == ('custom1','腰部',None)
        old['connected'] = True
        with pytest.raises(ValueError,match='已被占用'):
            c.resolve_placement('custom','腰部')
        with pytest.raises(ValueError,match='名称'):
            c.resolve_placement('custom','  ')
        with pytest.raises(ValueError,match='位置'):
            c.resolve_placement('waist')
        await c.close()
    asyncio.run(run())


def test_placement_is_unique_and_profile_free(tmp_path, monkeypatch):
    # 协议与位置解耦：位置只决定 foot，不限制协议；鞋垫标在非脚背位置按仅采集处理
    async def run():
        import backend.capture as capture_module
        async def missing(*args, **kwargs):
            return None
        monkeypatch.setattr(capture_module.BleakScanner, 'find_device_by_address', missing)
        c = Capture(tmp_path)
        d = c._device('a','a','wt',None,'addr',placement='leftWrist',label='左手腕')
        d['connected'] = True
        with pytest.raises(ValueError,match='该位置已有设备连接'):
            await c.connect('addr2','wt','leftWrist')
        with pytest.raises(ValueError,match='未扫描到设备'):
            await c.connect('addr3','insole','rightWrist')
        await c.close()
    asyncio.run(run())


def test_recorder_flush_requires_running(tmp_path):
    recorder = Recorder(tmp_path)
    with pytest.raises(RuntimeError,match='记录未运行'):
        recorder.flush()


def test_stale_connection_notification_ignored(tmp_path):
    async def run():
        c = Capture(tmp_path)
        await c.start()
        d = c._device('test','test','wt','l')
        d['generation'] = 2
        c.submit('test',demo_packet('wt',1,'l',time.time()*1000),1,1,generation=1)
        await c.ingress.join()
        assert d['frames'] == 0
        c.submit('test',demo_packet('wt',1,'l',time.time()*1000),1,1,generation=2)
        d['generation'] = 3
        await c.ingress.join()
        assert d['frames'] == 0
        await c.close()
    asyncio.run(run())


def test_legacy_reply_crc_and_measurement_isolation(tmp_path):
    from backend.protocol import crc_ccitt
    async def run():
        c = Capture(tmp_path)
        d = c._device('test','test','insole','l')
        d['command_type'] = 13; d['command_link'] = 'v1'
        d['command_future'] = asyncio.get_running_loop().create_future()
        frame = bytearray(83)
        frame[:3] = b'\x5a\x00\x0d'
        struct.pack_into('<H',frame,15,42)
        frame[20] = 4
        struct.pack_into('<H',frame,81,crc_ccitt(frame[:81]))
        bad = bytearray(frame); bad[20] = 3
        c._control_reply(d,bad)
        assert d['format'] is None and not d['command_future'].done()
        c._control_reply(d,demo_packet('insole',1,'l',time.time()*1000)+frame[:30])
        assert d['format'] is None
        c._control_reply(d,frame[30:])
        assert d['format'] == 4 and d['command_future'].done()
        await c.close()
    asyncio.run(run())


def test_replay(tmp_path):
    async def run():
        c = Capture(tmp_path,demo=True)
        await c.start(); await c.demo('wt'); await c.record_start(); await asyncio.sleep(.15)
        state = await c.record_stop()
        await c.replay(state['session_id'],speed=10)
        await c.replay_task
        assert any(d['frames'] > 0 for d in c.devices.values() if d['source'] == 'replay')
        await c.close()
    asyncio.run(run())


def test_record_pause_resume_splits_periods(tmp_path):
    async def run():
        c = Capture(tmp_path,demo=True)
        await c.start()
        await c.demo('insole')
        await c.record_start()
        await asyncio.sleep(.6)
        paused = await c.record_pause()
        assert paused['paused'] and not paused['recording'] and paused['periods'] == 1
        hold = paused['frames']
        await asyncio.sleep(.5)
        resumed = await c.record_resume()
        assert resumed['recording'] and resumed['periods'] == 2
        await asyncio.sleep(.6)
        state = await c.record_stop()
        assert not state['error'],state
        assert state['frames'] > hold
        manifest = json.loads((c.recorder.current/'manifest.json').read_text(encoding='utf-8'))
        assert [p['end_reason'] for p in manifest['periods']] == ['pause','stop']
        assert manifest['effective_time_rule'] == 'exclude_paused_intervals'
        # 暂停区间落在两个周期之间，有效时长必须短于整段会话时长
        assert state['effective_us'] < manifest['periods'][-1]['end_session_elapsed_us']
        with zipfile.ZipFile(export_package(c.recorder.current)) as archive:
            package = json.loads(archive.read('manifest.json'))
            rows = []
            for output in package['outputs']:
                if output['path'].startswith('frames/'):
                    rows += pq.read_table(io.BytesIO(archive.read(output['path'])),
                                          columns=['periodId','validity']).to_pylist()
        assert {r['periodId'] for r in rows} == {'p1','p2',None}
        assert any(r['validity'] == 'outside_period' for r in rows)
        assert len(package['sessionTimeline']['periods']) == 2
    asyncio.run(run())
