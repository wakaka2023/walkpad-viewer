import asyncio
import json
import ssl
import struct
import sys
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.vision_actions import LiveActions
from backend.vision_bridge import VisionBridge


def standing():
    pose = [(0., 0., 0.)] * 17
    for x, hip, knee, foot in ((.32, 11, 13, 15), (.52, 12, 14, 16)):
        pose[hip], pose[knee], pose[foot] = (x,.40,.9), (x,.55,.9), (x,.70,.9)
    return pose


def lowered():
    pose = standing()
    for x, hip, knee in ((.32, 11, 13), (.52, 12, 14)):
        pose[hip], pose[knee] = (x,.55,.9), (x+.105,.59,.9)
    return pose


def people(actions, t, *poses):
    return actions.frame(t, [(i+1,[0.,0.,1.,1.],pose) for i, pose in enumerate(poses)])


def test_two_tracks_count_only_upright_return():
    a = LiveActions()
    for i in range(15):
        last = people(a,i*.08,standing(), standing())
    assert [p['action'] for p in last] == ['站立','站立']
    for i in range(15,25):
        last = people(a,i*.08,lowered(),standing())
    assert last[0]['status'] == 'candidate' and last[0]['count'] == 0
    assert last[1]['action'] == '站立'
    for i in range(25,34):
        last = people(a,i*.08,standing(),standing())
    assert [p['count'] for p in last] == [1,0]


def test_gap_cannot_complete_repetition():
    a = LiveActions()
    for i in range(15):
        people(a,i*.08,standing())
    for i in range(15,24):
        last = people(a,i*.08,lowered())
    assert last[0]['status'] == 'candidate'
    assert people(a,2.8,standing())[0]['count'] == 0
    assert people(a,6.2,lowered())[0]['status'] == 'unknown'


def test_bridge_auth_frames_and_reconnect(monkeypatch):
    import backend.vision_bridge as bridge_module
    spawn = bridge_module.asyncio.create_subprocess_exec
    stub = ('import json, struct, sys; f=sys.stdin.buffer; '
            'import os; print(json.dumps(dict(type="ready",device=os.environ["WALKPAD_VISION_ID_MODEL"])),flush=True); '
            'exec("while size:=f.read(4):\\n p=f.read(int.from_bytes(size,\'big\')); n,t=struct.unpack_from(\'!Id\',p); print(json.dumps(dict(type=\'result\',seq=n,t=t,people=[])),flush=True)")')

    async def fake_worker(*args, **kwargs):
        return await spawn(sys.executable, '-u', '-c', stub, **kwargs)

    monkeypatch.setattr(bridge_module.asyncio, 'create_subprocess_exec', fake_worker)
    app = FastAPI()
    app.state.token = 'sample-secret'
    bridge = VisionBridge()

    @app.websocket('/ws/vision')
    async def socket(ws: WebSocket):
        await bridge.serve(ws)

    origin = {'origin':'http://testserver'}
    protocols = ['walkpad-vision', 'sample-secret']
    with TestClient(app) as client:
        for headers, sub in [({'origin':'http://elsewhere'}, protocols),(origin,['walkpad-vision','invalid'])]:
            try:
                with client.websocket_connect('/ws/vision', headers=dict(headers), subprotocols=sub):
                    raise AssertionError('unauthorized connection accepted')
            except WebSocketDisconnect as exc:
                assert exc.code == 1008
        with client.websocket_connect('/ws/vision?id_model=invalid', headers=dict(origin), subprotocols=protocols) as ws:
            assert ws.receive_json() == {'type': 'error', 'message': '不支持的 ID 识别模型'}
            try:
                ws.receive_json()
            except WebSocketDisconnect:
                pass
        for seq, query, model in ((11, '', 'bytetrack'), (12, '?id_model=reid', 'reid')):
            with client.websocket_connect('/ws/vision'+query, headers=dict(origin), subprotocols=protocols) as ws:
                assert ws.accepted_subprotocol == 'walkpad-vision'
                assert ws.receive_json() == {'type': 'ready', 'device': model}
                ws.send_bytes(struct.pack('!Id', seq, .5)+b'fake')
                assert ws.receive_json() == {'type':'result','seq':seq,'t':.5,'people':[]}
                ws.send_bytes(b'invalid')
                assert ws.receive_json()['type'] == 'error'
                try:
                    ws.receive_json()
                except WebSocketDisconnect:
                    pass


def test_action_protocols_upgrade_r5_clients_without_running_retired_rules():
    from backend.vision_workstation import ID_MODELS, PROTOCOLS, selection
    assert set(PROTOCOLS) == {(mode, rule) for mode in ID_MODELS for rule in ('r13', 'live')}
    for pair, protocol in PROTOCOLS.items():
        assert selection(protocol) == pair
    assert selection('walkpad-vision-r5/1') == ('bytetrack', 'r13')
    assert selection('walkpad-vision-collection-reid-r5/1') == ('collection-reid', 'r13')
    assert selection('walkpad-vision-reid-r10/1') == ('reid', 'r13')
    assert selection('walkpad-vision-tracktrack-reid-r10/1') == ('tracktrack-reid', 'r13')
    assert selection('walkpad-vision-r4/1') is None
    assert selection(None) is None


@pytest.mark.parametrize('snapshot, expected', [
    ('0,GPU-a,914,47490,0\n1,GPU-b,531,47873,0', (1, 'GPU-b')),
    ('0,GPU-a,0,49000,44\n1,GPU-b,531,47873,0', (1, 'GPU-b')),
    ('0,GPU-a,100,4096,0\n1,GPU-b,4096,46100,0', None),
    ('0,GPU-a,100,45000,N/A\n1,GPU-b,1000,45000,4\n2,GPU-c,888,45000,7', (1, 'GPU-b')),
])
def test_gpu_selection_allows_light_residency_but_avoids_busy_cards(snapshot, expected):
    from backend.vision_workstation import select_gpu
    assert select_gpu(snapshot) == expected


@pytest.mark.parametrize('override', [None, 'default', 'custom-tls', 'custom-mutual'])
def test_workstation_tls_client_config(monkeypatch, tmp_path, override):
    from backend import vision_workstation as service

    config = tmp_path / 'workstation.json'
    mutual = override == 'custom-mutual'
    custom = override in ('custom-tls', 'custom-mutual')
    host, port = ('192.168.10.20', 18844) if custom else service.DEFAULT_ENDPOINT
    if override:
        data = {'host': host, 'port': port}
        if override == 'custom-tls':
            data['client_auth'] = False
        config.write_text(json.dumps(data), encoding='utf-8')
    writer = Mock()
    writer.get_extra_info.return_value.selected_alpn_protocol.return_value = service.PROTOCOLS['collection-reid', 'r13']
    dial = AsyncMock(return_value=(None, writer))
    context = Mock()
    trust = Mock(return_value=context)
    real_context = service.ssl.create_default_context(cafile=str(service.LAN_CA))
    assert real_context.check_hostname and real_context.verify_mode == ssl.CERT_REQUIRED
    monkeypatch.setattr(service.ssl, 'create_default_context', trust)
    monkeypatch.setattr(service.asyncio, 'open_connection', dial)
    asyncio.run(service.connect(config, 'collection-reid', 'r13'))
    trust.assert_called_once_with(cafile=str(tmp_path / 'tls' / 'ca.crt' if custom else service.LAN_CA))
    dial.assert_awaited_once_with(host, port, ssl=context)
    if mutual:
        context.load_cert_chain.assert_called_once_with(tmp_path / 'tls' / 'client.crt', tmp_path / 'tls' / 'client.key')
    else:
        context.load_cert_chain.assert_not_called()


@pytest.mark.parametrize('host,cidr', [('0.0.0.0', '192.168.100.0/24'),
                                       ('192.168.100.29', '0.0.0.0/0'),
                                       ('192.168.100.29', '192.168.101.0/24')])
def test_credential_free_listener_must_bind_inside_private_network(host, cidr):
    from backend.vision_workstation import lan_network

    with pytest.raises(ValueError):
        lan_network(host, cidr)


def test_workstation_rejects_out_of_network_before_gpu(monkeypatch):
    from backend import vision_workstation as service

    gpu = AsyncMock()
    monkeypatch.setattr(service, 'available_gpu', gpu)
    worker = service.VisionService(service.lan_network('192.168.100.29', '192.168.100.0/24'))
    writer = Mock()
    writer.get_extra_info.return_value = ('192.168.101.42', 45678)
    writer.drain, writer.wait_closed = AsyncMock(), AsyncMock()
    asyncio.run(worker.serve(None, writer))
    assert json.loads(writer.write.call_args.args[0])['type'] == 'error'
    gpu.assert_not_awaited()
    writer.close.assert_called_once()
    assert not worker.active


@pytest.mark.parametrize('model,rules', [('reid', 'r13'), ('tracktrack-reid', 'r13')])
def test_workstation_bridge_routes_and_reports_disconnect(monkeypatch, model, rules):
    import backend.vision_bridge as bridge_module

    async def infer(reader, writer):
        try:
            writer.write(b'{"type":"ready","device":"Workstation GPU"}\n')
            size = int.from_bytes(await reader.readexactly(4), 'big')
            seq, t = struct.unpack_from('!Id', await reader.readexactly(size))
            writer.write(f'{{"type":"result","seq":{seq},"t":{t},"people":[]}}\n'.encode())
            await reader.readexactly(4)
        finally:
            writer.close()
            await writer.wait_closed()

    async def stub_connect(config, id_model, action_rules):
        assert config.name == 'workstation.json'
        assert id_model == model
        assert action_rules == rules
        async with await asyncio.start_server(infer, '127.0.0.1', 0) as server:
            return await asyncio.open_connection(*server.sockets[0].getsockname())

    async def no_local_process(*args, **kwargs):
        raise AssertionError('unexpected local process')

    monkeypatch.setattr(bridge_module, 'connect', stub_connect)
    monkeypatch.setattr(bridge_module.asyncio, 'create_subprocess_exec', no_local_process)
    app = FastAPI()
    app.state.token = 'sample-secret'
    app.websocket('/ws/vision')(VisionBridge().serve)
    with TestClient(app) as client:
        with client.websocket_connect(f'/ws/vision?target=workstation&id_model={model}&action_rules={rules}',
                                      headers={'origin':'http://testserver'},
                                      subprotocols=['walkpad-vision', 'sample-secret']) as ws:
            assert ws.receive_json() == {'type':'ready','device':'Workstation GPU'}
            ws.send_bytes(struct.pack('!Id', 7, .5)+b'fake')
            assert ws.receive_json() == {'type':'result','seq':7,'t':.5,'people':[]}
            ws.send_bytes(struct.pack('!Id', 8, .6)+b'fake')
            assert '工作站识别连接已断开' in ws.receive_json()['message']


@pytest.mark.parametrize('rule', ['r13', 'live'])
def test_collection_stop_relays_review_without_turning_it_into_an_action(monkeypatch, rule):
    import backend.vision_bridge as bridge_module

    async def infer(reader, writer):
        try:
            writer.write((json.dumps(dict(type='ready', identity='collection-reid', rules=rule)) + '\n').encode())
            assert await reader.readexactly(4) == bytes(4)
            writer.write(b'{"type":"review","overflow":false,"tracks":[{"person_id":null,"suggested_id":2}]}\n')
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def stub_connect(config, id_model, action_rules):
        assert (id_model, action_rules) == ('collection-reid', rule)
        async with await asyncio.start_server(infer, '127.0.0.1', 0) as server:
            return await asyncio.open_connection(*server.sockets[0].getsockname())

    monkeypatch.setattr(bridge_module, 'connect', stub_connect)
    app = FastAPI()
    app.state.token = 'sample-secret'
    app.websocket('/ws/vision')(VisionBridge().serve)
    with TestClient(app) as client:
        with client.websocket_connect(f'/ws/vision?target=workstation&id_model=collection-reid&action_rules={rule}',
                                      headers={'origin': 'http://testserver'},
                                      subprotocols=['walkpad-vision', 'sample-secret']) as ws:
            assert ws.receive_json()['type'] == 'ready'
            ws.send_bytes(bytes(4))
            assert ws.receive_json() == {'type': 'review', 'overflow': False,
                                         'tracks': [{'person_id': None, 'suggested_id': 2}]}


def test_action_rules_env_accepts_retired_r10_but_rejects_unknown(monkeypatch):
    from backend.vision_worker import DEFAULT_RULES, action_rules
    monkeypatch.delenv('WALKPAD_VISION_ACTION_RULES', raising=False)
    assert action_rules() == DEFAULT_RULES
    monkeypatch.setenv('WALKPAD_VISION_ACTION_RULES', 'r10')
    assert action_rules() == DEFAULT_RULES
    monkeypatch.setenv('WALKPAD_VISION_ACTION_RULES', 'live')
    assert action_rules() == 'live'
    monkeypatch.setenv('WALKPAD_VISION_ACTION_RULES', 'r5')
    with pytest.raises(ValueError):
        action_rules()
