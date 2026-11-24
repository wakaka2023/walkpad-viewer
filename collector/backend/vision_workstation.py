import argparse
import asyncio
import json
import os
import ssl
import sys
from contextlib import suppress
from ipaddress import IPv4Address, IPv4Network, ip_address
from pathlib import Path

from .version import VERSION
from .vision_worker import (
    ACTION_RULES, COLLECTION_ID_MODE, DEFAULT_RULES, HEADER, IDENTITY_MODES as ID_MODELS, MAX_FRAME, VISION,
)

DEFAULT_ENDPOINT = ('192.168.100.29', 18843)
LAN_CA = Path(__file__).with_name('vision_lan_ca.crt')
PROTOCOLS = {(mode, rules): f'walkpad-vision-{mode}-{rules}/1'
             for mode in ID_MODELS for rules in ACTION_RULES}


class WorkstationError(ValueError):
    pass


# 已发布的 r5/r6/r10 客户端可继续连接，ready 中回报实际的规则版本。
NEGOTIATED = {name: pair for pair, name in PROTOCOLS.items()}
NEGOTIATED.update({f'walkpad-vision-{prefix}r5/1': (mode, DEFAULT_RULES)
                   for mode, prefix in (('bytetrack', ''), ('reid', 'reid-'),
                                        ('tracktrack-reid', 'tracktrack-reid-'),
                                        (COLLECTION_ID_MODE, 'collection-reid-'))})
NEGOTIATED.update({f'walkpad-vision-{mode}-r6/1': (mode, DEFAULT_RULES) for mode in ID_MODELS})
NEGOTIATED.update({f'walkpad-vision-{mode}-r10/1': (mode, DEFAULT_RULES) for mode in ID_MODELS})


def selection(protocol):
    return NEGOTIATED.get(protocol)


async def connect(config_path, id_model='bytetrack', action_rules=DEFAULT_RULES):
    if id_model not in ID_MODELS:
        raise WorkstationError('不支持的 ID 识别模型')
    if (id_model, action_rules) not in PROTOCOLS:
        raise WorkstationError('该 ID 模型不支持所选动作规则')
    protocol = PROTOCOLS[(id_model, action_rules)]
    try:
        try:
            config = json.loads(config_path.read_text(encoding='utf-8'))
        except FileNotFoundError:
            host, port = DEFAULT_ENDPOINT
            mutual_tls = False
        else:
            host, port = config['host'], config['port']
            mutual_tls = config.get('client_auth', (host, port) != DEFAULT_ENDPOINT)
        if (not isinstance(host, str) or not host or type(port) is not int
                or not 0 < port < 65536 or type(mutual_tls) is not bool):
            raise ValueError('invalid endpoint')
        tls = config_path.parent / 'tls'
        ca = tls / 'ca.crt' if mutual_tls or (host, port) != DEFAULT_ENDPOINT else LAN_CA
        context = ssl.create_default_context(cafile=str(ca))
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        if protocol:
            context.set_alpn_protocols([protocol])
        if mutual_tls:
            context.load_cert_chain(tls / 'client.crt', tls / 'client.key')
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise WorkstationError('工作站地址或证书配置无效') from exc
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port, ssl=context), 5.)
    except ssl.SSLError as exc:
        raise WorkstationError('工作站证书验证失败，未发送识别画面') from exc
    except (OSError, TimeoutError) as exc:
        raise WorkstationError('无法连接工作站，请检查公司局域网或切换到本机') from exc
    if protocol and writer.get_extra_info('ssl_object').selected_alpn_protocol() != protocol:
        await close(writer)
        raise WorkstationError('工作站尚未支持所选识别选项，请更新工作站服务或切换选项')
    return reader, writer


async def message(reader, timeout):
    line = await asyncio.wait_for(reader.readline(), timeout)
    if not line:
        raise EOFError
    return json.loads(line)


async def send(writer, data):
    writer.write(data)
    await asyncio.wait_for(writer.drain(), 5.)


async def close(writer):
    writer.close()
    with suppress(OSError, TimeoutError):
        await asyncio.wait_for(writer.wait_closed(), 2.)


async def encode(writer, payload):
    await send(writer, (json.dumps(payload, ensure_ascii=True, allow_nan=False) + '\n').encode('ascii'))


def select_gpu(snapshot):
    eligible = []
    for row in snapshot.splitlines():
        fields = [v.strip() for v in row.split(',')]
        if len(fields) != 5 or not fields[1].startswith('GPU-'):
            continue
        try:
            index = int(fields[0])
            used, free, utilization = map(int, fields[2:])
        except ValueError:
            continue
        if 0 <= utilization < 10 and 0 <= used <= 2048 and free >= 8192:
            eligible.append((utilization, used, -free, index, fields[1]))
    return min(eligible)[-2:] if eligible else None


async def available_gpu():
    proc = await asyncio.create_subprocess_exec(
        'nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits', stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), 3.)
    finally:
        if proc.returncode is None:
            with suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
    if proc.returncode:
        raise OSError('nvidia-smi failed')
    return select_gpu(out.decode('ascii'))


def service_context(tls, *, require_client_cert=True):
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.set_alpn_protocols(list(NEGOTIATED))
    context.load_cert_chain(tls / 'server.crt', tls / 'server.key')
    if require_client_cert:
        context.load_verify_locations(cafile=str(tls / 'ca.crt'))
        context.verify_mode = ssl.CERT_REQUIRED
    return context


def lan_network(host, cidr):
    network = IPv4Network(cidr)
    private = ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16')
    if IPv4Address(host) not in network or not any(network.subnet_of(IPv4Network(n)) for n in private):
        raise ValueError('免凭证接入必须绑定到指定私有 IPv4 网段内的地址')
    return network


class VisionService:
    def __init__(self, network=None):
        self.active = False
        self.network = network

    async def serve(self, reader, writer):
        if self.network is not None:
            peer = writer.get_extra_info('peername')
            if not peer or ip_address(peer[0]) not in self.network:
                try:
                    await encode(writer, {'type': 'error', 'message': '不在允许的采集局域网网段'})
                finally:
                    await close(writer)
                return
        protocol = writer.get_extra_info('ssl_object').selected_alpn_protocol()
        selected = selection(protocol)
        if selected is None:
            try:
                await encode(writer, {'type': 'error', 'message': '工作站仅支持 r13 和实时分相，请更新采集器'})
            finally:
                await close(writer)
            return
        if self.active:
            try:
                await encode(writer, {'type': 'error', 'message': '工作站已有一个实时识别会话'})
            finally:
                await close(writer)
            return
        self.active = True
        proc = None
        try:
            try:
                gpu = await available_gpu()
            except (OSError, TimeoutError) as exc:
                print(f'Vision GPU query: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
                await encode(writer, {'type': 'error', 'message': '工作站无法查询 GPU 状态，请检查 NVIDIA 驱动或稍后重试'})
                return
            if gpu is None:
                await encode(writer, {'type': 'error', 'message': '工作站暂无符合负载与显存要求的 GPU，请稍后重试或切换到本机'})
                return
            index, uuid = gpu
            id_model, action_rules = selected
            proc = await asyncio.create_subprocess_exec(
                sys.executable, '-u', '-m', 'backend.vision_worker',
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': uuid, 'WALKPAD_VISION_REQUIRE_GPU': '1',
                     'WALKPAD_VISION_ID_MODEL': id_model, 'WALKPAD_VISION_ACTION_RULES': action_rules},
            )
            ready = await message(proc.stdout, 55.)
            if ready.get('type') == 'ready':
                ready['device'] = f'工作站 GPU {index}'
                ready['server_version'] = VERSION
            await encode(writer, ready)
            if ready.get('type') != 'ready':
                return
            while True:
                length = int.from_bytes(await asyncio.wait_for(reader.readexactly(4), 30.), 'big')
                if length == 0 and id_model == COLLECTION_ID_MODE:
                    await send(proc.stdin, bytes(4))
                    await encode(writer, await message(proc.stdout, 10.))
                    return
                if not HEADER.size < length <= MAX_FRAME:
                    raise ValueError('识别帧大小无效')
                packet = await asyncio.wait_for(reader.readexactly(length), 5.)
                await send(proc.stdin, length.to_bytes(4, 'big') + packet)
                result = await message(proc.stdout, 10.)
                await encode(writer, result)
                if result.get('type') != 'result':
                    return
        except (EOFError, OSError, ValueError) as exc:
            if not isinstance(exc, asyncio.IncompleteReadError):
                if not isinstance(exc, ConnectionError):
                    print(f'Vision session: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
                with suppress(OSError):
                    await encode(writer, {'type': 'error', 'message': '工作站识别会话中断，请重新开启'})
        finally:
            if proc is not None:
                if proc.returncode is None:
                    with suppress(ProcessLookupError):
                        proc.kill()
                await proc.wait()
            self.active = False
            await close(writer)


async def run(host, port, network=None):
    service = VisionService(network)
    context = service_context(VISION / 'tls', require_client_cert=network is None)
    server = await asyncio.start_server(service.serve, host, port, ssl=context,
                                        ssl_handshake_timeout=5., ssl_shutdown_timeout=2.)
    mode = f'LAN TLS, {network}' if network is not None else 'mutual TLS'
    print(f'Walkpad Vision listening on {host}:{port} ({mode})', flush=True)
    async with server:
        await server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, default=DEFAULT_ENDPOINT[1])
    parser.add_argument('--allow-lan', metavar='CIDR', help='该私有 IPv4 网段无需客户端证书')
    args = parser.parse_args()
    try:
        network = lan_network(args.host, args.allow_lan) if args.allow_lan else None
    except ValueError as exc:
        parser.error(str(exc))
    asyncio.run(run(args.host, args.port, network))
