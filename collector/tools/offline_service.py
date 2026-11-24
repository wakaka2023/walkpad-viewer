"""离线动作标注任务服务：接收录像，排队跑离线包，返回动作分段。"""

import argparse
import collections
import csv
import hashlib
import ipaddress
import json
import os
import shutil
import signal
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

MAX_BODY = 4 * 1024 ** 3
QUEUE_MAX = 32
CHUNK = 4 * 1024 ** 2

JOBS = {}
LOCK = threading.Lock()
QUEUE = collections.deque()
WAKE = threading.Condition(LOCK)
CONFIG = {}


def now():
    return time.time()


def pick_gpu():
    """无指定卡时选空闲显存最大的卡，避开共享机其他账户在跑的卡。"""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=10).stdout
        rows = [ln.split(',') for ln in out.strip().splitlines() if ln.strip()]
        rows = [(int(i), int(m)) for i, m in rows]
        return str(max(rows, key=lambda r: r[1])[0])
    except Exception:
        return '0'


def job_dir(job_id):
    return Path(CONFIG['out_root']) / job_id


def save(job):
    tmp = job_dir(job['id']) / 'job.json.tmp'
    tmp.write_text(json.dumps(job, ensure_ascii=False), encoding='utf-8')
    tmp.replace(job_dir(job['id']) / 'job.json')


def log_tail(path, lines=4):
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 4096))
            text = f.read().decode('utf-8', 'ignore')
        return [ln for ln in text.splitlines() if ln.strip()][-lines:]
    except Exception:
        return []


SUPPORTED = {'.mp4', '.qt', '.mov', '.mkv', '.avi', '.m4v'}


def prepare(job):
    """离线包只认 mp4/mov/mkv 一类；浏览器录的 webm 先无转码重封装到 mkv。"""
    src = job_dir(job['id']) / job['video']
    if src.suffix.lower() in SUPPORTED:
        job['run_video'] = job['video']
        return src
    dst = src.with_suffix('.mkv')
    try:
        import av
        with av.open(str(src)) as fin, av.open(str(dst), 'w') as fout:
            mapping = {}
            for stream in fin.streams:
                if stream.type in ('video', 'audio'):
                    mapping[stream.index] = fout.add_stream(template=stream)
            for packet in fin.demux():
                target = mapping.get(packet.stream.index)
                if packet.dts is None or target is None:
                    continue
                packet.stream = target
                fout.mux(packet)
    except Exception:
        shutil.copyfile(src, dst)
    job['run_video'] = dst.name
    return dst


def run_job(job):
    pkg = Path(CONFIG['pkg'])
    video = prepare(job)
    out = job_dir(job['id']) / 'out'
    out.mkdir(exist_ok=True)
    log = job_dir(job['id']) / 'run.log'
    gpu = CONFIG['gpu'] or pick_gpu()
    cmd = [CONFIG['python'], 'recognize_actions.py', '--multi-person', str(video), '--output', str(out),
           '--device', '0']
    if not job.get('preview', True):
        cmd.append('--no-preview')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED='1')
    job.update(state='running', started=now(), gpu=gpu)
    save(job)
    with open(log, 'wb') as fh:
        # 包内是裸名导入，cwd 必须是包目录
        proc = subprocess.Popen(cmd, cwd=str(pkg), stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True, env=env)
        job['pid'] = proc.pid
        save(job)
        code = proc.wait()
    job['pid'] = 0
    job['elapsed_s'] = round(now() - job['started'], 1)
    if job.get('cancel'):
        job['state'] = 'cancelled'
    elif code == 0 and folder(job):
        job['state'] = 'done'
    else:
        job['state'] = 'failed'
        job['error'] = 'exit code %s' % code
    save(job)


def folder(job):
    out = job_dir(job['id']) / 'out'
    stem = Path(job.get('run_video') or job['video']).stem
    return out / stem if (out / stem).is_dir() else None


def worker():
    while True:
        with WAKE:
            while not QUEUE:
                WAKE.wait()
            job = QUEUE.popleft()
            job['state'] = 'running'
        try:
            run_job(job)
        except Exception as exc:
            job.update(state='failed', error=repr(exc))
            save(job)


def read_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as fh:
        return list(csv.DictReader(fh))


def result_payload(job):
    base = folder(job)
    if base is None:
        return None
    acts = read_csv(base / 'activities.csv') if (base / 'activities.csv').exists() else []
    tracks = read_csv(base / 'tracks.csv') if (base / 'tracks.csv').exists() else []
    meta = {}
    if (base / 'metadata.json').exists():
        meta = json.loads((base / 'metadata.json').read_text(encoding='utf-8'))
    keep = ('track_id', 'track_fragment', 'segment_id', 'activity', 'activity_zh', 'start_video_s',
            'end_video_s', 'evidence_level', 'start_video_frame', 'end_video_frame', 'repetition_count')
    segments = []
    for row in acts:
        item = {k: row.get(k) for k in keep}
        for k in ('start_video_s', 'end_video_s'):
            item[k] = float(item[k]) if item[k] else None
        for k in ('track_id', 'track_fragment', 'segment_id', 'start_video_frame', 'end_video_frame', 'repetition_count'):
            item[k] = int(item[k]) if item[k] else None
        segments.append(item)
    people = []
    for row in tracks:
        if not (row.get('track_id') or '').strip():
            continue
        people.append({
            'track_id': int(row['track_id']),
            'accepted': row.get('accepted') == 'True',
            'start_s': float(row['start_video_s']) if row.get('start_video_s') else None,
            'end_s': float(row['end_video_s']) if row.get('end_video_s') else None,
            'observed_frames': int(row['observed_frames']) if row.get('observed_frames') else None,
        })
    return {
        'job': job['id'],
        'video': job['video'],
        'version': meta.get('version'),
        'action_rules': meta.get('action_rules'),
        'fps': meta.get('fps'),
        'frame_count': meta.get('frame_count'),
        'people': people,
        'segments': segments,
        'products': sorted(p.name for p in base.iterdir()) if base else [],
    }


class Handler(BaseHTTPRequestHandler):
    server_version = 'walkpad-offline/1'

    def log_message(self, fmt, *args):
        pass

    def allowed(self):
        net = CONFIG['allow_lan']
        if not net:
            return True
        ip = ipaddress.ip_address(self.client_address[0])
        return ip.is_loopback or ip in ipaddress.ip_network(net, strict=False)

    def send_json(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def authed(self):
        if not self.allowed():
            self.send_json(403, {'error': 'forbidden'})
            return False
        token = CONFIG['token']
        if token and self.headers.get('X-Token') != token:
            self.send_json(401, {'error': 'token required'})
            return False
        return True

    def parts(self):
        path = urlparse(self.path).path.strip('/')
        return path.split('/') if path else []

    def do_GET(self):
        try:
            self.get()
        except Exception as exc:
            self.send_json(500, {'error': repr(exc)})

    def get(self):
        if not self.authed():
            return
        parts = self.parts()
        if parts == ['health']:
            with LOCK:
                return self.send_json(200, {
                    'ok': True, 'pkg': CONFIG['pkg'], 'jobs': len(JOBS),
                    'queued': len(QUEUE), 'running': sum(1 for j in JOBS.values() if j['state'] == 'running'),
                    'busy': bool(QUEUE) or any(j['state'] == 'running' for j in JOBS.values()),
                })
        if len(parts) >= 2 and parts[0] == 'jobs':
            with LOCK:
                job = JOBS.get(parts[1])
            if job is None:
                return self.send_json(404, {'error': 'no such job'})
            if len(parts) == 2:
                out = {k: job.get(k) for k in ('id', 'state', 'video', 'bytes', 'sha256', 'created',
                                               'started', 'elapsed_s', 'gpu', 'error', 'cancel')}
                out['log_tail'] = log_tail(job_dir(job['id']) / 'run.log')
                return self.send_json(200, out)
            if parts[2] == 'result':
                payload = result_payload(job)
                if payload is None:
                    return self.send_json(409, {'error': 'job %s' % job['state']})
                return self.send_json(200, payload)
            if parts[2] == 'log':
                q = parse_qs(urlparse(self.path).query)
                lines = int(q.get('tail', ['80'])[0])
                return self.send_json(200, {'log_tail': log_tail(job_dir(job['id']) / 'run.log', lines)})
            if parts[2] == 'file':
                q = parse_qs(urlparse(self.path).query)
                name = q.get('name', [''])[0]
                base = folder(job)
                if base is None or name not in {p.name for p in base.iterdir()}:
                    return self.send_json(404, {'error': 'no such file'})
                data = (base / name).read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
        return self.send_json(404, {'error': 'not found'})

    def do_POST(self):
        try:
            self.post()
        except Exception as exc:
            self.send_json(500, {'error': repr(exc)})

    def post(self):
        if not self.authed():
            return
        parts = self.parts()
        if parts == ['jobs']:
            return self.create()
        if len(parts) == 3 and parts[0] == 'jobs' and parts[2] == 'cancel':
            with LOCK:
                job = JOBS.get(parts[1])
                if job is None:
                    return self.send_json(404, {'error': 'no such job'})
                job['cancel'] = True
                pid = job.get('pid')
                queued = job in QUEUE
                if queued:
                    QUEUE.remove(job)
                    job['state'] = 'cancelled'
                save(job)
            if pid:
                try:
                    os.killpg(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            return self.send_json(200, {'id': job['id'], 'state': job['state']})
        return self.send_json(404, {'error': 'not found'})

    def create(self):
        length = int(self.headers.get('Content-Length') or 0)
        if length <= 0 or length > MAX_BODY:
            return self.send_json(400, {'error': 'bad length'})
        name = os.path.basename(unquote(self.headers.get('X-Filename') or 'video.mp4'))
        if Path(name).suffix.lower() not in ('.mp4', '.webm', '.mov', '.mkv', '.avi'):
            return self.send_json(400, {'error': 'unsupported video'})
        with LOCK:
            if len(QUEUE) >= QUEUE_MAX:
                return self.send_json(429, {'error': 'queue full'})
            job_id = uuid.uuid4().hex[:16]
            job = {'id': job_id, 'state': 'uploading', 'video': name, 'bytes': 0,
                   'created': now(), 'sha256': '', 'pid': 0,
                   'preview': self.headers.get('X-Preview', '1') != '0'}
            JOBS[job_id] = job
            job_dir(job_id).mkdir(parents=True, exist_ok=True)
        part = job_dir(job_id) / (name + '.part')
        digest = hashlib.sha256()
        left = length
        with open(part, 'wb') as fh:
            while left > 0:
                chunk = self.rfile.read(min(CHUNK, left))
                if not chunk:
                    break
                fh.write(chunk)
                digest.update(chunk)
                left -= len(chunk)
        if left:
            shutil.rmtree(job_dir(job_id), ignore_errors=True)
            with LOCK:
                JOBS.pop(job_id, None)
            return self.send_json(400, {'error': 'short body'})
        part.replace(job_dir(job_id) / name)
        with LOCK:
            job.update(state='queued', bytes=length, sha256=digest.hexdigest())
            save(job)
            QUEUE.append(job)
            WAKE.notify()
        return self.send_json(202, {'id': job_id, 'state': 'queued', 'bytes': length})


def recover(out_root):
    if not out_root.exists():
        return
    for path in out_root.glob('*/job.json'):
        try:
            job = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        if job.get('state') in ('running', 'uploading', 'queued'):
            job.update(state='interrupted', pid=0, error='service restarted')
            path.write_text(json.dumps(job, ensure_ascii=False), encoding='utf-8')
        JOBS[job['id']] = job


def main():
    parser = argparse.ArgumentParser(description='离线动作标注任务服务')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=18844)
    parser.add_argument('--pkg', default='/home/wjb-006/walkpad-vision/offline/v4.9.8-r13-gait-switch-fix')
    parser.add_argument('--out-root', default='/home/wjb-006/walkpad-vision/runtime/offline_jobs')
    parser.add_argument('--python', default='/home/wjb-006/squat_venv/bin/python')
    parser.add_argument('--gpu', default=os.environ.get('WALKPAD_OFFLINE_GPU', ''))
    parser.add_argument('--allow-lan', default='')
    parser.add_argument('--token', default=os.environ.get('WALKPAD_OFFLINE_TOKEN', ''))
    args = parser.parse_args()
    CONFIG.update(pkg=args.pkg, out_root=Path(args.out_root), python=args.python,
                  gpu=args.gpu, allow_lan=args.allow_lan, token=args.token)
    CONFIG['out_root'].mkdir(parents=True, exist_ok=True)
    recover(CONFIG['out_root'])
    threading.Thread(target=worker, daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print('offline service on %s:%d pkg=%s' % (args.host, args.port, args.pkg), flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
