import http.server
import json
import threading
import zipfile
from urllib.parse import quote, unquote

import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.offline import OfflineError, OfflineJobs


def build(tmp_path, monkeypatch):
    records = tmp_path/'records'
    (records/'s1').mkdir(parents=True)
    jobs = OfflineJobs(tmp_path/'offline',records,url='http://node:18844')
    calls = {'upload':[], 'get':[], 'post':[], 'files':{}}
    def upload(path,filename,headers):
        calls['upload'].append((path,filename,headers))
        return {'id':'r'*16,'state':'queued'}
    def get(path,timeout=None,raw=False):
        calls['get'].append(path)
        if path == '/jobs/'+'r'*16:
            return {'id':'r'*16,'state':'done','elapsed_s':12.5,'gpu':'0','log_tail':['ok'],'error':None}
        if path.endswith('/result'):
            return {'version':'v4.9.8-r13','action_rules':'r13','fps':25,'frame_count':100,
                    'people':[{'track_id':1}],
                    'segments':[{'track_id':1,'activity':'walking','activity_zh':'走路','start_video_s':1.0,
                                 'end_video_s':2.0,'duration_s':1.0,'evidence_level':'direct','reason':'x',
                                 'repetition_count':3,'unknown_field':'drop'}],
                    'products':['review.mp4','activities.csv']}
        if path.startswith('/jobs/'+'r'*16+'/file'):
            return calls['files'].get(path.split('name=')[1],b'data')
        return {'ok':True}
    def post(path,data=b''):
        calls['post'].append(path)
        return {'id':'r'*16,'state':'cancelled'}
    monkeypatch.setattr(jobs,'_upload',upload)
    monkeypatch.setattr(jobs,'_get',get)
    monkeypatch.setattr(jobs,'_post',post)
    return jobs, calls


def test_submit_records_job(tmp_path, monkeypatch):
    jobs, calls = build(tmp_path, monkeypatch)
    source = tmp_path/'clip.mp4'
    source.write_bytes(b'video')
    job = jobs.submit(source,'clip.mp4',preview=False,session='s1')
    assert job['state'] == 'queued' and job['remote'] == 'r'*16
    assert job['preview'] is False and job['session'] == 's1'
    path, filename, headers = calls['upload'][0]
    assert filename == 'clip.mp4' and headers['X-Preview'] == '0'
    assert not source.exists()
    assert (jobs.directory(job['id'])/'clip.mp4').read_bytes() == b'video'
    assert jobs.get(job['id'])['state'] == 'done'


def test_submit_rejects_unknown_suffix(tmp_path, monkeypatch):
    jobs, _ = build(tmp_path, monkeypatch)
    source = tmp_path/'clip.txt'
    source.write_bytes(b'x')
    with pytest.raises(OfflineError):
        jobs.submit(source,'clip.txt')


def test_result_writes_sidecar(tmp_path, monkeypatch):
    jobs, _ = build(tmp_path, monkeypatch)
    source = tmp_path/'clip.mp4'
    source.write_bytes(b'video')
    job = jobs.submit(source,'clip.mp4',session='s1')
    payload = jobs.result(job['id'])
    assert payload['segments'][0]['activity'] == 'walking'
    labels = json.loads((tmp_path/'records'/'s1'/'offline'/'labels.json').read_text(encoding='utf-8'))
    assert labels['job'] == job['id'] and labels['segments'][0]['activity_zh'] == '走路'
    assert 'unknown_field' not in labels['segments'][0]


def test_review_needs_preview_product(tmp_path, monkeypatch):
    jobs, calls = build(tmp_path, monkeypatch)
    source = tmp_path/'clip.mp4'
    source.write_bytes(b'video')
    job = jobs.submit(source,'clip.mp4')
    calls['files']['review.mp4'] = b'fake-mp4'
    path = jobs.review(job['id'])
    assert path.read_bytes() == b'fake-mp4'
    assert path == jobs.review(job['id'])


def test_archive_packs_products(tmp_path, monkeypatch):
    jobs, calls = build(tmp_path, monkeypatch)
    source = tmp_path/'clip.mp4'
    source.write_bytes(b'video')
    job = jobs.submit(source,'clip.mp4')
    calls['files']['review.mp4'] = b'fake-mp4'
    calls['files']['activities.csv'] = b'a,b\n'
    archive = jobs.archive(job['id'])
    with zipfile.ZipFile(archive) as bundle:
        assert sorted(bundle.namelist()) == ['activities.csv','labels.json','review.mp4']


def test_cancel_forwards_to_remote(tmp_path, monkeypatch):
    jobs, calls = build(tmp_path, monkeypatch)
    source = tmp_path/'clip.mp4'
    source.write_bytes(b'video')
    job = jobs.submit(source,'clip.mp4')
    assert jobs.cancel(job['id'])['state'] == 'cancelled'
    assert calls['post'] == ['/jobs/'+'r'*16+'/cancel']


def test_api_submit_requires_token_and_video(tmp_path, monkeypatch):
    monkeypatch.setattr(OfflineJobs,'_upload',lambda self,path,filename,headers: {'id':'r'*16,'state':'queued'})
    monkeypatch.setattr(OfflineJobs,'_get',lambda self,path,timeout=None,raw=False: {'ok':True,'pkg':'pkg','busy':False})
    with TestClient(create_app(tmp_path,demo=True)) as client:
        token = client.get('/api/config').json()['control_token']
        assert client.get('/api/offline/config').json()['reachable'] is True
        body = b'not-a-real-video'
        assert client.post('/api/offline/jobs',content=body,headers={'X-Filename':'a.mp4'}).status_code == 403
        job = client.post('/api/offline/jobs',content=body,
                          headers={'X-Walkpad-Token':token,'X-Filename':'a.mp4','X-Preview':'0'})
        assert job.status_code == 200 and job.json()['state'] == 'queued'
        assert job.json()['preview'] is False
        bad = client.post('/api/offline/jobs',content=body,headers={'X-Walkpad-Token':token,'X-Filename':'a.txt'})
        assert bad.status_code == 400


def test_upload_sends_percent_encoded_filename(tmp_path):
    """头部只允许 ISO-8859-1：中文名不编码会在 http.client 里直接抛 UnicodeEncodeError。"""
    seen = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            seen['filename'] = self.headers['X-Filename']
            body = b'{"id":"%s","state":"queued"}' % (b'r'*16)
            self.send_response(202)
            self.send_header('Content-Length',str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = http.server.HTTPServer(('127.0.0.1',0),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        jobs = OfflineJobs(tmp_path/'offline',tmp_path/'records',url='http://127.0.0.1:%d' % server.server_port)
        source = tmp_path/'clip.mp4'
        source.write_bytes(b'video')
        job = jobs.submit(source,'走路 测试.mp4')
        assert job['video'] == '走路 测试.mp4'
        assert seen['filename'] == quote('走路 测试.mp4')
        assert unquote(seen['filename']) == '走路 测试.mp4'
    finally:
        server.shutdown()


def test_api_decodes_percent_encoded_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(OfflineJobs,'_upload',lambda self,path,filename,headers: {'id':'r'*16,'state':'queued'})
    with TestClient(create_app(tmp_path,demo=True)) as client:
        token = client.get('/api/config').json()['control_token']
        response = client.post('/api/offline/jobs',content=b'video',
                               headers={'X-Walkpad-Token':token,'X-Filename':quote('走路 1.mp4')})
        assert response.status_code == 200
        assert response.json()['video'] == '走路 1.mp4'
