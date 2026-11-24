"""离线动作标注：把视频转发给工作站上的离线包服务，取回分段与标注视频。"""
import http.client
import json
import os
import shutil
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path
from urllib.parse import quote, urlsplit

DEFAULT_URL = 'http://192.168.100.29:18844'
ACTIVE = ('uploading','queued','running')
SUFFIXES = ('.mp4','.mov','.mkv','.avi','.m4v','.webm')


class OfflineError(RuntimeError):
    pass


class OfflineJobs:
    def __init__(self, root, records, url=None, token=None, timeout=30):
        self.root = Path(root)
        self.root.mkdir(parents=True,exist_ok=True)
        self.records = Path(records)
        self.url = (url or os.environ.get('WALKPAD_OFFLINE_URL') or DEFAULT_URL).rstrip('/')
        self.token = token if token is not None else os.environ.get('WALKPAD_OFFLINE_TOKEN','')
        self.timeout = timeout

    def directory(self, job_id):
        if not job_id.isalnum() or len(job_id) != 16:
            raise OfflineError('任务号无效')
        return self.root/job_id

    def _record_path(self, job_id):
        return self.directory(job_id)/'job.json'

    def _save(self, job):
        path = self._record_path(job['id'])
        temp = path.with_suffix('.pending')
        temp.write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding='utf-8')
        temp.replace(path)

    def _headers(self, extra=None):
        headers = dict(extra or {})
        if self.token:
            headers['X-Token'] = self.token
        return headers

    def _get(self, path, timeout=None, raw=False):
        request = urllib.request.Request(self.url+path,headers=self._headers())
        try:
            with urllib.request.urlopen(request,timeout=timeout or self.timeout) as response:
                return response.read() if raw else json.load(response)
        except urllib.error.HTTPError as exc:
            raise OfflineError(self._message(exc))
        except (urllib.error.URLError,OSError) as exc:
            raise OfflineError('离线服务不可达：%s' % exc)

    @staticmethod
    def _message(exc):
        try:
            return json.loads(exc.read().decode('utf-8','ignore')).get('error') or str(exc)
        except Exception:
            return str(exc)

    def _post(self, path, data=b''):
        request = urllib.request.Request(self.url+path,data=data,method='POST',headers=self._headers())
        try:
            with urllib.request.urlopen(request,timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise OfflineError(self._message(exc))
        except (urllib.error.URLError,OSError) as exc:
            raise OfflineError('离线服务不可达：%s' % exc)

    def _upload(self, path, filename, headers):
        parts = urlsplit(self.url)
        conn = http.client.HTTPConnection(parts.hostname,parts.port or 80,timeout=1800)
        data = dict(headers, **{'Content-Type':'application/octet-stream','X-Filename':quote(filename),
                                'Content-Length':str(os.path.getsize(path))})
        for key,value in self._headers().items():
            data[key] = value
        try:
            with open(path,'rb') as body:
                conn.request('POST','/jobs',body=body,headers=data)
                response = conn.getresponse()
                payload = response.read()
            if response.status >= 400:
                raise OfflineError(payload.decode('utf-8','ignore')[:200] or '离线服务拒绝上传')
            return json.loads(payload)
        except (OSError,http.client.HTTPException) as exc:
            raise OfflineError('离线服务不可达：%s' % exc)
        finally:
            conn.close()

    def config(self):
        info = {'url':self.url,'reachable':False,'pkg':None,'busy':None,'error':None}
        try:
            health = self._get('/health',timeout=5)
            info.update(reachable=True,pkg=health.get('pkg'),busy=health.get('busy'),
                        running=health.get('running'),queued=health.get('queued'))
        except OfflineError as exc:
            info['error'] = str(exc)
        return info

    def submit(self, path, filename, preview=True, session=None):
        name = os.path.basename(filename or 'video.mp4')
        if Path(name).suffix.lower() not in SUFFIXES:
            raise OfflineError('只支持 mp4 / mov / mkv / avi / m4v / webm 视频')
        job_id = uuid.uuid4().hex[:16]
        directory = self.directory(job_id)
        directory.mkdir(parents=True,exist_ok=True)
        target = directory/name
        shutil.move(str(path),target)
        job = dict(id=job_id,remote=None,video=name,bytes=target.stat().st_size,preview=bool(preview),
                   session=session,created=time.time(),state='uploading',error=None)
        self._save(job)
        try:
            remote = self._upload(target,name,{'X-Preview':'1' if preview else '0'})
        except OfflineError as exc:
            job.update(state='failed',error=str(exc))
            self._save(job)
            raise
        job.update(remote=remote.get('id'),state=remote.get('state') or 'queued')
        self._save(job)
        return job

    def get(self, job_id):
        job = self._load(job_id)
        if job.get('remote') and job['state'] in ACTIVE:
            try:
                remote = self._get('/jobs/'+job['remote'])
            except OfflineError as exc:
                job['remote_error'] = str(exc)
            else:
                job['remote_error'] = None
                job.update(state=remote.get('state'),elapsed_s=remote.get('elapsed_s'),gpu=remote.get('gpu'),
                           log_tail=remote.get('log_tail'),error=remote.get('error') or job.get('error'))
                self._save(job)
                if job['state'] not in ACTIVE:
                    # 工作站已留有副本，本地这份只在提交阶段有用
                    source = self.directory(job_id)/job['video']
                    try:
                        source.unlink()
                    except OSError:
                        pass
        return job

    def _load(self, job_id):
        path = self._record_path(job_id)
        if not path.is_file():
            raise OfflineError('任务不存在')
        return json.loads(path.read_text(encoding='utf-8'))

    def list(self):
        jobs = []
        for path in sorted(self.root.glob('*/job.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:20]:
            try:
                jobs.append(json.loads(path.read_text(encoding='utf-8')))
            except (OSError,ValueError):
                continue
        return jobs

    def result(self, job_id):
        job = self.get(job_id)
        if not job.get('remote'):
            raise OfflineError(job.get('error') or '任务尚未提交成功')
        payload = self._get('/jobs/%s/result' % job['remote'])
        job['result'] = {key:payload.get(key) for key in ('version','action_rules','fps','frame_count','people','segments','products')}
        job['result_at'] = time.time()
        self._save(job)
        if job.get('session'):
            self._write_sidecar(job,payload)
        return payload

    def _write_sidecar(self, job, payload):
        """旁挂到记录目录：不改原包，查看器后续可直接读这份分段。"""
        directory = (self.records/job['session']/('offline')).resolve()
        if self.records.resolve() not in directory.parents or not (self.records/job['session']).is_dir():
            return
        directory.mkdir(parents=True,exist_ok=True)
        keep = ('track_id','track_fragment','segment_id','activity','activity_zh','start_video_s','end_video_s',
                'duration_s','evidence_level','repetition_count','reason')
        labels = dict(job=job['id'],video=job['video'],version=payload.get('version'),
                      action_rules=payload.get('action_rules'),fps=payload.get('fps'),
                      frame_count=payload.get('frame_count'),created=time.time(),
                      segments=[{key:row.get(key) for key in keep} for row in payload.get('segments',[])])
        (directory/'labels.json').write_text(json.dumps(labels,ensure_ascii=False,indent=2),encoding='utf-8')

    def product(self, job_id, name):
        job = self._load(job_id)
        if not job.get('remote'):
            raise OfflineError('任务尚未提交成功')
        data = self._get('/jobs/%s/file?name=%s' % (job['remote'],quote(name)),timeout=300,raw=True)
        return data

    def review(self, job_id):
        """标注视频较大，只在需要播放时取一次，之后用本地副本支持拖动。"""
        job = self._load(job_id)
        target = self.directory(job_id)/'review.mp4'
        if target.is_file() and target.stat().st_size:
            return target
        if not job.get('result'):
            self.result(job_id)
            job = self._load(job_id)
        if 'review.mp4' not in ((job.get('result') or {}).get('products') or []):
            raise OfflineError('该任务没有生成标注视频')
        data = self.product(job_id,'review.mp4')
        temp = target.with_suffix('.pending')
        temp.write_bytes(data)
        temp.replace(target)
        return target

    def archive(self, job_id):
        job = self._load(job_id)
        target = self.directory(job_id)/(job['video'].rsplit('.',1)[0]+'-offline.zip')
        if target.is_file():
            return target
        products = (job.get('result') or {}).get('products') or []
        if not products:
            self.result(job_id)
            products = (self._load(job_id).get('result') or {}).get('products') or []
        if not products:
            raise OfflineError('该任务没有可用产物')
        cached = self.directory(job_id)/'products'
        cached.mkdir(exist_ok=True)
        temp = target.with_suffix('.pending')
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_STORED) as bundle:
            for name in products:
                path = cached/name
                if not path.is_file():
                    path.write_bytes(self.product(job_id,name))
                bundle.write(path,name)
            bundle.writestr('labels.json',(self.directory(job_id)/'job.json').read_text(encoding='utf-8'))
        temp.replace(target)
        return target

    def cancel(self, job_id):
        job = self._load(job_id)
        if job.get('remote'):
            remote = self._post('/jobs/%s/cancel' % job['remote'])
            job.update(state=remote.get('state'),cancel=True)
        else:
            job.update(state='cancelled')
        self._save(job)
        return job
