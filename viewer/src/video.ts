import {el} from './core/dom';
import {state} from './core/state';
import {paintVideoCursor} from './imu';
import {squatCaptureMeta, squatInput, squatPackage} from './parse';
import {paintSquatCursor} from './squat';
import SquatWorker from './squat-worker?worker&inline';
import {workstationCalibration} from './video-workstation';
import type {CalibrationInput} from './squat-worker';
import type {VideoInfo} from './types';

const verifiedCaptures = new Map<number, [string, number]>([
  [11216967, ['f900cebccd3ce29b99201189527493699dfdf3de99ac07d01b3960a29906d170', 5.358]],
  [20155226, ['e3d882d9e37399ab61d7bdae4ee18db31671fb3e6b4d4ecae740afbd62a45b92', .011]],
]);

export async function verifiedVideoOffset(archive: ArrayBuffer): Promise<number | undefined> {
  const entry = verifiedCaptures.get(archive.byteLength);
  if (!entry || !globalThis.crypto?.subtle) return;
  try {
    const digest = await crypto.subtle.digest('SHA-256', archive);
    const hex = Array.from(new Uint8Array(digest), x => x.toString(16).padStart(2, '0')).join('');
    if (hex === entry[0]) return entry[1];
  } catch { /* 哈希不可用时不按文件名或大小套用偏移。 */ }
}

let calibrationJob: AbortController;
const cancelCalibration = () => calibrationJob?.abort();
function startCalibration(): AbortController {
  cancelCalibration();
  el.vidAuto.disabled = el.vidFirstBottom.disabled = true;
  return calibrationJob = new AbortController();
}
function finishCalibration(job: AbortController): void {
  job.abort();
  if (calibrationJob !== job) return;
  calibrationJob = undefined;
  el.vidAuto.disabled = el.vidFirstBottom.disabled = false;
}

const MIME: Record<string, string> = {
  mp4: 'video/mp4', m4v: 'video/mp4', mov: 'video/quicktime', webm: 'video/webm'
};

const s2 = (v: number): string => Number.isFinite(v) ? v.toFixed(2) : '—';

// 会话末端：视频起点推定要拿它当参照，取所有流里最晚的末端
const sessionDur = (): number => state.sides.length ? Math.max(...state.sides.map(s => s.dur)) : 0;

export function attachVideo(u8: Uint8Array, name: string, total = 1, verifiedOffset?: number,
    packOffset?: number, packUncertaintyUs?: number): void {
  clearVideo();
  const ext = (name.split('.').pop() || '').toLowerCase();
  // zip 里的字节是整包 buffer 上的视图，Blob 要独占的 ArrayBuffer 才收
  const ab = u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength) as ArrayBuffer;
  const blob = new Blob([ab], {type: MIME[ext] || 'video/mp4'});
  const url = URL.createObjectURL(blob);
  // 包内落地对时优先于按文件名硬编码的旧校准表：前者来自这次采集的时钟，后者是补丁
  const packed = typeof packOffset === 'number' && Number.isFinite(packOffset) ? packOffset : null;
  state.video = {url, blob, name, bytes: u8.byteLength,
    offset: packed ?? verifiedOffset ?? 0,
    offsetSource: packed != null ? 'pack' : verifiedOffset == null ? 'estimate' : 'verified',
    verifiedOffset, packUncertaintyUs: packed != null ? packUncertaintyUs : undefined, dur: NaN, total};
  if (el.vidToggle) el.vidToggle.hidden = false;
  if (el.vid) el.vid.src = url;
  if (el.vidPanel) el.vidPanel.hidden = false;
  renderVideo();
}

// 换文件时必须撤销旧 URL：blob 不撤销就一直占着内存，几个大包之后页面会明显变卡
export function clearVideo(): void {
  const old = state.video;
  if (old) URL.revokeObjectURL(old.url);
  state.video = null;
  if (calibrationJob) finishCalibration(calibrationJob);
  seekStatus('');
  stopCursorLoop();
  if (el.vidToggle) el.vidToggle.hidden = true;
  if (el.vidPanel) el.vidPanel.hidden = true;
  if (el.vid) { el.vid.pause(); el.vid.removeAttribute('src'); el.vid.load(); }
  syncTime();
}

/** 起点推定：录制在停止时收尾，视频结束大致就是会话结束，据此反推视频 0 s 落在哪 */
function guessOffset(): number {
  const vid = state.video;
  if (!vid) return 0;
  const D = sessionDur();
  if (!Number.isFinite(vid.dur) || !(D > 0)) return 0;
  return Math.max(0, Math.round((D - vid.dur) * 100) / 100);
}

function syncTime(): void {
  const vid = state.video;
  const t = vid && el.vid?.readyState >= 1 && !el.vid.error && !el.vidPanel?.hidden ? el.vid.currentTime + vid.offset : null;
  if (el.vidNow) el.vidNow.textContent = nowText();
  const changed = state.videoT !== t;
  state.videoT = t;
  if (changed) paintVideoCursor();
  paintSquatCursor();
}

let cursorFrame = 0;
function stopCursorLoop(): void {
  if (cursorFrame) cancelAnimationFrame(cursorFrame);
  cursorFrame = 0;
}
function startCursorLoop(): void {
  if (cursorFrame || state.tab !== 'squat' || !state.video || el.vidPanel.hidden || el.vid.paused || el.vid.ended) return;
  cursorFrame = requestAnimationFrame(() => {
    cursorFrame = 0;
    syncTime();
    startCursorLoop();
  });
}
export function syncVideoTab(): void {
  if (state.tab === 'squat') startCursorLoop();
  else stopCursorLoop();
}

function nowText(): string {
  const vid = state.video;
  if (!vid || !el.vid) return '';
  const t = el.vid.currentTime || 0;
  return `视频 ${s2(t)} s → 会话 ${s2(t + vid.offset)} s`;
}

function renderVideo(): void {
  const vid = state.video;
  if (!vid || !el.vidPanel) return;
  el.vidName.textContent = `${vid.name} · ${s2(vid.dur)} s`
    + (vid.total > 1 ? `（包内共 ${vid.total} 段，只载入第一段）` : '');
  if (document.activeElement !== el.vidOff) el.vidOff.value = vid.offset.toFixed(3);
  el.vidNote.textContent = {
    estimate: '未校准 · 起点仅供定位' + (squatPackage(state.sides) ? '；可按普通深蹲动作自动校准' : ''),
    pack: '已用包内落地对时' + (Number.isFinite(vid.packUncertaintyUs)
      ? ` · 视频与设备同一会话轴 ±${Math.max(1, Math.round((vid.packUncertaintyUs ?? 0) / 1000))} ms`
      : ' · 视频与设备同一会话轴'),
    verified: '已应用此采集包的落地校准',
    firstBottom: '已按首次蹲底手动对时 · 单点估计，未校正漂移',
    motion: '已由工作站 GPU 估计起点 · 非硬件对时',
    manual: '起点为手动给定',
  }[vid.offsetSource];
  el.vidNote.title = '会话时刻 = 视频时刻 + 起点；「跳到会话」只接受视频覆盖到的区间，越界会提示而不会贴边。';
  syncTime();
}

function setOffset(v: number, source: VideoInfo['offsetSource'] = 'manual'): void {
  const vid = state.video;
  if (!vid || !Number.isFinite(v)) return;
  vid.offset = v;
  vid.offsetSource = source;
  cancelCalibration();
  seekStatus('');
  renderVideo();
}

function runFirstBottom(input: CalibrationInput, signal: AbortSignal): Promise<number> {
  const worker = new SquatWorker();
  return new Promise<number>((ok, fail) => {
    signal.throwIfAborted();
    const end = (next:()=>void) => { clearTimeout(timer); signal.removeEventListener('abort',abort); next(); };
    const abort = () => end(()=>fail(signal.reason));
    const timer = setTimeout(()=>end(()=>fail(new Error('双踝对时计算超时。'))),12000);
    signal.addEventListener('abort',abort,{once:true});
    worker.onmessage = ({data}) => end(()=>{
      if (!Number.isFinite(data.firstBottom)) fail(new Error(data.error || '后台对时没有返回结果。'));
      else ok(data.firstBottom);
    });
    worker.onerror = e => {e.preventDefault(); end(()=>fail(new Error('后台对时无法运行。')));};
    try { worker.postMessage(input,input.inputs.flatMap(s=>[s.t.buffer,s.receive.buffer,s.elapsed.buffer,s.values.buffer])); }
    catch (e) { end(()=>fail(e)); }
  }).finally(()=>worker.terminate());
}

async function calibrateMotion(): Promise<void> {
  const vid = state.video, sides = state.sides, files = squatPackage(sides);
  if (!vid || !Number.isFinite(vid.dur)) {seekStatus('请等视频加载完毕。'); return;}
  if (vid.dur < 5 || vid.dur > 120) {seekStatus('工作站自动校准支持 5–120 秒的视频。'); return;}
  if (!files) {seekStatus('自动校准需要同包的左右踝 IMU；不会使用鞋垫 IMU 替代。'); return;}
  const controller = startCalibration();
  const alive = () => !controller.signal.aborted && state.video===vid && state.sides===sides;
  seekStatus('正在准备工作站 GPU 校准…');
  try {
    const inputs: CalibrationInput['inputs'] = [await squatInput(files[0], alive), await squatInput(files[1], alive)];
    if (!alive()) return;
    seekStatus('正在加密传输数据并等待工作站 GPU 校准…');
    const timeout = AbortSignal.timeout(120000);
    const fit = await workstationCalibration(vid.blob,sides[0].t0,inputs,AbortSignal.any([controller.signal,timeout]));
    if (!alive()) return;
    if (fit.status === 'rejected') {
      const {best_candidate_seconds:a, alternative_seconds:b} = fit.diagnostics ?? {};
      const detail = Number.isFinite(a) && Number.isFinite(b) ? `（${a.toFixed(2)} s 与 ${b.toFixed(2)} s）` : '';
      const messages = {
        ambiguous_motion: `工作站 GPU：动作节奏重复，多个偏移都吻合${detail}`,
        gpu_unavailable: '工作站没有可用的校准 GPU，可能正在被占用或初始化失败',
        worker_unavailable: '工作站 GPU 校准失败或超时',
        insufficient_visible_knees_and_ankles: '视频中的膝踝可见性不足',
        invalid_input: '工作站无法读取视频或双踝数据',
      };
      seekStatus(`${Object.hasOwn(messages,fit.reason) ? messages[fit.reason] : '工作站校准证据不足'}，未覆盖原值。`);
      return;
    }
    el.vidOff.blur();
    setOffset(fit.offset_seconds,'motion');
    const service = fit.version ? `（v${fit.version}）` : '';
    seekStatus(`工作站 GPU${service} 已估计视频起点 ${fit.offset_seconds.toFixed(3)} s；预测结果不变。`);
  } catch (e) {
    if (alive()) seekStatus(`${e instanceof DOMException && e.name==='TimeoutError' ? '等待工作站 GPU 超时' : e instanceof Error ? e.message : String(e)}（原偏移保留）`);
  } finally {
    finishCalibration(controller);
  }
}

async function calibrateFirstBottom(): Promise<void> {
  const vid = state.video, v = el.vid;
  if (!vid) return;
  if (!Number.isFinite(vid.dur) || v.readyState < 2 || v.seeking || v.error) {
    seekStatus('请等画面加载完成，再暂停在第一次蹲到最低的画面。'); return;
  }
  if (!v.paused || v.ended) { seekStatus('请暂停在第一次蹲到最低的画面，再点首次蹲底对时。'); return; }
  const sides = state.sides, files = squatPackage(sides);
  if (!files) { seekStatus('首次蹲底对时需要同一个采集包的左踝与右踝 IMU。'); return; }
  const frame = v.currentTime, controller = startCalibration();
  const {signal} = controller;
  const alive = () => !signal.aborted && state.video === vid && state.sides === sides;
  const moved = () => {
    if (!alive()) return;
    controller.abort();
    seekStatus('视频位置已变化，原偏移保留。');
  };
  for (const event of ['seeking', 'play', 'error']) v.addEventListener(event, moved, {signal});
  seekStatus('正在从整段双踝数据定位第 1 次完整深蹲…');
  try {
    const inputs: CalibrationInput['inputs'] = [await squatInput(files[0], alive), await squatInput(files[1], alive)];
    if (!alive()) return;
    const t = await runFirstBottom({inputs,origin:sides[0].t0,firstBottom:true,aligned:squatCaptureMeta(sides)?.aligned===true},signal);
    if (!alive()) return;
    if (!v.paused || v.seeking || v.currentTime !== frame) { moved(); return; }
    el.vidOff.blur();
    setOffset(t-frame, 'firstBottom');
    seekStatus(`视频 ${frame.toFixed(3)} s ↔ 双踝第 1 次蹲底中点 ${t.toFixed(3)} s。`);
  } catch (e) {
    if (alive()) seekStatus(`${e instanceof Error ? e.message : String(e)}（原偏移保留）`);
  } finally {
    finishCalibration(controller);
  }
}

function seekStatus(text: string): void {
  if (el.vidSeekStatus) {
    el.vidSeekStatus.hidden = !text;
    el.vidSeekStatus.textContent = text;
  }
}

export function seekVideoSession(t: number, reveal = false): void {
  const vid = state.video;
  if (!vid || !el.vid || !Number.isFinite(t)) return;
  if (reveal) toggleVideoPanel(true);
  if (!Number.isFinite(vid.dur)) {
    seekStatus('视频时间轴尚不可用。');
    return;
  }
  const vt = t - vid.offset;
  if (vt < -1e-8 || vt > vid.dur + 1e-8) {
    seekStatus(`视频只覆盖会话 ${s2(vid.offset)}–${s2(vid.offset + vid.dur)} s。`);
    return;
  }
  seekStatus('');
  el.vid.currentTime = Math.max(0, Math.min(vid.dur, vt));
  syncTime();
}

// 竖屏录制（手机竖着拍）按 340 px 摆只能看到一条窄缝，改按高度上限收窄浮窗，
// 横屏片仍用基准宽度
function fitPanel(): void {
  const v = el.vid;
  if (!el.vidPanel || !v || !v.videoWidth || !v.videoHeight) return;
  const ar = v.videoWidth / v.videoHeight;
  const maxH = window.innerHeight * 0.40;
  const w = ar < 1 ? Math.min(maxH * ar, 340) : 340;
  el.vidPanel.style.width = Math.round(Math.max(240, Math.min(440, w))) + 'px';
}

export function toggleVideoPanel(show?: boolean): void {
  if (!el.vidPanel || !state.video) return;
  el.vidPanel.hidden = show == null ? !el.vidPanel.hidden : !show;
  if (el.vidPanel.hidden) stopCursorLoop();
  else startCursorLoop();
  syncTime();
}

/* 拖动标题栏挪位置：浮窗压在波形上时得能挪开，挪过之后不再回到右下角 */
function bindDrag(): void {
  let drag: {dx: number; dy: number} | null = null;
  el.vidHead.addEventListener('pointerdown', (ev: PointerEvent) => {
    if (ev.button !== 0 || (ev.target as HTMLElement).closest('button')) return;
    const r = el.vidPanel.getBoundingClientRect();
    drag = {dx: ev.clientX - r.left, dy: ev.clientY - r.top};
    el.vidPanel.style.right = 'auto';
    el.vidPanel.style.bottom = 'auto';
    el.vidPanel.style.left = r.left + 'px';
    el.vidPanel.style.top = r.top + 'px';
    el.vidHead.setPointerCapture(ev.pointerId);
  });
  el.vidHead.addEventListener('pointermove', (ev: PointerEvent) => {
    if (!drag) return;
    const w = el.vidPanel.offsetWidth, h = el.vidPanel.offsetHeight;
    el.vidPanel.style.left = Math.max(0, Math.min(window.innerWidth - w, ev.clientX - drag.dx)) + 'px';
    el.vidPanel.style.top = Math.max(0, Math.min(window.innerHeight - h, ev.clientY - drag.dy)) + 'px';
  });
  el.vidHead.addEventListener('pointerup', () => { drag = null; });
  el.vidHead.addEventListener('pointercancel', () => { drag = null; });
}

export function bindVideo(): void {
  if (!el.vidPanel) return;
  bindDrag();
  el.vidToggle.addEventListener('click', () => toggleVideoPanel());
  el.vidClose.addEventListener('click', () => toggleVideoPanel(false));
  el.vid.addEventListener('loadedmetadata', () => {
    const vid = state.video;
    if (!vid) return;
    vid.dur = el.vid.duration;
    if (vid.offsetSource === 'estimate') vid.offset = guessOffset();
    fitPanel();
    renderVideo();
  });
  for (const event of ['timeupdate', 'seeked']) el.vid.addEventListener(event, syncTime);
  el.vid.addEventListener('seeking', () => { if (!calibrationJob) seekStatus(''); syncTime(); });
  el.vid.addEventListener('playing', startCursorLoop);
  for (const event of ['pause', 'ended', 'emptied']) el.vid.addEventListener(event, () => { stopCursorLoop(); syncTime(); });
  el.vid.addEventListener('error', () => { stopCursorLoop(); syncTime(); seekStatus('视频无法解码或读取。'); });
  el.vidOff.addEventListener('input', () => {
    const v = parseFloat(el.vidOff.value);
    if (Number.isFinite(v)) setOffset(v);
  });
  el.vidAuto.addEventListener('click', () => {
    const vid = state.video;
    if (!vid) return;
    // 包内起点是硬件时钟算出来的，再跑一次动作匹配只会把它换差
    if (vid.offsetSource === 'pack') { seekStatus('这份包已带落地对时，无需校准。'); return; }
    if (vid.verifiedOffset != null) setOffset(vid.verifiedOffset, 'verified');
    else void calibrateMotion();
  });
  el.vidFirstBottom.addEventListener('click', () => void calibrateFirstBottom());
  el.vidJump.addEventListener('keydown', (ev: KeyboardEvent) => {
    if (ev.key !== 'Enter') return;
    ev.preventDefault();
    seekVideoSession(parseFloat(el.vidJump.value));
  });
}
