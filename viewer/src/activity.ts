import b1q from './b1q-model.json';
import ActivityWorker from './activity-worker?worker&inline';
import type {ActivitySegment, GaitSegment} from './gait/core';
import {reportSummary} from './gait/walking';
import type {GaitReportSegment, ReportSummary} from './gait/walking';
import {gaitBadge, gaitFormat, initGaitDetail, phaseValue, renderGaitDetail, renderGaitOverview} from './gait/view';
import {IC_TO_LABELS} from './gait/running-icto';
import type {IctoVersion} from './gait/running-icto';
import {ACTIVITY4, ACTIVITY4_ANNOTATED, ACTIVITY_STILL, ACTIVITY_TURN, ACTIVITY_TURN_L, ACTIVITY_TURN_R, SEG_C, SEG_CN, SIDE_CN} from './core/constants';
import {state} from './core/state';
import {activityInput} from './parse';
import {applyWin, winOf} from './timewin';
import {setTab} from './app';
import {WALK_REVIEW} from './turn-rule';

const names = SEG_CN;
const colors = SEG_C;
const MODELS = [
  {label: 'B1Q 实测候选', name: 'B1Q（默认）', contract: b1q as any},
  {label: 'Moticon 多域时序模型', name: 'Moticon 多域时序模型（新版试用）', contract: {id: 'WT-offline-Moticon-TAS-v1', kind: 'moticon-tas'} as any},
  {label: 'GaitWindow 优化版', name: 'GaitWindow 优化版（对照）', contract: {id: 'WT-offline-GaitWindow-Refined', kind: 'gait-window', reviewGait: true} as any},
  {label: '正常走跑提取', name: '正常走跑提取（TAS + 步态复核 + 转向）', contract: {id: 'WT-gait-extraction-v1', kind: 'gait-extraction'} as any},
  {label: '50 Hz 双脚 IMU 四分类', name: '50 Hz 双脚 IMU 四分类（试用）', contract: ACTIVITY4},
  {label: '50 Hz 四分类 + 步态复核 + 转向', name: '50 Hz 四分类 + 步态复核 + 转向（试用）', contract: ACTIVITY4_ANNOTATED}
];
const isAnnotated = () => MODELS[modelIndex].contract.kind === ACTIVITY4_ANNOTATED.kind;
const modelStill = () => isAnnotated() || ['moticon-tas', ACTIVITY4.kind].includes(MODELS[modelIndex].contract.kind);
const isExtraction = () => MODELS[modelIndex].contract.kind === 'gait-extraction';
const stateName = (cls: number) => cls === ACTIVITY_STILL && modelStill() ? '静止' : names[cls];
const contextSeconds = (c) => ((2 * c.radius + 1) * .1 + .4).toFixed(1);
let source = null, worker: Worker = null, result = null, running = false, token = 0, selected = -1, page = 0, modelIndex = 0;
let segments: ActivitySegment[] = [], gait: GaitSegment[] = [], runningGait: GaitSegment[] = null;
let report: GaitReportSegment[] = [], refinedRunReport: GaitReportSegment[] = null;
let baseStats: ReportSummary[] = [], runStats: ReportSummary[] = [], baseOverview: ReportSummary[] = [], runOverview: ReportSummary = null;
let ictoVersion: IctoVersion = 'legacy', runningError = false, gaitNote = '', status = '';
const usesRefined = (i: number) => ictoVersion === 'running' && !!refinedRunReport?.[i];
const showDetail = (i: number) => renderGaitDetail(usesRefined(i) ? runningGait?.[i] : gait[i],
  usesRefined(i) ? refinedRunReport[i] : report[i], segments[i], i, usesRefined(i), state.receiptTiming);
function turnQuality(s: ActivitySegment) {
  const run = (result?.turnRuns ?? []).find(r => r.a <= s.a + .05 && r.b >= s.b - .05);
  if (!run) return '转身判据命中';
  const angle = Math.abs(run.angle), span = run.b - run.a;
  const feet = run.dir === 0 ? `双脚方向证据不足（${run.left.toFixed(0)}° / ${run.right.toFixed(0)}°）`
    : `双脚同向（${run.left.toFixed(0)}° / ${run.right.toFixed(0)}°）`;
  const detail = isAnnotated() ? `转角约 ${angle.toFixed(0)}° · 转向 ${span.toFixed(1)} s` : `同向 ${angle.toFixed(0)}° / ${span.toFixed(1)} s`;
  return `${detail} · ${feet}${angle >= 300 && span >= 6 ? ' · 已达转圈档' : ''}`;
}
function walkReviewReasons(mask = 0) {
  return [mask & WALK_REVIEW.lowSwing ? '摆动偏低' : '', mask & WALK_REVIEW.pause ? '双脚摆动中断' : '',
    mask & WALK_REVIEW.short ? '连续步行段过短' : ''].filter(Boolean);
}
const summaryClasses = (totals: number[], still: boolean) => isExtraction()
  ? [0, 1, 2, ACTIVITY_TURN_L, ACTIVITY_TURN_R, ...(totals[ACTIVITY_TURN] > 0 ? [ACTIVITY_TURN] : []), 3]
  : still ? [0, 1, ACTIVITY_STILL, 2, 3] : [0, 1, 2, 3];
const probClasses = () => isExtraction() ? [0, 1, 2, ACTIVITY_STILL, ACTIVITY_TURN]
  : modelStill() ? [0, 1, 2, ACTIVITY_STILL] : [0, 1, 2];
// 分析完成（或失败/取消）时兑现，IMU 页的「分析整段」靠它知道什么时候该刷新底色
let pending: (() => void) | null = null;
const listeners: Array<() => void> = [];
function finish() { const f = pending; pending = null; f?.(); }
const $ = (id) => document.getElementById(id) as any;
const hint = () => `导入双脚数据后，点击「分析整段」运行本地 ${MODELS[modelIndex].label}。`;

function clearResult() {
  result = null; segments = []; gait = []; runningGait = null; report = []; refinedRunReport = null; runningError = false;
  baseStats = []; runStats = []; baseOverview = []; runOverview = null;
  gaitNote = ''; selected = -1; page = 0; state.gaitFocus = null;
}
function cancel(message = '已取消，可重新分析。') {
  token++; worker?.terminate(); worker = null; running = false; status = message; finish(); renderActivity();
}
function reset() {
  token++; worker?.terminate(); worker = null; running = false; clearResult();
  source = state.sides; status = hint(); finish();
}
function start(): Promise<void> {
  if (running) return Promise.resolve();
  return new Promise<void>(resolve => { pending = resolve; void run(); });
}
async function run() {
  // 对齐 status 的归属，但不走 reset()：那里会兑现 pending，让 IMU 的 await 提前返回
  if (source !== state.sides) { token++; worker?.terminate(); worker = null; source = state.sides; }
  clearResult();
  const contract = MODELS[modelIndex].contract;
  if (state.rebuilt) { status = '上位机 TXT 的写盘重建时间不满足双脚同步模型输入，不能自动推断；请使用 APP 测量包或物理量 parquet。'; finish(); renderActivity(); return; }
  const left = state.sides.filter(s => s.side === 'left'), right = state.sides.filter(s => s.side === 'right');
  if (left.length !== 1 || right.length !== 1) { status = '整段分析必须同时有明确的左脚和右脚 IMU，不能把单脚复制成双脚。'; finish(); renderActivity(); return; }
  const mine = ++token, original = state.sides; running = true; status = '正在读取整段双脚 IMU（不使用画面抽样数据）…'; renderActivity();
  const alive = () => token === mine && state.sides === original;
  try {
    const inputs = [await activityInput(left[0], alive), await activityInput(right[0], alive)];
    if (!alive()) return;
    const duration = Math.max(...original.map(s => s.dur));
    if (!(duration > 0) || duration > 86400) throw new Error('记录时长无效或超过单次 24 小时上限');
    worker = new ActivityWorker();
    worker.onerror = ev => { if (!alive()) return; status = '计算失败：' + ev.message; running = false; worker?.terminate(); worker = null; finish(); renderActivity(); };
    worker.onmessage = ev => {
      if (!alive()) return;
      if (ev.data.error) { status = '分析失败：' + ev.data.error; running = false; worker?.terminate(); worker = null; finish(); }
      else if (ev.data.result) {
        ({result, segments, gait, report, refinedRunReport, runningGait, runningError} = ev.data);
        baseStats = report.map(s => s && reportSummary([s]));
        runStats = refinedRunReport?.map(s => s && reportSummary([s])) ?? [];
        const overview = (cls: number, reports: GaitReportSegment[]) => ({
          ...reportSummary(reports.filter(s => s?.cls === cls)),
          durationMs: segments.reduce((sum, s) => s.cls === cls && !s.excludeFromGait ? sum + Math.round(s.b * 1000) - Math.round(s.a * 1000) : sum, 0)
        });
        baseOverview = [0, 1].map(cls => overview(cls, report));
        runOverview = refinedRunReport ? overview(1, refinedRunReport) : null;
        const failed: string[] = ev.data.failed;
        gaitNote = ev.data.reportError ? '步态指标未完成，状态分段仍可查看。'
          : failed.length ? `${failed.map(s => SIDE_CN[s]).join('、')}步态候选计算未完成。`
          : ev.data.debugError ? '部分跑步候选事件核对未完成。' : '';
        const first = report.find(s => s?.steps.length);
        selected = first?.index ?? segments.findIndex(s => s.cls < 2); page = Math.floor(Math.max(0, selected) / 100);
        running = false; worker?.terminate(); worker = null;
        status = `分析完成 · ${result.modelId} · 有效 ${(result.validTicks / 10).toFixed(1)} s / 全段 ${duration.toFixed(1)} s · ${segments.filter(s => s.cls !== 3).length} 个动作段`;
        finish();
      } else if (ev.data.phase === 'gait') status = '正在计算分段步态指标…';
      else status = `本地离线推断 ${ev.data.progress}% · 可切换其他标签页，原始数据不上传`;
      renderActivity();
    };
    // Transfer copies only: the waveform's original typed arrays remain usable.
    const copies = inputs.map(s => ({t: s.t.slice(), seq: s.seq.slice(), ok: s.ok.slice(), accAtRail: s.accAtRail?.slice(),
      deviceAngles: s.deviceAngles && {pitch: s.deviceAngles.pitch.slice(), sideTilt: s.deviceAngles.sideTilt.slice()},
      axes: s.axes.map(a => a.slice())}));
    worker.postMessage({contract, inputs: copies, duration, receiptTiming: state.receiptTiming}, copies.flatMap(s => [s.t.buffer, s.seq.buffer, s.ok.buffer,
      ...(s.accAtRail ? [s.accAtRail.buffer] : []), ...(s.deviceAngles ? [s.deviceAngles.pitch.buffer, s.deviceAngles.sideTilt.buffer] : []),
      ...s.axes.map(a => a.buffer)]));
  } catch (e) { if (!alive()) return; running = false; status = '分析失败：' + (e?.message || e); finish(); renderActivity(); }
}
export function selectedCopSegment() {
  const s = source === state.sides ? segments[selected] : undefined;
  return s && (s.cls === 0 || s.cls === 1) ? s : null;
}

function choose(i) {
  const s = segments[i]; if (!s) return;
  selected = i; state.gaitFocus = null; applyWin(s.a, s.b); drawActivity(); renderRows();
  $('activityImu').disabled = false;
  $('activityCop').disabled = !state.hasP || !selectedCopSegment();
  showDetail(i);
}
function saveCsv() {
  if (!result) return;
  const refined = MODELS[modelIndex].contract.reviewGait;
  const reviewCol = isAnnotated() ? ',turn,walk_review,gait_eligible'
    : isExtraction() ? ',walk_review' : refined ? ',state_source' : result.reviewed ? ',run_review' : '';
  const rows = ['start_s,end_s,duration_s,state,mean_model_probability,limited_context,model' + reviewCol];
  for (const s of segments) {
    const extra = isAnnotated() ? [s.turn ? names[s.turn] : '', walkReviewReasons(s.walkReview).join(' / '),
      s.cls < 2 ? s.excludeFromGait ? 'no' : 'yes' : '']
      : isExtraction() ? [walkReviewReasons(s.walkReview).join(' / ')]
      : refined ? [s.cls === ACTIVITY_STILL ? 'imu_still' : s.cls === 3 ? 'unclassified' : s.reviewed ? 'local_review' : 'model']
      : result.reviewed ? [s.reviewed ? 'yes' : 'no'] : [];
    rows.push([s.a.toFixed(3), s.b.toFixed(3), (s.b - s.a).toFixed(3), stateName(s.cls), s.confidence?.toFixed(6) ?? '', s.edge ? 'yes' : 'no', result.modelId, ...extra].join(','));
  }
  const url = URL.createObjectURL(new Blob(['\uFEFF' + rows.join('\r\n')], {type: 'text/csv;charset=utf-8'})), a = document.createElement('a'); a.href = url; a.download = state.name.replace(/\.[^.]+$/, '') + '_' + result.modelId + '_状态分段.csv'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function initActivity() {
  if (!$('viewActivity')) return;
  if (!result) status = hint();
  $('activityIctoVersion').value = ictoVersion;
  $('activityRun').onclick = () => void start(); $('activityCancel').onclick = () => cancel(); $('activityCsv').onclick = saveCsv;
  $('activityModel').replaceChildren(...MODELS.map(m => new Option(m.name, m.label)));
  $('activityModel').selectedIndex = modelIndex;
  $('activityModel').onchange = ev => {
    modelIndex = Math.max(0, ev.currentTarget.selectedIndex);
    if (running) cancel(`已中止，模型改为 ${MODELS[modelIndex].label}。`); else { clearResult(); status = hint(); renderActivity(); }
  };
  $('activityIctoVersion').onchange = ev => {
    ictoVersion = ev.currentTarget.value === 'running' ? 'running' : 'legacy';
    state.gaitFocus = null;
    renderActivity();
  };
  $('gaitWaveClear').onclick = () => { state.gaitFocus = null; setTab('imu'); };
  initGaitDetail(({side, ic, to, nextIc, peak, at, icUncertain, nextIcUncertain, note}) => {
    state.gaitFocus = {side, ic, to, nextIc, peak, icUncertain, nextIcUncertain, note,
      algorithm: usesRefined(selected) ? IC_TO_LABELS.running : IC_TO_LABELS.legacy};
    if (Number.isFinite(nextIc)) applyWin(ic - .15, nextIc + .15);
    else applyWin(at - .45, at + .45);
    setTab('imu');
  });
  $('activityImu').onclick = () => { const s = segments[selected]; if (s) { state.gaitFocus = null; applyWin(s.a, s.b); setTab('imu'); } };
  $('activityCop').onclick = () => { if (state.hasP && selectedCopSegment()) setTab('cop'); };
  $('activityProbability').ontoggle = () => { if ($('activityProbability').open) drawActivity(); };
  $('activityPrev').onclick = () => { page = Math.max(0, page - 1); renderRows(); };
  $('activityNext').onclick = () => { page++; renderRows(); };
  $('activityTimeline').onclick = ev => {
    if (!result) return; const rect = ev.currentTarget.getBoundingClientRect(), w = winOf(), t = w.a + (ev.clientX - rect.left - 48) / Math.max(1, rect.width - 64) * (w.b - w.a);
    const index = segments.findIndex(s => s.a <= t && s.b > t); if (index >= 0) { page = Math.floor(index / 100); choose(index); }
  };
}
function renderRows() {
  const body = $('activityRows'); if (!body) return; body.replaceChildren();
  const pages = Math.max(1, Math.ceil(segments.length / 100)); page = Math.min(page, pages - 1);
  $('activityPage').textContent = `${segments.length} 段 · 第 ${page + 1} / ${pages} 页`;
  $('activityPrev').disabled = page === 0; $('activityNext').disabled = page >= pages - 1;
  segments.slice(page * 100, page * 100 + 100).forEach((s, offset) => {
    const i = page * 100 + offset, row = document.createElement('tr'); row.classList.toggle('selected', selected === i);
    const refined = usesRefined(i), w = refined ? runStats[i] : baseStats[i];
    const phase = (refined ? refinedRunReport[i] : report[i])?.phaseTotals;
    const pair = s.cls < 2 && phase ? (s.cls === 0 ? `双脚支撑 ${phaseValue(phase, 'doubleSupportMs')}` : `双脚腾空 ${phaseValue(phase, 'flightMs')}`) : '—';
    const flags = s.edge ? ['含上下文不足的边缘'] : [];
    if (s.cls < 2) flags.push(s.excludeFromGait ? '不纳入常规步态统计'
      : w?.countableMs ? `可计步 ${gaitFormat(w.countableMs / 1000, 2)} s` : '无双脚可计步覆盖');
    if (isAnnotated()) flags.unshift(...(s.turn ? [turnQuality(s)] : []), ...walkReviewReasons(s.walkReview));
    if (s.reviewed) flags.unshift(MODELS[modelIndex].contract.reviewGait ? '含短段复核' : '含双脚周期复核');
    const quality = isAnnotated() && s.turn && s.cls !== 3 ? flags.join(' · ')
      : s.cls >= ACTIVITY_TURN ? turnQuality(s)
      : s.cls === ACTIVITY_STILL && !modelStill() ? '双脚 IMU 局部低运动'
      : s.cls === 3 ? '缺脚 / 缺口 / 时间或物理量无效 / 窗口上下文不足'
      : isExtraction() && s.cls === 2 ? [...flags, ...walkReviewReasons(s.walkReview)].join(' · ') || '未提取为走跑'
      : isExtraction() && s.cls < 2 ? [s.cls === 0 ? '通过步态复核' : 'TAS 跑步候选，未做跑姿复核', ...flags].join(' · ')
      : flags.join(' · ') || '模型推断';
    for (const value of [String(i + 1), s.a.toFixed(2), s.b.toFixed(2), (s.b - s.a).toFixed(2), gaitBadge(s.cls, stateName(s.cls)), s.confidence == null ? '—' : (s.confidence * 100).toFixed(1) + '%', gaitFormat(w?.cadence, 1), gaitFormat(w?.steps, 0), quality, pair]) {
      const td = document.createElement('td'); td.append(value);
      if (s.turn && row.cells.length === 4) td.append(' ', gaitBadge(s.turn, names[s.turn]));
      row.append(td);
    }
    row.tabIndex = 0; row.onclick = () => choose(i); row.onkeydown = e => { if (e.key === 'Enter') { choose(i); body.children[offset]?.focus(); } }; body.append(row);
  });
}
export function renderActivity() {
  if (!$('viewActivity')) return;
  if (source !== state.sides) reset();
  $('activityStatus').textContent = status + (state.receiptTiming ? ' · WT 本地版使用手机接收时钟估计采样时间，双脚传输延迟未知。' : ''); $('activityRun').disabled = running || !state.sides.length; $('activityCancel').hidden = !running;
  $('activityCsv').disabled = !result; $('activityImu').disabled = !segments[selected];
  $('activityCop').disabled = !state.hasP || !selectedCopSegment();
  const experimental = ictoVersion === 'running';
  $('activityIctoHint').textContent = experimental ? '步行沿用原版；跑步按所选 IC/TO 结果重算同一套指标。' + (runningError ? '实验版计算失败，跑步当前显示原版。'
    : result && !refinedRunReport ? '本次没有可供实验版复核的跑步周期，显示原版。' : '未用同步压力或视频验证。') : '走跑共用指标；切换 IC/TO 不改变状态分段。';
  $('activityGaitOverview').hidden = !result;
  const totals = names.map(() => 0); for (const s of segments) totals[s.cls] += s.b - s.a;
  const annotated = isAnnotated();
  const annotations = $('activityAnnotations');
  annotations.hidden = !annotated || !result;
  if (annotated && result) {
    const durationOf = (filter: (s: ActivitySegment) => boolean) => segments.reduce((sum, s) => sum + (filter(s) ? s.b - s.a : 0), 0).toFixed(1);
    annotations.textContent = `转向 ${durationOf(s => !!s.turn)} s（与状态重叠） · 步行复核未通过 ${durationOf(s => !!s.walkReview)} s`;
  }
  renderGaitOverview(result ? [baseOverview[0], experimental && runOverview ? runOverview : baseOverview[1]] : [],
    [annotated ? '本概览仅汇总不带转向标记且通过适用复核的区间；上方状态总时长不受筛选影响。' : '', gaitNote].filter(Boolean).join(' '));
  showDetail(selected);
  const contract = MODELS[modelIndex].contract;
  $('activityStillLegend').hidden = !(contract.reviewGait || modelStill());
  $('activityStillLegend').textContent = '■ ' + stateName(ACTIVITY_STILL);
  $('activityTurnLLegend').hidden = !(isExtraction() || annotated);
  $('activityTurnRLegend').hidden = !(isExtraction() || annotated);
  const classes = summaryClasses(totals, !$('activityStillLegend').hidden);
  $('activitySummary').textContent = result ? classes.map(i => `${stateName(i)} ${totals[i].toFixed(1)} s`).join('　·　')
    : `${contract.id} · 双脚 12 通道 · 50 Hz 输入 · ${contract.kind === 'gait-window'
      ? '2.4 秒窗口 / 每 0.2 秒分类 · 双脚静止判据 + 短段复核'
      : isExtraction() ? 'TAS 找候选 · 双脚摆动复核步行 · 独立检测转向并排除转向区间'
      : annotated ? '四分类 + 活动边界解码 · 复核和转向只用于标记与筛选步态指标'
      : contract.kind === ACTIVITY4.kind ? '10 Hz 四分类与活动边界 · 整段联合解码'
      : modelStill() ? '10 Hz 四类概率 · 双向 TCN + 边界解码'
      : `10 Hz 概率 · 约 ${contextSeconds(contract)} 秒上下文 · 整段联合解码`}`;
  $('activityProbability').querySelector('summary').textContent = isExtraction()
    ? '原始 TAS 概率 + 转向判据强度（玫红；不是最终分段的置信度）'
    : '原始模型概率（不含静止判据或复核）';
  renderRows(); if (state.tab === 'activity') drawActivity();
  $('gaitWaveFocus').hidden = !state.gaitFocus;
  state.seg = {busy: running, status, modelId: MODELS[modelIndex].contract.id, list: segments};
  for (const fn of listeners) fn();
}
/** 分析整段（IMU 页的按钮走这里）；模型沿用分段页当前所选 */
export function analyzeActivity() { return start(); }
/** 结果或进度变化时通知，IMU 底色靠它刷新 */
export function onActivityChange(fn: () => void) { listeners.push(fn); }
export function drawActivity() {
  if (!$('viewActivity') || state.tab !== 'activity') return;
  for (const [id, height, prob] of [['activityTimeline', 112, false], ['activityProb', 220, true]] as const) {
    if (prob && !$('activityProbability').open) continue;
    const cv = $(id) as HTMLCanvasElement, width = Math.max(260, cv.clientWidth), dpr = devicePixelRatio || 1; cv.width = width * dpr; cv.height = height * dpr; cv.style.height = height + 'px';
    const theme = getComputedStyle(document.documentElement), bg = theme.getPropertyValue('--card').trim() || '#ffffff', ink = theme.getPropertyValue('--muted').trim() || '#536075', line = theme.getPropertyValue('--line').trim() || '#dce3ed';
    const ctx = cv.getContext('2d'); ctx.scale(dpr, dpr); ctx.fillStyle = bg; ctx.fillRect(0, 0, width, height); ctx.font = '12px sans-serif';
    const w = winOf(), pw = width - 64, x = t => 48 + (t - w.a) / (w.b - w.a) * pw;
    ctx.strokeStyle = line; ctx.fillStyle = ink;
    for (let k = 0; k <= 5; k++) { const t = w.a + (w.b - w.a) * k / 5, xx = x(t); ctx.beginPath(); ctx.moveTo(xx, 12); ctx.lineTo(xx, height - 27); ctx.stroke(); ctx.fillText(t.toFixed(1) + 's', Math.min(width - 43, xx - 12), height - 8); }
    if (!result) { ctx.fillText('分析整段后显示状态与逐时概率', 60, 48); continue; }
    if (!prob) {
      for (let i = 0; i < segments.length; i++) {
        const s = segments[i]; if (s.b <= w.a || s.a >= w.b) continue;
        const xx = x(Math.max(s.a, w.a)), ww = Math.max(.8, x(Math.min(s.b, w.b)) - xx);
        ctx.fillStyle = colors[s.cls]; ctx.fillRect(xx, 27, ww, 40);
        if (s.turn) { ctx.fillStyle = colors[s.turn]; ctx.fillRect(xx, 19, ww, 5); }
        if (i === selected) { ctx.strokeStyle = '#182b46'; ctx.lineWidth = 3; ctx.strokeRect(xx + 1, 26, Math.max(1, ww - 2), 42); ctx.lineWidth = 1; }
        if (ww > 48) { ctx.fillStyle = '#ffffff'; ctx.fillText(stateName(s.cls), xx + 7, 51); }
      }
    } else {
      const top = 16, ph = height - 48, first = Math.max(0, Math.floor(w.a * 10)), last = Math.min(result.ticks, Math.ceil(w.b * 10));
      ctx.fillStyle = ink; ctx.fillText('1.0', 10, top + 6); ctx.fillText('0.0', 10, top + ph);
      for (const c of probClasses()) {
        ctx.strokeStyle = colors[c]; ctx.lineWidth = 1.5; ctx.beginPath(); let active = false;
        const step = Math.max(1, Math.floor((last - first) / Math.max(1, pw * 2)));
        for (let k = first; k < last; k += step) { const v = result.probs[c * result.ticks + k]; if (!Number.isFinite(v)) { active = false; continue; } const xx = x((k + .5) / 10), yy = top + (1 - v) * ph; if (!active) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy); active = true; } ctx.stroke();
      }
    }
  }
}
