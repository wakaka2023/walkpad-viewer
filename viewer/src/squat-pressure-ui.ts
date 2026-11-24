import PressureWorker from './squat-pressure-worker?worker&inline';
import {SIDE_CN} from './core/constants';
import {copPoint, copTrailPath, squatFootMap} from './squat-pressure-foot-map';
import {summarizeSquatPressure} from './squat-pressure-summary';
import type {SquatRep} from './squat-core';
import {state} from './core/state';
import {squatPressurePackage} from './parse';
import type {SquatPlot} from './squat-angle-chart';
import {drawPressureChart, pressureChartHeight, pressureExtrema, pressureLanes} from './squat-pressure-chart';
import {leftSignalShareAt, pressureAt, relativeSignal} from './squat-pressure-core';
import type {Foot, PressureResult, PressureSource} from './squat-pressure-core';
import type {PressureRequest, PressureResponse} from './squat-pressure-worker';

const $=<T extends HTMLElement=HTMLElement>(id: string)=>document.getElementById(id) as T;
const feet: Foot[]=['left','right'];
const sideName=(foot: Foot)=>SIDE_CN[foot];
const setText=(el: HTMLElement,text: string)=>{if(el.textContent!==text) el.textContent=text;};
const percentage=(v: number)=>Number.isFinite(v) ? `${v.toFixed(1)}%` : '—';
function shareText(leftPct: number) {
  if(!Number.isFinite(leftPct)) return '—';
  const p=Math.round(leftPct*10);
  return `左 ${percentage(p/10)} / 右 ${percentage((1000-p)/10)}`;
}
let capture: typeof state.sides, sources: PressureSource[]=[];
let result: PressureResult=null, worker: Worker=null, loadError='', ran=false;
let lanes: ReturnType<typeof pressureLanes>=[], windowKey='', refresh=()=>{};
let cards: ReturnType<typeof footCard>[]=[];
let reps: readonly SquatRep[]=[], selectedRep=-1, pickRep=(i: number)=>{};

export const hasSquatPressure=()=>lanes.length>0;
const traceFor=(foot: Foot)=>result?.traces.find(t=>t.foot===foot);
const available=(foot: Foot)=>sources.find(s=>s.foot===foot);

function footCard(foot: Foot) {
  const card=document.createElement('article');card.className='squat-pressure-foot';card.dataset.foot=foot;
  const heading=`squat-pressure-foot-title-${foot}`;
  card.setAttribute('aria-labelledby',heading);
  card.innerHTML=`<div class="squat-pressure-figure">
      <h3 id="${heading}">${sideName(foot)} <span>${foot==='left'?'L':'R'}</span></h3>
      <div class="squat-pressure-map" aria-hidden="true"><span>脚尖</span><span>脚跟</span></div>
      <div class="squat-pressure-side-note" aria-hidden="true">${foot==='left'?'内侧 →':'← 内侧'}</div>
    </div>
    <div class="squat-pressure-foot-data"><small class="squat-pressure-mode">相对信号 · D56</small>
      <dl><dt>窗内单点峰值</dt><dd data-slot="windowPeak">—</dd><dt>COP 前后 / 横向</dt><dd data-slot="cop">—</dd>
        <dt>当前单点峰值</dt><dd data-slot="peak">—</dd><dt>当前单脚总信号</dt><dd data-slot="total">—</dd></dl></div>`;
  const {map,path,dot}=squatFootMap(foot), figure=card.querySelector('.squat-pressure-map');
  figure.insertBefore(map,figure.lastElementChild);
  const slot=(key: string)=>card.querySelector<HTMLElement>(`[data-slot="${key}"]`);
  return {card,foot,path,dot,windowPeak:slot('windowPeak'),cop:slot('cop'),peak:slot('peak'),total:slot('total')};
}

export function selectSquatPressureRep(i: number) {
  selectedRep=i;
  const rows=$('squatPressureRepRows').children;
  for(let j=0;j<rows.length;j++) rows[j].classList.toggle('selected',j===i);
}

const coverageText=(coverage: number)=>coverage>0 && coverage<.01 ? '<1%' : `${Math.round(coverage*100)}%`;

function renderRepPressure() {
  const table=$('squatPressureRepTable'),rows=$('squatPressureRepRows'),note=$('squatPressureRepNote');
  const ready=hasSquatPressure() && reps.length>0;
  table.hidden=!ready;
  rows.replaceChildren();
  if(!ready) {setText(note,'识别出完整深蹲后，按每次下降起点到上升结束汇总。');return;}
  setText(note,'每次从开始下降到结束上升。相对峰值和总信号均以整份记录的各自最高值为 100%；点次数定位。覆盖率为有效时长占本次时长的比例。');
  const fragment=document.createDocumentFragment();
  for(const [i,[a,,,b]] of reps.entries()) {
    const stat=summarizeSquatPressure(result,a,b),row=document.createElement('tr');
    const cell=(key: string,text: string,sub?: string)=>{
      const td=row.insertCell();td.dataset.metric=key;td.textContent=text;
      if(sub) {const hint=document.createElement('small');hint.textContent=sub;td.append(hint);}
      return td;
    };
    const button=document.createElement('button');button.type='button';button.className='lnk';button.textContent=`第 ${i+1} 次`;
    button.title='选中该次并定位到开始下降';
    cell('rep','').append(button);
    cell('time',`${a.toFixed(2)}–${b.toFixed(2)}`);
    for(const foot of feet) {
      const side=stat[foot], peak=side?.peakSignalPct;
      cell(foot+'Peak',percentage(peak),Number.isFinite(peak) ? '' : side ? '无有效读数' : '无数据');
    }
    for(const foot of feet) {
      const side=stat[foot];
      const hasTotal=Number.isFinite(side?.meanTotalPct) || Number.isFinite(side?.peakTotalPct);
      cell(foot+'Total',hasTotal ? `${percentage(side.meanTotalPct)} / ${percentage(side.peakTotalPct)}` : '—',
        hasTotal ? `有效覆盖 ${coverageText(side.coverage)}` : side ? '无有效读数' : '无数据');
    }
    cell('share',shareText(stat.leftPct),Number.isFinite(stat.leftPct) ? `配对覆盖 ${coverageText(stat.pairedCoverage)}`
      : stat.left && stat.right ? '无有效配对' : '缺少一侧数据');
    row.classList.toggle('selected',i===selectedRep);
    row.onclick=()=>pickRep(i);
    fragment.append(row);
  }
  rows.append(fragment);
}

function refreshStatus() {
  const status=$('squatPressureStatus');
  $('squatPressureBody').hidden=!hasSquatPressure();
  if(!sources.length) {setText(status,'本包没有布局受支持的 D56 压力鞋垫。');return;}
  if(loadError) {setText(status,'读取鞋垫压力失败：'+loadError);return;}
  if(!ran || worker) {setText(status,'正在本地读取 D56 鞋垫原始帧…');return;}
  if(!hasSquatPressure()) {setText(status,'鞋垫没有完整的 56 通道有效压力帧。');return;}
  const present=feet.filter(available),names=present.map(sideName).join('、');
  const notes=[`D56 · ${names} · 每脚 56 通道`,'百分比基于鞋垫信号，不等于实际承重比例。'];
  if(!result.reference.peakAdc) notes.push('没有大于零的有效信号。');
  if(present.length===1) notes.push('仅有单侧鞋垫，不计算左右占比。');
  else if(!result.share) notes.push('双脚未能有效配对，或配对时没有可用信号。');
  else {
    const timing=feet.map(f=>Math.ceil(Math.max(0,...available(f).timing.map(m=>m.uncertaintyUs))/1000));
    notes.push(`包内估计对时不确定度：左约 ${timing[0]} ms / 右约 ${timing[1]} ms。`);
  }
  setText(status,notes.join('　·　'));
}

function start() {
  worker?.terminate();worker=null;result=null;lanes=[];windowKey='';loadError='';ran=true;
  const done=(response: PressureResponse)=>{
    worker?.terminate();worker=null;
    if(response.error) loadError=response.error;
    else {
      result=response.result;
      lanes=result.traces.some(t=>t.values.peakAdc.some(Number.isFinite)) ? pressureLanes(result) : [];
      const chart=$('squatPressureChart');
      chart.style.height=`${pressureChartHeight(lanes)}px`;
      chart.setAttribute('aria-label',lanes.map(x=>x.title).join('、')+'随时间的变化');
    }
    refreshStatus();renderRepPressure();refresh();
  };
  try {
    const active=new PressureWorker();worker=active;
    active.onmessage=(e: MessageEvent<PressureResponse>)=>{if(worker===active && state.sides===capture) done(e.data);};
    active.onerror=e=>{e.preventDefault();if(worker===active && state.sides===capture) done({error:e.message || '后台计算不可用。'});};
    const request: PressureRequest={sources,originUs:capture[0].t0};
    active.postMessage(request);
  } catch(e) {done({error:e instanceof Error ? e.message : String(e)});}
  refreshStatus();renderRepPressure();
}

export function renderSquatPressure(nextReps: readonly SquatRep[],picked: number) {
  reps=nextReps ?? [];selectedRep=picked;
  if(capture!==state.sides) {
    worker?.terminate();worker=null;capture=state.sides;sources=squatPressurePackage(capture) ?? [];
    result=null;lanes=[];loadError='';ran=false;windowKey='';
    cards=feet.filter(available).map(footCard);$('squatPressureFeet').replaceChildren(...cards.map(c=>c.card));
  }
  if(state.tab==='squat' && sources.length && !ran) start();
  else {refreshStatus();renderRepPressure();}
}

export function drawSquatPressure(chart: HTMLCanvasElement,p: SquatPlot,clips: readonly [number,number,number][],colors: readonly string[]) {
  if(!hasSquatPressure()) return;
  drawPressureChart(chart,p,lanes,clips,colors);
  const key=`${p.a}:${p.b}`;
  if(key===windowKey) return;
  windowKey=key;
  for(const c of cards) {
    const trace=traceFor(c.foot),max=trace && pressureExtrema(trace.values.peakAdc,trace.t,p.a,p.b).max;
    setText(c.windowPeak,percentage(relativeSignal(max,result.reference.peakAdc)));
  }
}

export function paintSquatPressure(time: number|null,trailStart: number|null,repIndex: number) {
  if(!hasSquatPressure()) return;
  const left=traceFor('left'),right=traceFor('right');
  const pct=time!=null && result.share ? leftSignalShareAt(left,right,time) : NaN;
  const scope=repIndex>=0 ? ` · 第 ${repIndex+1} 次 COP` : trailStart==null ? '' : ' · COP 最近 1 s';
  setText($('squatPressureTime'),time==null ? '悬停或点击曲线查看 COP；视频播放时自动跟随。' : `会话 ${time.toFixed(2)} s${scope}`);
  setText($('squatPressureRatio'),`左右信号占比：${shareText(pct)}`);
  for(const c of cards) {
    const trace=traceFor(c.foot),at=(key: Parameters<typeof pressureAt>[1])=>time!=null && trace ? pressureAt(trace,key,time) : NaN;
    const ap=at('ap'),ml=at('ml'),cop=Number.isFinite(ap) && Number.isFinite(ml);
    setText(c.cop,cop ? `${ap.toFixed(0)}% / ${ml.toFixed(0)}%` : '—');
    setText(c.peak,percentage(relativeSignal(at('peakAdc'),result.reference.peakAdc)));
    setText(c.total,percentage(relativeSignal(at('sumAdc'),result.reference.sumAdc)));
    const d=cop && trailStart!=null ? copTrailPath(trace,trailStart,time,[ap,ml]) : '';
    if((c.path.getAttribute('d') ?? '')!==d) c.path.setAttribute('d',d);
    if(cop) {const [x,y]=copPoint(ap,ml);c.dot.setAttribute('cx',x);c.dot.setAttribute('cy',y);}
    c.dot.toggleAttribute('hidden',!cop);
  }
}

export function initSquatPressure(draw: ()=>void,onPickRep: (i: number)=>void) {
  refresh=draw;pickRep=onPickRep;
}
