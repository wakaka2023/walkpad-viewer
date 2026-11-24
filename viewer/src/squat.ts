import SquatWorker from './squat-worker?worker&inline';
import type {AnkleStream, SquatResult} from './squat-core';
import type {SideAngles} from './squat-axis';
import type {RepFeatures} from './squat-score-features';
import {initSquatMetrics, renderSquatMetrics, resetSquatMetrics, selectSquatMetrics, setSquatMetrics} from './squat-score-ui';
import {version as modelVersion} from './squat-v03-model.json';
import {state} from './core/state';
import {squatCaptureMeta, squatInput, squatPackage} from './parse';
import {applyWin, winOf} from './timewin';
import {seekVideoSession} from './video';
import {drawSquatAngleChart, drawSquatTimeLabels, squatAngleLabels, squatTicks, squatTraceAt, squatX} from './squat-angle-chart';

const HZ=50, colors=['#74849a','#526cdb','#b87616','#178477'];
const labels=['非深蹲','下降','蹲底','上升'];
const events=['开始下降','进入蹲底','开始上升','结束上升'];
const $=<T extends HTMLElement=HTMLElement>(id:string)=>document.getElementById(id) as T;
let source: typeof state.sides, worker: Worker, result: SquatResult, token=0, selected=-1, busy=false, clips: [number,number,number][]=[];
let angles: SideAngles[] = [];
let message='';
let hoverTime: number|null=null, pickedAngleTime: number|null=null;
const hasAngles=()=>angles.some(side=>side.traces?.length);

function reset(newCapture=false) {
  token++; worker?.terminate(); worker=null; busy=false; result=null; selected=-1; clips=[]; message=''; angles=[];
  hoverTime=pickedAngleTime=null;
  resetSquatMetrics(newCapture);
}

async function run() {
  const files=squatPackage(state.sides);
  if (!files || busy) return;
  const range=$<HTMLSelectElement>('squatScope').value==='window' ? winOf() : null;
  reset(); const mine=token; const alive=()=>mine===token && source===state.sides;
  busy=true; message='正在读取双踝数据…'; renderSquat();
  try {
    const inputs=[await squatInput(files[0],alive),await squatInput(files[1],alive)];
    const meta=squatCaptureMeta(source), w=meta?.wrist;
    const wristMode=w?.family==='walkpadImu' ? 'wrist-comp' : 'wrist-raw';
    let wrist: AnkleStream, wristReadIssue='';
    if(w) {
      try {wrist=await squatInput(w.files,alive,wristMode);}
      catch {wristReadIssue='腕部所需六轴或时间戳不可用；不影响双踝分段。';}
    }
    if (!alive()) return;
    worker=new SquatWorker(); message='正在本地分段并计算逐次指标…'; renderSquat();
    const finish=(value?:SquatResult,error?:string,features?:RepFeatures[],featureIssue?:string,wristIssue?:string,
        repAngles?:SideAngles[])=>{
      if (!alive()) return;
      worker?.terminate(); worker=null; busy=false;
      result=value ?? null;
      angles=repAngles ?? [];
      selected=result?.reps.length ? 0 : -1;
      if (!result) message='预测失败：'+error;
      else {
        const {phase,start,valid}=result;
        for (let i=0,j;i<phase.length;i=j) {
          for (j=i+1;j<phase.length && phase[j]===phase[i];j++);
          clips.push([start+i/HZ,start+j/HZ,phase[i]]);
        }
        message=`预测完成 · ${result.reps.length} 次完整深蹲 · 实际范围 ${start.toFixed(3)}–${(start+(phase.length-1)/HZ).toFixed(3)} s · 可判断 ${(valid/HZ).toFixed(2)} / ${(phase.length/HZ).toFixed(2)} s`;
      }
      setSquatMetrics(result,features,{recordId:meta?.recordId ?? null,
        wrist:w ? {placement:w.placement,family:w.family,gyro:wristMode==='wrist-comp' ? 'compensated' : 'raw'} : undefined},
        [featureIssue,wristReadIssue,wristIssue].filter(Boolean).join(' · '));
      renderSquat();
    };
    worker.onmessage=ev=>finish(ev.data.result,ev.data.error,ev.data.features,ev.data.scoreIssue,ev.data.wristIssue,ev.data.repAngles);
    worker.onerror=ev=>{ev.preventDefault();finish(undefined,ev.message || '后台计算不可用。');};
    const streams=wrist ? [...inputs,wrist] : inputs;
    worker.postMessage({inputs,wrist,origin:source[0].t0,range,aligned:meta?.aligned===true},streams.flatMap(s=>[s.t,s.receive,s.elapsed,s.values,...(s.angles ?? [])].map(a=>a.buffer)));
  } catch(e) {
    if (!alive()) return;
    reset(); message='预测失败：'+(e instanceof Error ? e.message : e); renderSquat();
  }
}

function focus(i:number) {
  selected=i; const r=result?.reps[i]; if (!r) return;
  selectSquatMetrics(i);
  applyWin(r[0]-.20,r[3]+.20); drawSquat();
  $('squatRows').querySelectorAll('tr').forEach((row,j)=>row.classList.toggle('selected',i===j));
}

function geometry(cv:HTMLCanvasElement) {
  const w=cv.clientWidth, h=cv.clientHeight;
  return {w,h,l:46,r:Math.max(47,w-16),border:cv.clientLeft,...winOf()};
}

function placeCursor(plot: ReturnType<typeof geometry>,cursor: HTMLElement,t: number|null,allowEdge: boolean) {
  const {a,b,w}=plot, edge=t==null ? '' : t<a ? 'left' : t>b ? 'right' : '';
  cursor.hidden=!Number.isFinite(t) || !state.sides.length || w<100 || !(b>a) || (!!edge && !allowEdge);
  if (!cursor.hidden) {
    if (edge) cursor.dataset.edge=edge;
    else cursor.removeAttribute('data-edge');
    cursor.style.left=`${plot.border+squatX(plot,Math.min(b,Math.max(a,t)))}px`;
  }
  return edge;
}

export function paintSquatCursor() {
  if (state.tab!=='squat') return;
  const cv=$<HTMLCanvasElement>('squatTimeline'), chart=$<HTMLCanvasElement>('squatAngleChart');
  if (!cv || !chart) return;
  const timelinePlot=geometry(cv), anglePlot=geometry(chart);
  const t=state.videoT, cursor=$('squatVideoCursor'), readout=$('squatVideoTime');
  const edge=placeCursor(timelinePlot,cursor,t,true);
  readout.hidden=cursor.hidden;
  if (!cursor.hidden) {
    const j=result ? Math.floor((t-result.start)*HZ+1e-8) : -1;
    const phase=j>=0 ? result.phase[j] : undefined;
    const text=`视频游标 · 会话 ${t.toFixed(2)} s`+(edge ? ' · 超出当前时间窗' : phase==null ? '' : ` · ${labels[phase] ?? '无法判断'}`);
    if (readout.textContent!==text) readout.textContent=text;
  }

  const ready=hasAngles(), video=$('squatAngleVideoCursor'), guide=$('squatAngleHoverCursor');
  const videoEdge=placeCursor(anglePlot,video,ready ? t : null,true);
  const inspected=ready ? hoverTime ?? (video.hidden || videoEdge ? pickedAngleTime : null) : null;
  placeCursor(timelinePlot,$('squatTimelineHoverCursor'),inspected,false);
  placeCursor(anglePlot,guide,inspected,false);
  const time=!guide.hidden ? inspected : !video.hidden && !videoEdge ? t : null;
  const values=$('squatAngleValues');
  values.hidden=!ready;
  if (!ready) return;
  let text='移到图上查看同一时刻的双踝角度';
  if (time!=null) {
    const sides=[0,1].map(i=>squatTraceAt(angles[i]?.traces ?? [],time));
    const details=sides.some(Boolean) ? squatAngleLabels.map((label,i)=>
      `${label} ${sides.map((side,j)=>`${j?'右':'左'} ${side ? signed(side[i])+'°' : '—'}`).join(' / ')}`).join('　·　')
      : '此时刻无角度数据';
    text=`${time.toFixed(2)} s　·　${details}`;
  }
  if (values.textContent!==text) values.textContent=text;
}

export function drawSquat() {
  const cv=$<HTMLCanvasElement>('squatTimeline');
  if (!cv || cv.clientWidth<100 || !state.sides.length) return;
  const plot=geometry(cv), {w,h,l,r,a,b}=plot; if (!(b>a)) return;
  hoverTime=null;
  const g=cv.getContext('2d'), k=window.devicePixelRatio||1, css=getComputedStyle(cv);
  const ink=css.getPropertyValue('--ink'),line=css.getPropertyValue('--line'), bg=css.getPropertyValue('--bg');
  cv.width=Math.round(w*k); cv.height=Math.round(h*k); g.setTransform(k,0,0,k,0,0);
  const x=(t:number)=>squatX(plot,t);
  g.fillStyle=bg; g.fillRect(0,0,w,h);
  const ticks=squatTicks(plot);
  g.strokeStyle=line; g.beginPath();
  for (const tick of ticks) {g.moveTo(tick.x,18);g.lineTo(tick.x,78);}
  g.stroke(); drawSquatTimeLabels(g,ticks,96,ink);
  for (const [s,e,p] of clips) {
    if(e<=a || s>=b) continue;
    const x0=x(Math.max(a,s)),x1=x(Math.min(b,e));
    g.fillStyle=p<0 ? '#a0a6b0' : colors[p];g.fillRect(x0,30,Math.max(.5,x1-x0),30);
    if(p<0) {
      g.save();g.beginPath();g.rect(x0,30,x1-x0,30);g.clip();g.strokeStyle=bg;
      for(let j=x0-30;j<x1;j+=7) {g.beginPath();g.moveTo(j,60);g.lineTo(j+30,30);g.stroke();}
      g.restore();
    }
  }
  result?.reps.forEach((rep,i)=>{
    if(rep[3]<a || rep[0]>b) return;
    const s=x(Math.max(a,rep[0])), e=x(Math.min(b,rep[3]));
    if(i===selected) {g.strokeStyle=ink;g.lineWidth=2;g.strokeRect(s,27,e-s,36);g.lineWidth=1;}
    if(e-s>22) {g.fillStyle=ink;g.textAlign='center';g.fillText(String(i+1),(s+e)/2,23);}
  });
  if (!result) {g.fillStyle=ink;g.textAlign='center';g.fillText('点击「预测」后显示状态时间轴', (l+r)/2,51);}
  else if (!clips.some(([s,e])=>e>a && s<b)) {g.fillStyle=ink;g.textAlign='center';g.fillText('当前时间窗不在上次预测范围内', (l+r)/2,51);}
  const chart=$<HTMLCanvasElement>('squatAngleChart');
  const empty=busy ? '正在计算双踝角度…' : !result ? '点击「预测」后显示角度曲线' : !result.reps.length
    ? '没有完整深蹲，无法绘制角度曲线' : !hasAngles() ? '双踝数据不足，无法绘制角度曲线' : '当前时间窗没有可绘制的深蹲角度';
  drawSquatAngleChart(chart,geometry(chart),[angles[0]?.traces ?? [],angles[1]?.traces ?? []],clips,colors,empty);
  paintSquatCursor();
}

export function renderSquat() {
  if (!$('viewSquat')) return;
  if(source!==state.sides) {reset(true);source=state.sides;}
  const enabled=!!squatPackage(source);
  $<HTMLButtonElement>('squatRun').disabled=!enabled || busy;
  $<HTMLSelectElement>('squatScope').disabled=busy;
  $<HTMLButtonElement>('squatCancel').hidden=!busy;
  $('squatStatus').textContent=message || (enabled ? '准备就绪。选择范围并点击「预测」。' : '需要同一个 Walkpad ZIP，且设备位置明确标为左踝、右踝。');
  const rows=$('squatRows'); rows.replaceChildren();
  result?.reps.forEach((rep,i)=>{
    const tr=document.createElement('tr'); tr.tabIndex=0; tr.classList.toggle('selected',i===selected);
    const go=(t:number)=>{focus(i);seekVideoSession(t,true);};
    tr.onclick=()=>go(rep[0]);
    tr.onkeydown=e=>{if(e.target===tr && (e.key==='Enter'||e.key===' ')) {e.preventDefault();go(rep[0]);}};
    for(const [j,val] of [String(i+1),...rep.map(v=>v.toFixed(3)),(rep[3]-rep[0]).toFixed(3)].entries()) {
      const cell=document.createElement('td');
      if(j>=1 && j<=4) {
        const button=document.createElement('button');
        button.type='button'; button.className='lnk'; button.textContent=val;
        button.setAttribute('aria-label',`第 ${i+1} 次 · ${events[j-1]} · ${val} s`);
        button.title='定位事件；如有视频则跳到该时刻';
        button.onclick=e=>{e.stopPropagation();go(rep[j-1]);};
        cell.append(button);
      } else cell.textContent=val;
      tr.append(cell);
    }
    rows.append(tr);
  });
  $('squatEmpty').textContent=!result ? '暂无预测结果' : result.reps.length ? '' : '该范围未识别到完整深蹲。';
  renderSquatAngles();
  renderSquatMetrics();
  drawSquat();
}

const CONF_CN = {hi: '高', mid: '中', lo: '低'};
const signed = (v: number, d = 1) => (v < 0 ? '−' : '+') + Math.abs(v).toFixed(d);
const num = (v: number | null | undefined, d = 1) => Number.isFinite(v) ? signed(v as number, d) : '—';

function angleNote(a: SideAngles) {
  if (!a?.ok) return a?.reason || '未计算';
  const src = a.upSource === 'stand' ? '蹲前站立定竖直' : `静止段 ${a.quietDur.toFixed(1)} s`;
  const axis = a.axisSource === 'net-turn' ? '净旋转定轴' : '屈伸主轴';
  return `${src}（离散 ${a.upSpread.toFixed(1)}°）· ${axis} 能量占比 ${(a.ratio*100).toFixed(0)}% · 逐次离散 ${a.spread.toFixed(1)}° · 可信度${CONF_CN[a.conf]}`;
}

function renderSquatAngles() {
  const rows = $('squatAngleRows');
  if (!rows) return;
  rows.replaceChildren();
  const L = angles[0], R = angles[1];
  const status: string[] = [];
  for (const [name, a] of [['左脚', L], ['右脚', R]] as [string, SideAngles][]) {
    if (a?.ok) status.push(`${name} ${angleNote(a)}`);
    else if (a) status.push(`${name} 无法给出三向角：${a.reason}`);
  }
  if (!angles.length) status.push('点「预测」后给出逐次三向角。');
  if (L?.ok || R?.ok) {
    const hasAppHeading=[L, R].some(a=>a?.ok && a.reps.some(x=>Number.isFinite(x.headingApp)));
    status.push(`${hasAppHeading ? '' : '无可用 APP 航向值；'}表中航向为站立期陀螺积分，可能累积漂移。`);
  }
  const warn = [L, R].flatMap(a => a?.ok ? a.warns ?? [] : []).filter((v, i, xs) => xs.indexOf(v) === i);
  $('squatAngleStatus').textContent = [...status, ...warn].join('　·　');
  const reps = result?.reps ?? [];
  for (let i = 0; i < reps.length; i++) {
    const l = L?.ok ? L.reps[i] : undefined, r = R?.ok ? R.reps[i] : undefined;
    const tr = document.createElement('tr');
    const cells = [String(i+1), num(l?.ap), num(l?.ml), num(l?.heading), num(r?.ap), num(r?.ml), num(r?.heading),
      `${num(l?.back)} / ${num(r?.back)}`, num(l?.hold ?? r?.hold, 2)];
    for (const v of cells) { const td = document.createElement('td'); td.textContent = v; tr.append(td); }
    rows.append(tr);
  }
}

export function initSquat() {
  const cv=$<HTMLCanvasElement>('squatTimeline'), chart=$<HTMLCanvasElement>('squatAngleChart');
  if (!cv || !chart) return;
  initSquatMetrics();
  $('squatVersion').textContent=`分段模型 v${modelVersion} · ${HZ} Hz`;
  $('squatRun').onclick=()=>void run();
  $('squatCancel').onclick=()=>{reset();message='预测已取消。';renderSquat();};
  const observer=new ResizeObserver(drawSquat); observer.observe(cv); observer.observe(chart);
  const hit=(canvas:HTMLCanvasElement,event:MouseEvent)=>{
    const {l,r,a,b}=geometry(canvas), dx=event.clientX-canvas.getBoundingClientRect().left-canvas.clientLeft;
    return dx>=l && dx<=r ? a+(b-a)*(dx-l)/(r-l) : NaN;
  };
  for (const canvas of [cv,chart]) {
    canvas.onpointermove=e=>{
      const t=hit(canvas,e); hoverTime=Number.isFinite(t) ? t : null; paintSquatCursor();
    };
    canvas.onpointerleave=canvas.onpointercancel=()=>{hoverTime=null; paintSquatCursor();};
  }
  cv.onclick=e=>{
    const t=hit(cv,e); if(!Number.isFinite(t)) return;
    const i=result?.reps.findIndex(r=>r[0]<=t && t<r[3]);
    if(i>=0) focus(i);
    seekVideoSession(t,true);
  };
  cv.onmousemove=e=>{
    const t=hit(cv,e), s=clips.find(c=>c[0]<=t && t<c[1]);
    cv.title=s ? `${t.toFixed(3)} s · ${labels[s[2]] ?? '无法判断'}` : '';
  };
  chart.onclick=e=>{
    const t=hit(chart,e); if(!Number.isFinite(t) || !hasAngles()) return;
    pickedAngleTime=t; seekVideoSession(t,true); paintSquatCursor();
  };
}
