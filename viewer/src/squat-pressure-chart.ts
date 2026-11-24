import {drawSquatTimeLabels, squatTicks, squatX} from './squat-angle-chart';
import type {SquatPlot} from './squat-angle-chart';
import {pressureIndex, relativeSignal} from './squat-pressure-core';
import type {Foot, PressureMetric, PressureResult, TimedSeries} from './squat-pressure-core';

type Clip = readonly [start: number,end: number,phase: number];
interface Series extends TimedSeries {foot: Foot; values: Float64Array; reference?: number; complement?: boolean}
interface Lane {title: string; series: Series[]}
const LANE_HEIGHT=75;
export const pressureChartHeight=(lanes: Lane[])=>lanes.length*LANE_HEIGHT+27;

export function pressureLanes({traces,share,reference}: PressureResult): Lane[] {
  const lane=(title: string,key: PressureMetric,ref?: number): Lane=>({title,
    series:ref===0 ? [] : traces.map(({foot,t,linked,values})=>({foot,t,linked,values:values[key],reference:ref}))});
  const lanes=[lane('COP 前后（鞋垫图 %）','ap'),lane('COP 横向（鞋垫图 %）','ml'),
    lane('单点峰值（相对 %）','peakAdc',reference.peakAdc),lane('单脚总信号（相对 %）','sumAdc',reference.sumAdc)];
  if(share) {
    const {t,linked,leftPct:values}=share;
    lanes.push({title:'左右信号占比（%）',series:[{foot:'left',t,linked,values},
      {foot:'right',t,linked,values,complement:true}]});
  }
  return lanes;
}

/** Exact frame extrema in the visible window; a frame outside it cannot inflate the peak. */
export function pressureExtrema(values: Float64Array, t: Float64Array, a: number,b: number) {
  let min=Infinity,max=-Infinity;
  const end=pressureIndex(t,b,true);
  for(let i=pressureIndex(t,a);i<end;i++) if(Number.isFinite(values[i])) {
    const v=values[i]; min=Math.min(min,v);max=Math.max(max,v);
  }
  return {min,max};
}

// Draw first/min/max/last of each pixel column in order, preserving one-frame peaks even at full-session zoom.
// Flush on every disconnect so neither decimation nor clipping can invent a line across missing samples.
function strokeSeries(g: CanvasRenderingContext2D, s: Series, p: SquatPlot, y: (v: number)=>number) {
  const {t,values,linked,reference,complement}=s, lo=Math.max(0,pressureIndex(t,p.a)-1), hi=Math.min(t.length,pressureIndex(t,p.b,true)+1);
  let pen=false, singleton=false, last=-1, column=NaN, first=-1, end=-1, min=-1, max=-1, visible=false;
  const x=(i: number)=>squatX(p,t[i]), yy=(i: number)=>y(complement ? 100-values[i] : reference==null ? values[i] : relativeSignal(values[i],reference));
  const emit=(i: number)=>{
    if(i===last) return;
    if((t[i]>=p.a && t[i]<=p.b) || (pen && t[last]<=p.b && t[i]>=p.a)) visible=true;
    if(pen) {g.lineTo(x(i),yy(i));singleton=false;}
    else {g.moveTo(x(i),yy(i));pen=singleton=true;}
    last=i;
  };
  const flush=()=>{
    if(first<0) return;
    emit(first);emit(Math.min(min,max));emit(Math.max(min,max));emit(end);
    first=-1;
  };
  g.beginPath();
  for(let i=lo;i<hi;i++) {
    if(!linked[i] || !Number.isFinite(values[i])) {
      flush();
      if(singleton) g.lineTo(x(last)+.5,yy(last));
      pen=singleton=false;last=-1;
    }
    if(!Number.isFinite(values[i])) continue;
    const col=Math.floor(x(i));
    if(first<0 || column!==col) {flush();column=col;first=end=min=max=i;}
    else {end=i;if(values[i]<values[min]) min=i;if(values[i]>values[max]) max=i;}
  }
  flush();
  if(singleton) g.lineTo(x(last)+.5,yy(last));
  g.stroke();
  return visible;
}

export function drawPressureChart(canvas: HTMLCanvasElement,p: SquatPlot,lanes: Lane[],clips: readonly Clip[],colors: readonly string[]) {
  const {w,h,l,r,a,b}=p,g=canvas.getContext('2d'),dpr=window.devicePixelRatio||1;
  if(!g || w<100 || !(b>a) || !lanes.length) return;
  const css=getComputedStyle(canvas), c=(key: string)=>css.getPropertyValue(key).trim();
  const ink=c('--ink'),muted=c('--muted'),line=c('--line'),card=c('--card');
  const strokes={left:c('--squat-angle-left'),right:c('--squat-angle-right')}, ticks=squatTicks(p);
  canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);g.setTransform(dpr,0,0,dpr,0,0);
  g.fillStyle=c('--bg');g.fillRect(0,0,w,h);drawSquatTimeLabels(g,ticks,h-8,ink);
  lanes.forEach((lane,i)=>{
    const top=5+i*LANE_HEIGHT, upper=top+18, lower=top+LANE_HEIGHT-6;
    const y=(v: number)=>lower-(lower-upper)*v/100;
    g.font='600 12px system-ui';g.fillStyle=ink;g.textAlign='left';g.fillText(lane.title,l+3,top+11,r-l);
    g.fillStyle=card;g.fillRect(l,upper,r-l,lower-upper);
    g.globalAlpha=.11;
    for(const [s,e,phase] of clips) if(phase>0 && e>a && s<b) {
      const x0=squatX(p,Math.max(a,s));
      g.fillStyle=colors[phase];g.fillRect(x0,upper,squatX(p,Math.min(b,e))-x0,lower-upper);
    }
    g.globalAlpha=1;g.strokeStyle=line;g.beginPath();
    for(const tick of ticks) {g.moveTo(tick.x,upper);g.lineTo(tick.x,lower);}
    g.font='10px system-ui';g.textAlign='right';g.textBaseline='middle';g.fillStyle=muted;
    for(const v of [0,50,100]) {
      const yy=y(v);g.moveTo(l,yy);g.lineTo(r,yy);
      g.fillText(`${v}%`,l-5,yy,l-6);
    }
    g.stroke();g.textBaseline='alphabetic';
    let visible=false;
    g.save();g.beginPath();g.rect(l,upper-1,r-l,lower-upper+2);g.clip();g.lineWidth=1.75;g.lineJoin='round';g.lineCap='round';
    for(const s of lane.series) {
      g.strokeStyle=strokes[s.foot];g.setLineDash(s.foot==='right' ? [5,3] : []);
      visible=strokeSeries(g,s,p,y) || visible;
    }
    g.restore();
    if(!visible) {
      g.fillStyle=muted;g.textAlign='center';g.font='11px system-ui';
      g.fillText('当前时间窗无有效读数',(l+r)/2,(upper+lower)/2+4,r-l-8);
    }
  });
}
