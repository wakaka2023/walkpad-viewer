import {OUT_H, OUT_W, outlinePerimeter} from './geometry.ts';
import {D56_CANVAS} from './squat-pressure-layout.ts';
import {pressureAt, pressureIndex} from './squat-pressure-core.ts';
import type {Foot, PressureTrace} from './squat-pressure-core';

// Draw the shared D56 outline, trail and marker in the source drawing's 505 × 1740 coordinates.
// The geometry module's outline uses an upward axis; SVG and the supplied D56 points use downward y.
const {width,height}=D56_CANVAS, padX=24, padY=32;
const viewBox=`${-padX} ${-padY} ${width+2*padX} ${height+2*padY}`;
const outlinePath=outlinePerimeter().map(([x,y],i)=>
  `${i?'L':'M'}${(x/OUT_W*width).toFixed(3)} ${(height*(1-y/OUT_H)).toFixed(3)}`).join('')+'Z';
const svg=<K extends keyof SVGElementTagNameMap>(tag: K,attrs: Record<string,string|number>={})=>{
  const element=document.createElementNS('http://www.w3.org/2000/svg',tag);
  for(const [key,value] of Object.entries(attrs)) element.setAttribute(key,String(value));
  return element;
};

export const copPoint=(ap: number,ml: number)=>[(width*ml/100).toFixed(3),(height*(1-ap/100)).toFixed(3)] as const;

/** Visible history up to the cursor. Interpolate only at the two supported boundaries; gaps lift the pen. */
export function copTrailPath(trace: PressureTrace,start: number,time: number,atEnd?: readonly [number,number]) {
  if(!trace || !(time>start)) return '';
  const [endAp,endMl]=atEnd ?? [pressureAt(trace,'ap',time),pressureAt(trace,'ml',time)];
  if(!Number.isFinite(endAp) || !Number.isFinite(endMl)) return '';
  const {t,linked,values:{ap,ml}}=trace, hi=pressureIndex(t,time);
  const parts: string[]=[];
  let last='', hasLine=false;
  const point=(a: number,m: number,join: boolean)=>{
    if(!Number.isFinite(a) || !Number.isFinite(m)) {last='';return;}
    const xy=copPoint(a,m).join(' ');
    if(!last || !join) parts.push('M'+xy);
    else if(xy!==last) {parts.push('L'+xy);hasLine=true;}
    last=xy;
  };
  point(pressureAt(trace,'ap',start),pressureAt(trace,'ml',start),false);
  for(let i=pressureIndex(t,start,true);i<hi;i++) point(ap[i],ml[i],!!linked[i]);
  point(endAp,endMl,!!linked[hi]);
  return hasLine ? parts.join('') : '';
}

export function squatFootMap(foot: Foot) {
  const map=svg('svg',{viewBox,class:'squat-pressure-footprint'});
  // Only the left-derived outline is mirrored. The supplied right-foot readings already use right-foot coordinates.
  const pose=foot==='right' ? {transform:`translate(${width} 0) scale(-1 1)`} : {};
  const outline=svg('path',{d:outlinePath,...pose,class:'squat-pressure-outline'});
  const path=svg('path',{class:'squat-pressure-path'});
  const dot=svg('circle',{class:'squat-pressure-dot',r:22,hidden:''});
  map.append(outline,path,dot);
  return {map,path,dot};
}
