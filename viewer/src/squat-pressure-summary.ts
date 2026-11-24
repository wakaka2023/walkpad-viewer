import {pressureAt, pressureIndex, relativeSignal} from './squat-pressure-core.ts';
import type {Foot, PressureResult, PressureTrace, SignalReference, TimedSeries} from './squat-pressure-core';

/** Linked segments clipped to the rep, never extrapolated or joined across a gap. */
function* validIntervals(series: TimedSeries,a: number,b: number,links=series.linked) {
  const {t}=series;
  for(let i=Math.max(1,pressureIndex(t,a,true));i<t.length && t[i-1]<b;i++) {
    const lo=Math.max(a,t[i-1]),hi=Math.min(b,t[i]);
    if(links[i] && hi>lo) yield [lo,hi] as const;
  }
}

function footSummary(trace: PressureTrace | undefined,reference: SignalReference,a: number,b: number) {
  if(!trace) return null;
  const {t,values}=trace;
  let peak=-Infinity,peakTotal=-Infinity;
  const end=pressureIndex(t,b,true);
  // Peaks use original in-rep measurements, not interpolation, plotting LOD or neighboring frames.
  for(let i=pressureIndex(t,a);i<end;i++) {
    if(Number.isFinite(values.peakAdc[i])) peak=Math.max(peak,values.peakAdc[i]);
    if(Number.isFinite(values.sumAdc[i])) peakTotal=Math.max(peakTotal,values.sumAdc[i]);
  }
  let integral=0,seconds=0;
  for(const [lo,hi] of validIntervals(trace,a,b)) {
    const s0=pressureAt(trace,'sumAdc',lo),s1=pressureAt(trace,'sumAdc',hi);
    if(!Number.isFinite(s0) || !Number.isFinite(s1)) continue;
    seconds+=hi-lo;
    integral+=(s0+s1)/2*(hi-lo);
  }
  return {peakSignalPct:relativeSignal(peak,reference.peakAdc),peakTotalPct:relativeSignal(peakTotal,reference.sumAdc),
    meanTotalPct:relativeSignal(seconds>0 ? integral/seconds : NaN,reference.sumAdc),coverage:Math.min(1,seconds/(b-a))};
}

/** For L/R share integrate the two signals on the SAME verified paired intervals. */
export function summarizeSquatPressure(result: PressureResult,a: number,b: number) {
  const empty={left:null,right:null,leftPct:NaN,pairedCoverage:0};
  if(!result || !Number.isFinite(a) || !Number.isFinite(b) || !(b>a)) return empty;
  const find=(foot: Foot)=>result.traces.find(t=>t.foot===foot);
  const left=find('left'),right=find('right'),{share,reference}=result;
  let leftIntegral=0,rightIntegral=0,seconds=0;
  if(left && right && share) {
    for(const [lo,hi] of validIntervals(share,a,b,share.paired)) {
      const l=(pressureAt(left,'sumAdc',lo)+pressureAt(left,'sumAdc',hi))/2*(hi-lo);
      const r=(pressureAt(right,'sumAdc',lo)+pressureAt(right,'sumAdc',hi))/2*(hi-lo);
      if(!Number.isFinite(l+r) || l+r<=0) continue;
      leftIntegral+=l;rightIntegral+=r;seconds+=hi-lo;
    }
  }
  return {left:footSummary(left,reference,a,b),right:footSummary(right,reference,a,b),
    leftPct:seconds>0 ? leftIntegral/(leftIntegral+rightIntegral)*100 : NaN,pairedCoverage:Math.min(1,seconds/(b-a))};
}
