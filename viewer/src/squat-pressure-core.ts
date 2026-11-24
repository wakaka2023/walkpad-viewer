/* Keep the squat pressure calculations independent of rendering and ankle prediction. */
import {NCH} from './core/constants.ts';
import {D56_CANVAS, D56_POINTS} from './squat-pressure-layout.ts';
import type {PkgFile} from './types';

export const PRESSURE_COLUMNS = ['effectiveElapsedUs', 'validity', 'frameType', 'pressureAdc', 'deviceSequence', 'segmentId', 'periodId'];
export type Foot = 'left' | 'right';
export interface PressureSource {
  foot: Foot;
  files: Pick<PkgFile, 'ab' | 'rgs' | 'cols'>[];
  timing: {segmentId: string; uncertaintyUs: number}[];
}

const object = (x: unknown): Record<string, unknown> | null => x != null && typeof x==='object' && !Array.isArray(x)
  ? x as Record<string, unknown> : null;
const identifier = (x: unknown) => typeof x==='string' && x.trim().length > 0 ? x.trim() : null;
const numeric = (x: unknown): number => typeof x==='number' || typeof x==='bigint' ? Number(x) : NaN;

/** Only a supported common-clock mapping permits pairing; independent per-device zeroes do not. */
export function pressureTiming(mappings: readonly unknown[], sourceId: string): PressureSource['timing'] {
  const bySegment=new Map<string,number>();
  for(const entry of mappings) {
    const m=object(entry), map=object(m?.mapping), segment=m?.segmentId;
    if(m?.sourceId!==sourceId || typeof segment!=='string' || !segment || m.timeBasis!=='deviceClock'
        || map?.offsetBasis!=='min_delay_envelope_v1') continue;
    const estimates=[m.timeUncertaintyUs,map.estimatedUncertaintyUs].filter((v): v is number=>
      typeof v==='number' && Number.isFinite(v) && v>=0);
    if(estimates.length) bySegment.set(segment,Math.max(bySegment.get(segment) ?? 0,...estimates));
  }
  return Array.from(bySegment,([segmentId,uncertaintyUs])=>({segmentId,uncertaintyUs}));
}

const keys=['peakAdc','sumAdc','ap','ml'] as const;
export type PressureMetric = typeof keys[number];
type Metrics = [peakAdc:number,sumAdc:number,ap:number,ml:number];
const missing = (): Metrics => [NaN,NaN,NaN,NaN];

export function pressureFrame(foot: Foot, p: ArrayLike<number | bigint>): Metrics {
  // All cells are treated as equal area. ADC is a signal proxy, not a calibrated pressure or force.
  // Missing channels invalidate the frame; a partial sum would bias the foot totals and the share.
  if(p?.length!==NCH) return missing();
  const points=D56_POINTS[foot];
  let peak=0,total=0,x=0,y=0;
  for(let i=0;i<NCH;i++) {
    const adc=numeric(p[i]);
    if(!Number.isFinite(adc)) return missing();
    const w=Math.max(0,adc);
    peak=Math.max(peak,w);total+=w;x+=w*points[i][0];y+=w*points[i][1];
  }
  return [peak,total,total>0 ? 100*(1-y/total/D56_CANVAS.height) : NaN,total>0 ? 100*x/total/D56_CANVAS.width : NaN];
}

export interface TimedSeries {
  t: Float64Array;
  /** linked[i] authorizes interpolation from t[i-1] to t[i], not extrapolation. */
  linked: Uint8Array;
}
export interface PressureTrace extends TimedSeries {
  foot: Foot;
  values: Record<PressureMetric, Float64Array>;
  /** Shared nonzero period code only on samples with recognized session clock mapping. */
  pairPeriod: Uint32Array;
}
export interface SignalShare extends TimedSeries {
  leftPct: Float64Array;
  /** A paired positive-signal interval can be integrated even if its ratio is undefined at a zero endpoint. */
  paired: Uint8Array;
}
export type SignalReference = Record<'peakAdc' | 'sumAdc', number>;
export interface PressureResult {traces: PressureTrace[]; share: SignalShare | null; reference: SignalReference}

/** Shared by both feet, all reps and time windows in the open recording; never recomputed on zoom. */
export function pressureReference(traces: PressureTrace[]): SignalReference {
  const reference={peakAdc:0,sumAdc:0};
  for(const trace of traces) for(const key of ['peakAdc','sumAdc'] as const) {
    for(const v of trace.values[key]) if(Number.isFinite(v) && v>reference[key]) reference[key]=v;
  }
  return reference;
}

export function relativeSignal(value: number, reference: number) {
  return reference>0 && Number.isFinite(value) ? value/reference*100 : NaN;
}

export interface PressureRow {
  effectiveElapsedUs?: unknown; validity?: unknown; frameType?: unknown;
  pressureAdc?: ArrayLike<number | bigint>; deviceSequence?: unknown; segmentId?: unknown; periodId?: unknown;
}
type Sample = {us: number; seq: number; segment: string | null; period: string | null; value: Metrics; invalid: boolean};

export class PressureCollector {
  private samples: Sample[]=[];
  private source: Pick<PressureSource,'foot'|'timing'>;
  private originUs: number;
  private periods: Map<string,number>;

  constructor(source: Pick<PressureSource,'foot'|'timing'>, originUs: number, periods: Map<string,number>) {
    this.source=source;this.originUs=originUs;this.periods=periods;
  }

  add(row: PressureRow) {
    if(row.frameType!=null && row.frameType!=='insole') return;
    const us=numeric(row.effectiveElapsedUs);
    // Invalid leading/trailing frames normally have no effective time; don't put them at t=0.
    if(!Number.isSafeInteger(us)) return;
    const value=row.validity==='valid' ? pressureFrame(this.source.foot,row.pressureAdc) : missing();
    this.samples.push({us,seq:numeric(row.deviceSequence),segment:identifier(row.segmentId),period:identifier(row.periodId),
      value,invalid:!Number.isFinite(value[0])});
  }

  finish(): PressureTrace {
    // A late BLE frame can arrive out of order. Sort on the recorded session clock, never the read order.
    const ordered=this.samples.sort((a,b)=>a.us-b.us), points: Sample[]=[];
    for(let i=0,j;i<ordered.length;i=j) {
      for(j=i+1;j<ordered.length && ordered[j].us===ordered[i].us;j++);
      const q=ordered[i];
      // Conflicting timestamps have no uniquely assignable sample. Leave a hole, don't invent time.
      points.push(j===i+1 ? q : {...q,value:missing(),invalid:true});
    }
    const known=new Set(this.source.timing.map(m=>m.segmentId)), gaps: number[]=[];
    const continuous=(a: Sample,b: Sample)=>!a.invalid && !b.invalid && a.segment===b.segment && a.period===b.period
      && (!Number.isSafeInteger(a.seq) || !Number.isSafeInteger(b.seq) || b.seq-a.seq===1);
    for(let i=1;i<points.length;i++) if(continuous(points[i-1],points[i])) gaps.push(points[i].us-points[i-1].us);
    gaps.sort((a,b)=>a-b);
    // At most 2.5 local sample periods (80 ms hard cap); explicit lost sequences always break.
    const maxGap=gaps.length ? Math.min(80000,gaps[gaps.length>>1]*2.5) : 0;
    const n=points.length, t=new Float64Array(n), linked=new Uint8Array(n), pairPeriod=new Uint32Array(n);
    const values=Object.fromEntries(keys.map(key=>[key,new Float64Array(n)])) as PressureTrace['values'];
    for(let i=0;i<n;i++) {
      const p=points[i];
      t[i]=(p.us-this.originUs)/1e6;
      if(i && continuous(points[i-1],p) && p.us-points[i-1].us<=maxGap) linked[i]=1;
      keys.forEach((k,j)=>{values[k][i]=p.value[j];});
      if(p.period && p.segment && known.has(p.segment) && !p.invalid) {
        if(!this.periods.has(p.period)) this.periods.set(p.period,this.periods.size+1);
        pairPeriod[i]=this.periods.get(p.period);
      }
    }
    this.samples=[];
    return {t,linked,pairPeriod,values,foot:this.source.foot};
  }
}

export function pressureIndex(t: Float64Array, time: number, upper=false) {
  let lo=0,hi=t.length;
  while(lo<hi) {const m=(lo+hi)>>>1;if(t[m]<time || (upper && t[m]===time)) lo=m+1;else hi=m;}
  return lo;
}

function valueAt(trace: TimedSeries, values: Float64Array, time: number, i: number) {
  const {t,linked}=trace;
  if(t[i]===time) return values[i];
  if(!linked[i] || !Number.isFinite(values[i-1]) || !Number.isFinite(values[i])) return NaN;
  return values[i-1]+(values[i]-values[i-1])*(time-t[i-1])/(t[i]-t[i-1]);
}

export function pressureAt(trace: PressureTrace, key: PressureMetric, time: number) {
  return valueAt(trace,trace.values[key],time,pressureIndex(trace.t,time));
}

export function leftSignalShareAt(left: PressureTrace, right: PressureTrace, time: number) {
  if(!left || !right) return NaN;
  const li=pressureIndex(left.t,time),ri=pressureIndex(right.t,time),lp=left.pairPeriod[li],rp=right.pairPeriod[ri];
  if(!lp || lp!==rp || (time!==left.t[li] && lp!==left.pairPeriod[li-1])
      || (time!==right.t[ri] && rp!==right.pairPeriod[ri-1])) return NaN;
  const l=valueAt(left,left.values.sumAdc,time,li),r=valueAt(right,right.values.sumAdc,time,ri),total=l+r;
  return Number.isFinite(total) && total>0 ? l/total*100 : NaN;
}

export function pressureSignalShare(left: PressureTrace, right: PressureTrace): SignalShare | null {
  if(!left?.t.length || !right?.t.length) return null;
  const a=Math.max(left.t[0],right.t[0]),b=Math.min(left.t.at(-1),right.t.at(-1));
  if(b<a) return null;
  const times: number[]=[],values: number[]=[],links: number[]=[],paired: number[]=[];
  let hasPair=false;
  for(let l=pressureIndex(left.t,a),r=pressureIndex(right.t,a);l<left.t.length || r<right.t.length;) {
    const t=Math.min(left.t[l] ?? Infinity,right.t[r] ?? Infinity);
    if(t>b) break;
    if(left.t[l]===t) l++;
    if(right.t[r]===t) r++;
    const v=leftSignalShareAt(left,right,t),last=times.length-1;
    // The union has a knot at each change on either side. A valid midpoint rules out hidden gaps.
    const overlap=last>=0 && Number.isFinite(leftSignalShareAt(left,right,(times[last]+t)/2));
    paired.push(Number(overlap));
    links.push(Number(overlap && Number.isFinite(values[last]) && Number.isFinite(v)));
    hasPair ||= overlap || Number.isFinite(v);
    times.push(t);values.push(v);
  }
  return hasPair ? {t:Float64Array.from(times),linked:Uint8Array.from(links),leftPct:Float64Array.from(values),paired:Uint8Array.from(paired)} : null;
}
