import {deviceAnglesOnSquatGrid, predictSquats, wristOnSquatGrid, type AnkleStream} from './squat-core';
import {extractSquatFeatures, type RepFeatures} from './squat-score-features';
import {squatSideAngles, type SideAngles} from './squat-axis';

type Streams = {inputs: [AnkleStream, AnkleStream]; origin: number; aligned?: boolean};
export type CalibrationInput = Streams & {firstBottom: true};
type Input = CalibrationInput | (Streams & {range: {a:number;b:number}|null; wrist?: AnkleStream});
self.onmessage = ({data}: MessageEvent<Input>) => {
  const aligned = data.aligned === true;
  try {
    if ('firstBottom' in data) {
      const [rep] = predictSquats(data.inputs, data.origin, null, undefined, aligned).reps;
      if (!rep) throw new Error('整段双踝数据没有识别到完整深蹲。');
      self.postMessage({firstBottom: (rep[1]+rep[2])/2});
      return;
    }
    let raw: [Float64Array,Float64Array], offset=0;
    const result=predictSquats(data.inputs,data.origin,data.range,(grid,mapping)=>{raw=grid;offset=mapping;},aligned);
    let wristGrid: Float64Array, wristIssue='', features: RepFeatures[], scoreIssue='', repAngles: SideAngles[] = [];
    if (data.wrist && result.reps.length) {
      try {wristGrid=wristOnSquatGrid(data.wrist,data.inputs[0],result.shifts[0],offset,result.start,result.phase.length,aligned);}
      catch(e) {wristIssue='腕部未通过对时质检：'+(e instanceof Error ? e.message : String(e));}
    }
    try {repAngles=data.inputs.map((s,i)=>squatSideAngles({t:s.t,values:s.values,heading:s.heading},result.shifts[i]+offset,result.reps));}
    catch(e) {scoreIssue='三向角计算失败：'+(e instanceof Error ? e.message : String(e));}
    try {
      const angles=data.inputs.map((s,i)=>deviceAnglesOnSquatGrid(s,result.shifts[i]+offset,result.start,result.phase.length));
      features=extractSquatFeatures(result.start,result.phase,raw,result.reps,wristGrid,angles,repAngles);
    }
    catch {scoreIssue += (scoreIssue ? '；' : '')+'逐次指标计算失败；分段结果仍可使用。';}
    const traces=repAngles.flatMap(side=>side.traces ?? []);
    const transfer=[result.phase.buffer,...traces.flatMap(({t,deg,linked})=>[t.buffer,deg.buffer,linked.buffer])];
    self.postMessage({result,features,scoreIssue,wristIssue,repAngles}, {transfer});
  } catch (e) { self.postMessage({error:e instanceof Error ? e.message : String(e)}); }
};
