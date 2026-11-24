import {ParquetHQ} from './data/parquet';
import {PressureCollector, pressureReference, pressureSignalShare} from './squat-pressure-core';
import type {PressureResult, PressureRow, PressureSource} from './squat-pressure-core';

export interface PressureRequest {sources: PressureSource[]; originUs: number}
export type PressureResponse = {result:PressureResult; error?:never} | {result?:never;error:string};

self.onmessage=async({data}: MessageEvent<PressureRequest>)=>{
  try {
    const {sources,originUs}=data, traces: PressureResult['traces']=[], periods=new Map<string,number>();
    for(const source of sources) {
      const collector=new PressureCollector(source,originUs,periods);
      for(const f of source.files) for(const [rowStart,rowEnd] of f.rgs) {
        // Read the original Parquet in row groups even for lazy recordings; the drawing LOD is not measurement data.
        const rows=await ParquetHQ.parquetReadObjects({file:f.ab,columns:f.cols,rowStart,rowEnd});
        for(const row of rows) collector.add(row as PressureRow);
      }
      traces.push(collector.finish());
    }
    const share=pressureSignalShare(traces.find(t=>t.foot==='left'),traces.find(t=>t.foot==='right'));
    const arrays=traces.flatMap(t=>[t.t,t.linked,t.pairPeriod,...Object.values(t.values)]);
    if(share) arrays.push(share.t,share.linked,share.leftPct,share.paired);
    const response: PressureResponse={result:{traces,share,reference:pressureReference(traces)}};
    self.postMessage(response,{transfer:arrays.map(a=>a.buffer)});
  } catch(e) {
    const response: PressureResponse={error:e instanceof Error ? e.message : String(e)};
    self.postMessage(response);
  }
};
