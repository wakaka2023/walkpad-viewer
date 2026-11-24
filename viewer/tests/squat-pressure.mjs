import assert from 'node:assert/strict';
import {PressureCollector, leftSignalShareAt, pressureAt, pressureFrame, pressureReference, pressureSignalShare,
  pressureTiming, relativeSignal} from '../src/squat-pressure-core.ts';
import {summarizeSquatPressure} from '../src/squat-pressure-summary.ts';
import {copPoint, copTrailPath} from '../src/squat-pressure-foot-map.ts';
import {D56_POINTS} from '../src/squat-pressure-layout.ts';

const almost=(a,b)=>assert.ok(Math.abs(a-b)<1e-9,`Expected ${b}; got ${a}`);
const signal=(toe=0,heel=0)=>{const a=new Int16Array(56);a[0]=toe;a[55]=heel;return a;};
const [peak,total,ap,ml]=pressureFrame('left',signal(100,50));
assert.equal(peak,100);
assert.equal(total,150); // No extra weighting by channel area.
almost(ap,100*(1-(112*100+1573*50)/150/1740)); // D56 v3 P1 and P56, not an evenly spaced grid.
almost(ml,100*(360*100+79*50)/150/505);
const rightToe=pressureFrame('right',signal(100));
assert.equal(rightToe[1],100);
almost(rightToe[2],100*(1-113/1740));
almost(rightToe[3],100*146/505); // Right coordinates are already mirrored.
const adc=new Int16Array(56).fill(-3);adc[0]=200;
assert.deepEqual(pressureFrame('left',adc),pressureFrame('left',signal(200)));
assert.deepEqual(pressureFrame('left',signal()),[0,0,NaN,NaN]);
assert.ok(pressureFrame('left',adc.slice(1)).every(Number.isNaN));
for(const invalid of [null,NaN,'10']) assert.ok(pressureFrame('left',[invalid,...adc.slice(1)]).every(Number.isNaN));

// With one loaded channel the SVG marker must land on the source drawing point, on both sides.
for(const [side,points] of Object.entries(D56_POINTS)) for(const [channel,expected] of points.entries()) {
  const p=new Uint16Array(points.length);p[channel]=100;
  const [, ,ap,ml]=pressureFrame(side,p);
  assert.deepEqual(copPoint(ap,ml).map(Number),[...expected],`${side} P${channel+1}`);
}
assert.deepEqual(copPoint(50,50).map(Number),[252.5,870]);

const leftMap={sourceId:'l',segmentId:'a-left',timeBasis:'deviceClock',timeUncertaintyUs:85129,
  mapping:{offsetBasis:'min_delay_envelope_v1',estimatedUncertaintyUs:84000}};
const rightMap={...leftMap,sourceId:'r',segmentId:'a-right',timeUncertaintyUs:104660};
const mappings=[leftMap,rightMap,{...rightMap,segmentId:'uncertain',mapping:{offsetBasis:'legacy_zero_fallback'}}];
assert.deepEqual(pressureTiming(mappings,'l'),[{segmentId:'a-left',uncertaintyUs:85129}]);
assert.deepEqual(pressureTiming([{...leftMap,timeUncertaintyUs:null,mapping:{offsetBasis:'min_delay_envelope_v1'}}],'l'),[]);
assert.deepEqual(pressureTiming([leftMap],'unknown'),[]);
const origin=2680000,periods=new Map();
function trace(foot,samples,{mapped=true,periodId='A'}={}) {
  const side=foot==='left'?'l':'r';
  const collector=new PressureCollector({foot,timing:mapped?pressureTiming(mappings,side):[]},origin,periods);
  samples.forEach(([ms,seq,toe=100,patch])=>collector.add({effectiveElapsedUs:BigInt(origin+ms*1000),deviceSequence:BigInt(seq),
    frameType:'insole',validity:'valid',periodId,segmentId:'a-'+foot,pressureAdc:signal(toe),...patch}));
  return collector.finish();
}
const lt=trace('left',[[20,1],[60,3],[40,2],[100,5],[120,6,100,{validity:'invalid'}],[140,7],[160,8,0],
  [180,9,100,{pressureAdc:adc.slice(0,55)}],[200,10],[220,11,100,{periodId:'B'}],[240,12],[240,13],
  [260,14,100,{frameType:'magnetic'}],[280,15,100,{segmentId:'other'}],[800,16],[820,17]]);
const rt=trace('right',[[10,1,50],[30,2,50],[50,3,50],[70,4,50],[90,5,50],[110,6,50],[130,7,50],
  [150,8,0],[160,9,0],[170,10,50],[190,11,50],[210,12,50],[230,13,50],[250,14,50]]);
assert.equal(lt.t[0],.02);
assert.equal(lt.t[1],.04); // Out-of-order BLE notifications use the recorded common time.
assert.ok(!lt.t.includes(.26));
assert.equal(pressureAt(lt,'sumAdc',.03),100);
assert.equal(pressureAt(lt,'sumAdc',.1),100);
for(const t of [.01,.08,.11,.12,.13,.18,.19,.215,.24,.5]) assert.ok(Number.isNaN(pressureAt(lt,'sumAdc',t)),`Gap at ${t}`);
almost(leftSignalShareAt(lt,rt,.05),100*100/150);
assert.ok(Number.isNaN(leftSignalShareAt(lt,rt,.08))); // Left loses a sequence.
assert.ok(Number.isNaN(leftSignalShareAt(lt,rt,.16))); // Both valid but no signal.
assert.ok(Number.isNaN(leftSignalShareAt(lt,rt,.22))); // Different recording periods.
const share=pressureSignalShare(lt,rt);
assert.equal(share.t[0],.02);
assert.equal(share.t.at(-1),.25);
assert.ok(share.leftPct.some(Number.isFinite));
assert.equal(share.linked[share.t.indexOf(.09)],0);
assert.equal(share.paired[share.t.indexOf(.09)],0);
assert.equal(pressureSignalShare(lt,null),null);
assert.equal(pressureSignalShare(lt,trace('right',[[30,1],[50,2]],{mapped:false})),null);
assert.equal(pressureSignalShare(lt,trace('right',[[900,1],[920,2]])),null);
const zeros=trace('left',[[10,1,0],[30,2,0]]);
assert.equal(leftSignalShareAt(zeros,rt,.02),0); // No left signal, but valid paired right signal.
assert.equal(leftSignalShareAt(rt,zeros,.02),100);
const zeroRight=trace('right',[[10,1,0],[30,2,0]]);
assert.equal(pressureSignalShare(zeros,zeroRight),null);
assert.ok(Number.isNaN(relativeSignal(0,pressureReference([zeros,zeroRight]).peakAdc)));

// Keep a paired interval with positive signal even if its zero endpoint cannot have an instantaneous share.
function resultFor(left,right) {
  const traces=[left,right].filter(Boolean);
  return {traces,share:pressureSignalShare(left,right),reference:pressureReference(traces)};
}
const ramp=resultFor(trace('left',[[10,1,0],[30,2,150]]),trace('right',[[10,1,0],[30,2,50]]));
assert.ok(Number.isNaN(ramp.share.leftPct[0]));
assert.deepEqual([...ramp.share.linked],[0,0]);
assert.deepEqual([...ramp.share.paired],[0,1]);
almost(summarizeSquatPressure(ramp,.01,.03).leftPct,75);
almost(summarizeSquatPressure(ramp,.01,.03).pairedCoverage,1);

// The rep window cuts through asynchronous sample intervals; outside frames set the recording reference,
// but never supply a rep's maximum or unpaired time to its L/R signal share.
function savedTrace(foot,t,sumAdc,peakAdc) {
  return {foot,t:Float64Array.from(t),linked:Uint8Array.from(t,(_,i)=>Number(i>0)),
    pairPeriod:new Uint32Array(t.length).fill(1),values:{sumAdc:Float64Array.from(sumAdc),peakAdc:Float64Array.from(peakAdc)}};
}
const repL=savedTrace('left',[0,.5,1,1.5,2],[10,30,40,NaN,200],[4,18,15,NaN,80]);
const repR=savedTrace('right',[0,.25,.75,1,1.5,2],[30,50,10,20,50,80],[15,17,5,10,22,40]);
const paired=resultFor(repL,repR);
assert.deepEqual(paired.reference,{peakAdc:80,sumAdc:200});
const rep=summarizeSquatPressure(paired,.25,1.75);
almost(rep.left.peakSignalPct,18/80*100);
almost(rep.left.peakTotalPct,40/200*100);
almost(rep.right.peakSignalPct,22/80*100);
almost(rep.right.peakTotalPct,50/200*100);
almost(rep.left.meanTotalPct,23.75/.75/200*100);
almost(rep.right.meanTotalPct,50.625/1.5/200*100);
almost(rep.left.coverage,.5);
almost(rep.right.coverage,1);
almost(rep.leftPct,100*23.75/(23.75+18.75));
almost(rep.pairedCoverage,.5);
const clipped=summarizeSquatPressure(paired,.26,.27); // Interpolable intervals without an original frame.
assert.ok(Number.isNaN(clipped.left.peakSignalPct) && Number.isNaN(clipped.right.peakTotalPct));
almost(clipped.left.meanTotalPct,20.6/200*100);
almost(clipped.pairedCoverage,1);
const noPair=summarizeSquatPressure({...paired,share:null},.25,1.75);
assert.ok(Number.isFinite(noPair.left.meanTotalPct) && Number.isNaN(noPair.leftPct));
assert.equal(noPair.pairedCoverage,0);
const oneFoot=summarizeSquatPressure(resultFor(repL,null),.25,1.75);
assert.ok(Number.isFinite(oneFoot.left.meanTotalPct) && Number.isNaN(oneFoot.leftPct));
assert.equal(oneFoot.right,null);

// COP trail interpolates the cursor and the rep start, draws nothing from the future, and does not join gaps.
const trail={...savedTrace('left',[0,.25,.5,.75,1,1.25],[],[]),linked:new Uint8Array([0,1,1,0,0,1]),
  values:{ap:Float64Array.from([0,10,30,NaN,60,70]),ml:Float64Array.from([0,20,50,NaN,100,85])}};
assert.equal(copTrailPath(trail,.125,.375),'M50.500 1653.000L101.000 1566.000L176.750 1392.000');
assert.equal(copTrailPath(trail,.26,.30),'M107.060 1552.080L131.300 1496.400');
assert.equal(copTrailPath(trail,.125,.625),'');
assert.equal(copTrailPath(trail,1,1),'');
assert.equal(copTrailPath(trail,.125,1.25),'M50.500 1653.000L101.000 1566.000L252.500 1218.000M505.000 696.000L429.250 522.000');
console.log('Squat pressure: 112 D56 v3 marker positions, shared relative references, per-rep extrema, paired integration/coverage and COP trail passed.');
