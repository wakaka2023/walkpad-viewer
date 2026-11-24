import assert from 'node:assert/strict';
import {decodeSquat, referenceWeight, squatEmissions} from '../src/squat-core.ts';
import fixture from './squat-v04-parity.json' with {type: 'json'};

const n=64;
let maxError=0;
for (const {pattern, alpha, emissions, phase} of fixture.cases) {
  const x=Array.from({length:24},(_,j)=>Float32Array.from({length:n},(_,t)=>{
    const v=.28*Math.sin((j+2)*(t+1)/83)+.20*Math.cos((t+2)/9+j/3);
    return ([2,8,12,18].includes(j) ? Math.abs(v) : v)+([2,8].includes(j) ? 1.45*pattern/2 : 0);
  }));
  const swapped=[...x.slice(6,12),...x.slice(0,6),...x.slice(18,24),...x.slice(12,18)];
  assert.ok(Math.abs(referenceWeight(x)-alpha[0])<2e-6);
  assert.equal(referenceWeight(x),referenceWeight(swapped));
  const z=squatEmissions(x);
  assert.deepEqual(z,squatEmissions(swapped));
  assert.deepEqual([...decodeSquat(z)],phase);
  for (let j=0;j<8;j++) for (const [k,t] of fixture.frames.entries()) {
    const error=Math.abs(z[j][t]-emissions[j][k]);
    assert.ok(error<2e-4, `PyTorch mismatch in case ${pattern}, head ${j}, frame ${t}: ${error}`);
    maxError=Math.max(maxError,error);
  }
}
console.log('squat-mixer: PyTorch emission parity and foot-order symmetry',maxError);
