import model from './squat-v03-model.json' with {type: 'json'};
import robust from './squat-v04-model.json' with {type: 'json'};
import {ankleTimebase, elapsedDelta, summarizeClock, type AnkleStream} from './squat-clock.ts';

export const SQUAT_VERSION = robust.version;
export const SQUAT_HZ = 50;
export type {AnkleStream};
export type SquatRep = [number, number, number, number];
export type SquatResult = {start: number; phase: Int8Array; reps: SquatRep[]; shifts: number[]; valid: number};

const seq = (n: number, f: (i: number) => number) => Float64Array.from({length: n}, (_, i) => f(i));
const sorted = (x: ArrayLike<number>) => Float64Array.from(x).sort();
const quantile = (x: Float64Array, q: number) => {
  const k = q * (x.length - 1), i = Math.floor(k), f = k - i;
  return f < .5 ? x[i] + f * (x[Math.ceil(k)] - x[i]) : x[Math.ceil(k)] - (1 - f) * (x[Math.ceil(k)] - x[i]);
};
const median = (x: ArrayLike<number>) => quantile(sorted(x), .5);
const clip = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross = (a: ArrayLike<number>, b: ArrayLike<number>) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const norm = (a: ArrayLike<number>) => Math.sqrt(dot(a, a));
const unit = (a: number[], floor = 1e-8) => { const n = Math.max(norm(a), floor); return a.map(v => v / n); };
const transpose = (x: Float64Array, n: number, k: number) => Array.from({length: k}, (_, j) => seq(n, i => x[i*k+j]));
const mirror = (n: number, i: number) => i < 0 ? -i-1 : i >= n ? 2*n-i-1 : i;

const kernels = new Map<number, Float64Array>();
function smooth(x: Float64Array, sigma: number) {
  let w = kernels.get(sigma);
  if (!w) {
    const r = Math.floor(4*sigma+.5);
    w = seq(2*r+1, i => Math.exp(-.5*((i-r)/sigma)**2));
    const s = w.reduce((a,b) => a+b); w = w.map(v => v/s); kernels.set(sigma,w);
  }
  const r = w.length >> 1, n = x.length;
  return seq(n, i => {
    let z = 0;
    for (let j = -r; j <= r; j++) z += w[j+r]*x[mirror(n,i+j)];
    return z;
  });
}

function filter(x: Float64Array, id: number) {
  const [b0,b1,b2,a1,a2,z0,z1] = model.filters[id];
  const n = x.length, pad = 9;
  const p = seq(n+2*pad, i => i<pad ? 2*x[0]-x[pad-i] : i>=n+pad ? 2*x[n-1]-x[2*n+pad-i-2] : x[i-pad]);
  function pass(s: Float64Array) {
    let u = z0*s[0], v = z1*s[0];
    return s.map(t => { const y = b0*t+u; u = b1*t-a1*y+v; v = b2*t-a2*y; return y; });
  }
  return pass(pass(p).reverse()).reverse().subarray(pad,pad+n);
}

function principal(m: number[][]) {
  const v = [[1,0,0],[0,1,0],[0,0,1]];
  for (let iter = 0; iter < 24; iter++) {
    let p = 0, q = 1;
    for (let i = 0; i < 3; i++) for (let j = i+1; j < 3; j++)
      if (Math.abs(m[i][j])>Math.abs(m[p][q])) { p=i; q=j; }
    const apq = m[p][q];
    if (Math.abs(apq) <= 1e-14*(1+Math.max(...m.map((r,i) => Math.abs(r[i]))))) break;
    const a = .5*Math.atan2(2*apq,m[q][q]-m[p][p]), c = Math.cos(a), s = Math.sin(a);
    const pp=m[p][p], qq=m[q][q];
    m[p][p]=c*c*pp-2*c*s*apq+s*s*qq; m[q][q]=s*s*pp+2*c*s*apq+c*c*qq; m[p][q]=m[q][p]=0;
    for (let k=0;k<3;k++) {
      if (k!==p && k!==q) {
        const u=m[k][p], w=m[k][q]; m[k][p]=m[p][k]=c*u-s*w; m[k][q]=m[q][k]=s*u+c*w;
      }
      const u=v[k][p], w=v[k][q]; v[k][p]=c*u-s*w; v[k][q]=s*u+c*w;
    }
  }
  const k = [0,1,2].reduce((b,i) => m[i][i]>m[b][b] ? i : b,0);
  return v.map(r => r[k]);
}

function foot(a: Float64Array[], w: Float64Array[]) {
  const n = a[0].length, vec = (a: Float64Array[], i: number) => a.map(x => x[i]);
  const an = seq(n,i=>norm(vec(a,i))), wn = smooth(seq(n,i=>norm(vec(w,i))),3);
  const order = sorted(wn), q32 = quantile(order,.32), q35 = quantile(order,.35);
  let quiet = Array.from({length:n},(_,i)=>i).filter(i=>Math.abs(an[i]-1)<.12 && wn[i]<q32);
  if (quiet.length<8) quiet=Array.from({length:n},(_,i)=>i).filter(i=>wn[i]<=q35);
  const g=unit(a.map(x=>median(quiet.map(i=>x[i]))));
  const w0=w.map(x=>{const m=median(quiet.map(i=>x[i]));return x.map(v=>v-m);});
  const proj=seq(n,i=>dot(vec(w0,i),g));
  const h=w0.map((x,j)=>x.map((v,i)=>v-proj[i]*g[j]));
  const cov=h.map(x=>h.map(y=>x.reduce((z,v,i)=>z+v*y[i],0)));
  let u=principal(cov), d=dot(u,g); u=u.map((v,i)=>v-d*g[i]);
  if (norm(u)<1e-8) {
    const j=[0,1,2].reduce((b,i)=>Math.abs(g[i])<Math.abs(g[b]) ? i : b,0);
    u=g.map((x,i)=>Number(i===j)-g[j]*x);
  }
  u=unit(u); const gu=cross(g,u);
  let z=smooth(seq(n,i=>dot(vec(w0,i),u)),2);
  const az=a.map(x=>filter(x,0)), rad=seq(n,i=>{const v=vec(az,i);return Math.atan2(dot(v,gu),dot(v,g));});
  let wrap=0;
  const accAngle=rad.map((x,i)=>{
    if (i) {const d=x-rad[i-1]; if (d>Math.PI) wrap-=2*Math.PI; else if (d< -Math.PI) wrap+=2*Math.PI;}
    return (x+wrap)*180/Math.PI;
  });
  let acc=0;
  const integ=z.map((x,i)=>i ? acc+=(z[i-1]+x)/(2*SQUAT_HZ) : 0);
  const low=filter(accAngle,2); const theta=filter(integ,1).map((x,i)=>x+low[i]);
  const center=median(quiet.map(i=>theta[i])); const th=theta.map(x=>x-center); const s=sorted(th);
  const sign=quantile(s,.95)>=-quantile(s,.05) ? 1 : -1;
  z=filter(z,1);
  const move=smooth(an.map(x=>(x-1)**2),4);
  return [th.map(x=>x*sign/30),z.map(x=>x*sign/75),th.map(x=>Math.abs(x)/30),seq(n,i=>norm(vec(h,i))/85),
    seq(n,i=>{const a=vec(w0,i), p=dot(a,u);return norm(a.map((v,j)=>v-p*u[j]))/40;}),move.map(x=>Math.sqrt(x)/.20)];
}

function radial(a: Float64Array[], w: Float64Array[]) {
  const n=a[0].length, x=a.map(v=>smooth(v,4)), vec=(a:Float64Array[],i:number)=>a.map(v=>v[i]);
  const an=seq(n,i=>norm(vec(x,i))), g=x.map(v=>v.map((t,i)=>t/Math.max(an[i],1e-8)));
  const wn=smooth(seq(n,i=>norm(vec(w,i))),3);
  const ix=Array.from({length:n},(_,i)=>i).filter(i=>i<45 || i>=n-45);
  const rank=wn.map((v,i)=>v+70*Math.abs(an[i]-1));
  const best=ix.sort((i,j)=>rank[i]-rank[j]).slice(0,Math.max(5,ix.length>>2));
  const ref=unit(g.map(v=>best.reduce((s,i)=>s+v[i],0)/best.length));
  const ang=new Float64Array(n), rw=new Float64Array(n), mag=new Float64Array(n), tang=new Float64Array(n), wh=new Float64Array(n);
  for (let i=0;i<n;i++) {
    const gg=vec(g,i), aa=vec(a,i), ww=vec(w,i), c=cross(ref,gg), si=norm(c), ag=dot(aa,gg), wg=dot(ww,gg);
    ang[i]=Math.atan2(si,dot(gg,ref))*180/Math.PI; rw[i]=dot(c,ww)/Math.max(si,.07);
    mag[i]=norm(aa)-1; tang[i]=norm(aa.map((v,j)=>v-ag*gg[j])); wh[i]=norm(ww.map((v,j)=>v-wg*gg[j]));
  }
  const sm=smooth(ang,3), slope=seq(n,i=>(sm[Math.min(n-1,i+1)]-sm[Math.max(0,i-1)])*SQUAT_HZ/(i===0||i===n-1 ? 1 : 2));
  return [ang.map(v=>v/30),smooth(rw,2).map(v=>v/75),slope.map(v=>v/75),smooth(mag,2).map(v=>v/.15),smooth(tang,2).map(v=>v/.15),smooth(wh,2).map(v=>v/85)];
}

function weights(source: Record<string,string>) {
  const params = new Map<string,Float32Array>();
  return (name: string) => {
    let x=params.get(name);
    if (!x) {
      const raw=atob(source[name]);
      x=new Float32Array(Uint8Array.from(raw,c=>c.charCodeAt(0)).buffer); params.set(name,x);
    }
    return x;
  };
}
const fixedWeights=weights(model.params), robustWeights=weights(robust.params);

function conv(x: Float32Array[], tensor: ReturnType<typeof weights>, p: string, outs: number,
    k = 1, dilation = 1, depthwise = false, relu = false) {
  const w=tensor(p+'.weight'), b=tensor(p+'.bias'), n=x[0].length, ins=depthwise ? 1 : x.length;
  return Array.from({length:outs},(_,o)=>{
    const y=new Float32Array(n), off=o*ins*k;
    for (let t=0;t<n;t++) {
      let z=b[o];
      if (k===1) {for (let j=0;j<ins;j++) z+=w[off+j]*x[j][t];}
      else {
        const a=t-(k>>1)*dilation;
        for (let j=0;j<ins;j++) {
          const q=x[depthwise ? Math.floor(o/3) : j], u=off+j*k;
          for (let i=0;i<k;i++) {const at=a+i*dilation; if (at>=0 && at<n) z+=w[u+i]*q[at];}
        }
      }
      y[t]=relu ? Math.max(0,z) : z;
    }
    return y;
  });
}

function ensemble(x: Float32Array[], tensor: ReturnType<typeof weights>, count: number) {
  const out=Array.from({length:8},()=>new Float64Array(x[0].length));
  for (let k=0;k<count;k++) {
    const p=`nets.${k}.`;
    let h=conv(conv(x,tensor,p+'stem',72,5,1,true),tensor,p+'mix',24,1,1,false,true);
    for (let i=0;i<5;i++) {
      const base=p+`layers.${i}.`;
      const y=conv(conv(h,tensor,base+'0',24,3,1<<i,false,true),tensor,base+'2',24);
      h=h.map((q,j)=>q.map((v,t)=>v+y[j][t]));
    }
    const z=[...conv(h,tensor,p+'phase',4),...conv(h,tensor,p+'edge',4)];
    out.forEach((q,j)=>{for(let i=0;i<q.length;i++) q[i]+=z[j][i]/count;});
  }
  return out;
}

const sigmoid = (x:number) => 1/(1+Math.exp(-x));
export function referenceWeight(x: Float32Array[]) {
  const evidence=[0,0,0,0];
  for (const k of [0,1]) {
    const main=x[2+6*k], radial=x[12+6*k], n=main.length;
    const delta=seq(n,i=>main[i]-radial[i]), mismatch=delta.map(Math.abs);
    let sum=0, weighted=0;
    for (let i=0;i<n;i++) {
      const w=Math.exp(-12*radial[i]);
      sum+=w; weighted+=w*mismatch[i];
    }
    const order=sorted(mismatch);
    const stats=[order[Math.floor((n-1)/2)],order[Math.floor(9*(n-1)/10)],
      Math.abs(delta.reduce((a,b)=>a+b,0)/n),weighted/sum];
    for (let i=0;i<4;i++) evidence[i]=Math.max(evidence[i],stats[i]);
  }
  return sigmoid(evidence.reduce((z,v,i)=>z+v*robust.gate.coef[i],robust.gate.intercept));
}

export function squatEmissions(x: Float32Array[]) {
  const alpha=referenceWeight(x), a=1-alpha;
  const swap=[...x.slice(6,12),...x.slice(0,6),...x.slice(18,24),...x.slice(12,18)];
  const out=Array.from({length:8},()=>new Float64Array(x[0].length));
  // Fuse probabilities for each foot order before averaging decoder scores.
  for (const u of [x,swap]) {
    const f=ensemble(u,fixedWeights,3), r=ensemble(u,robustWeights,robust.netCount);
    for (let t=0;t<x[0].length;t++) {
      const mf=Math.max(f[0][t],f[1][t],f[2][t],f[3][t]);
      const mr=Math.max(r[0][t],r[1][t],r[2][t],r[3][t]);
      let sf=0, sr=0;
      for (let j=0;j<4;j++) {
        sf+=(f[j][t]=Math.exp(f[j][t]-mf));
        sr+=(r[j][t]=Math.exp(r[j][t]-mr));
      }
      for (let j=0;j<4;j++) {
        out[j][t]+=.5*Math.log(Math.max(1e-8,a*f[j][t]/sf+alpha*r[j][t]/sr));
        const k=j+4, p=clip(a*sigmoid(f[k][t])+alpha*sigmoid(r[k][t]),1e-6,1-1e-6);
        out[k][t]+=.5*Math.log(p/(1-p));
      }
    }
  }
  return out;
}

function emissions(axes: Float64Array[][]) {
  const f=axes.map(s=>[s.slice(0,3),s.slice(3)] as const);
  const x=[...f.flatMap(([a,w])=>foot(a,w)),...f.flatMap(([a,w])=>radial(a,w))].map(a=>Float32Array.from(a,v=>clip(v,-12,12)));
  return squatEmissions(x);
}

export function decodeSquat(out: Float64Array[]) {
  const n=out[0].length, k=13, back=new Uint8Array(n*k);
  const phases=[0,1,1,1,1,2,2,2,2,3,3,3,3];
  const edge=out.slice(4).map(x=>smooth(x,1).map(v=>model.edgeWeight*clip(v+1,-5,2)));
  // Log-softmax subtracts the same value from every path at each tick.
  let prev=new Float64Array(k).fill(-Infinity), next=new Float64Array(k); prev[0]=out[0][0];
  for (let i=1;i<n;i++) {
    for (let j=0;j<k;j++) {
      let from=j-1, val=prev[from];
      if (j===0 || j%4===1) {from=j===0 ? 12 : j-1; val=prev[from]-2+edge[j===0 ? 3 : (j-1)>>2][i];}
      if ((j===0 || j%4===0) && (prev[j]>val || (prev[j]===val && j<from))) {from=j; val=prev[j];}
      back[i*k+j]=from; next[j]=val+out[phases[j]][i];
    }
    [prev,next]=[next,prev];
  }
  const y=new Int8Array(n);
  for (let i=n-1,q=0;i>0;i--) {y[i]=phases[q];q=back[i*k+q];}
  return y;
}

export function wristOnSquatGrid(wrist: AnkleStream, left: AnkleStream, leftShift: number, offset: number, start: number, n: number, aligned=false) {
  // 包内已声明会话时间轴时两路本就同轴，只有老包才需要用接收时刻反推腕踝之间的时钟差
  const rel = aligned ? elapsedDelta(wrist, left)
    : summarizeClock(wrist,true)[0] - summarizeClock(left,true)[0];
  const adjusted=(t:number)=>t/1e6+offset+leftShift+rel;
  const w=new Float64Array(n*6).fill(NaN), t=wrist.t;
  for (let i=0,j=1;i<n;i++) {
    const s=start+i/SQUAT_HZ;
    while(j<t.length-1 && adjusted(t[j])<s) j++;
    const lo=adjusted(t[j-1]), hi=adjusted(t[j]);
    if (s<lo-1e-6 || s>hi+1e-6 || (hi-lo>.12 && Math.min(Math.abs(s-lo),Math.abs(s-hi))>1e-6)) continue;
    const f=clip((s-lo)/(hi-lo),0,1);
    for (let c=0;c<6;c++) w[i*6+c]=wrist.values[(j-1)*6+c]+f*(wrist.values[j*6+c]-wrist.values[(j-1)*6+c]);
  }
  return w;
}

export function deviceAnglesOnSquatGrid(stream: AnkleStream, shift: number, start: number, n: number) {
  const t=stream.t, maxGap=.12;
  return stream.angles?.map(raw=>{
    const out=new Float64Array(n).fill(NaN);
    for(let i=0,j=1;i<n;i++) {
      const target=start+i/SQUAT_HZ-shift;
      while(j<t.length-1 && t[j]/1e6<target) j++;
      const a=t[j-1]/1e6, b=t[j]/1e6;
      if(Math.abs(target-a)<1e-6) {out[i]=raw[j-1];continue;}
      if(Math.abs(target-b)<1e-6) {out[i]=raw[j];continue;}
      if(target<a || target>b || b-a>maxGap || !Number.isFinite(raw[j-1]) || !Number.isFinite(raw[j])) continue;
      const delta=((raw[j]-raw[j-1]+540)%360)-180;
      out[i]=raw[j-1]+(target-a)/(b-a)*delta;
    }
    return out;
  }) as AnkleStream['angles'];
}

export function predictSquats(streams: [AnkleStream,AnkleStream], origin: number, range: {a:number;b:number}|null,
    onGrid?: (raw: [Float64Array,Float64Array], offset: number) => void, aligned=false): SquatResult {
  const {shifts,offset}=ankleTimebase(streams,origin,aligned);
  const parts=streams.map((s,j)=>{
    const t=s.t.map(v=>v/1e6+shifts[j]); let a=0,b=t.length;
    if (range) {while(a<b && t[a]+offset<range.a) a++; while(b>a && t[b-1]+offset>range.b) b--;}
    return {t:t.subarray(a,b),values:s.values.subarray(a*6,b*6)};
  });
  if(parts.some(s=>s.t.length<10)) throw new Error('所选范围的双踝有效帧不足。');
  const a=Math.max(...parts.map(s=>s.t[0])), b=Math.min(...parts.map(s=>s.t[s.t.length-1]));
  const n=Math.floor((b-a)*SQUAT_HZ)+1;
  if (!(n>=32)) throw new Error('所选范围内的双踝重叠时间不足。');
  if (n>36000) throw new Error('一次最多预测 12 分钟，请缩小到当前时间窗。');
  const valid=new Uint8Array(n).fill(1), raw=parts.map(({t,values})=>{
    const x=new Float64Array(n*6);
    for(let i=0,j=1;i<n;i++) {
      const s=a+i/SQUAT_HZ;
      while(j<t.length-1 && t[j]<s) j++;
      const delta=t[j]-t[j-1], f=clip((s-t[j-1])/delta,0,1);
      if(delta>.12 && Math.min(Math.abs(s-t[j-1]),Math.abs(s-t[j]))>=1e-6) valid[i]=0;
      for(let c=0;c<6;c++) x[i*6+c]=values[(j-1)*6+c]+f*(values[j*6+c]-values[(j-1)*6+c]);
    }
    return x;
  });
  const phase=new Int8Array(n).fill(-1);
  for(let i=0;i<n;) {
    if(!valid[i]) {i++;continue;}
    let j=i+1; while(j<n && valid[j]) j++;
    const l=j-i;
    if(l>=32) {
      const axes=raw.map(x=>transpose(x.subarray(i*6,j*6),l,6));
      const mag=new Float64Array(l*2); axes.forEach((x,k)=>{for(let t=0;t<l;t++) mag[k*l+t]=Math.sqrt(x[3][t]**2+x[4][t]**2+x[5][t]**2);});
      if(quantile(mag.sort(),.90)<8) phase.fill(0,i,j);
      else phase.set(decodeSquat(emissions(axes)),i);
    }
    i=j;
  }
  const start=a+offset, reps:SquatRep[]=[];
  let boundaries:number[]=[];
  for(let i=1;i<n;i++) {
    const from=phase[i-1],to=phase[i]; if (from===to) continue;
    if (from===0 && to===1) boundaries=[i];
    else if (boundaries.length && from===boundaries.length && to===(from+1)%4) {
      boundaries.push(i);
      if(boundaries.length===4) {reps.push(boundaries.map(j=>a+j/SQUAT_HZ+offset) as SquatRep); boundaries=[];}
    } else boundaries=[];
  }
  const usable=phase.reduce((n,x)=>n+Number(x>=0),0);
  if (!usable) throw new Error('没有连续且足够长的双踝数据可供预测。');
  onGrid?.([raw[0],raw[1]],offset);
  return {start,phase,reps,shifts,valid:usable};
}
