import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import {chromium} from 'playwright-core';
import {D56_CANVAS, D56_POINTS} from '../src/squat-pressure-layout.ts';
import {summarizeSquatPressure} from '../src/squat-pressure-summary.ts';

// node tests/squat-pressure-browser.mjs <local dual-insole + ankles + video ZIP> [entry.html] [screenshot.png] [feet-screenshot.png]
const [capture,entry,screenshot,feetScreenshot]=process.argv.slice(2);
assert.ok(capture,'Pass a local D56 measurement package with mapped clocks, both ankles and a video.');
const browser=await chromium.launch({channel:process.env.VIEWER_BROWSER || 'chrome',headless:true,
  args:['--disable-background-networking','--disable-component-update','--disable-sync']});
try {
  const context=await browser.newContext({offline:true,viewport:{width:1280,height:1050},colorScheme:'light'});
  await context.route(/^(?:https?|wss?):\/\//,route=>route.abort());
  const page=await context.newPage(),errors=[];
  page.on('pageerror',err=>errors.push(err.message));
  await page.addInitScript(()=>{
    const NativeWorker=window.Worker;
    window.Worker=class extends NativeWorker {
      constructor(...args) {
        super(...args);
        this.addEventListener('message',({data})=>{
          if(data.result?.traces?.some(t=>t.values?.peakAdc)) window.shoePressure=data.result;
          if(data.result?.phase instanceof Int8Array) window.squatPacket=data;
        });
      }
    };
  });
  await page.goto((entry ? pathToFileURL(entry) : new URL('../index.html',import.meta.url)).href);
  await page.locator('#file').setInputFiles(capture);
  await page.waitForFunction(()=>!document.querySelector('#bar').hidden && document.querySelector('#busy').hidden);
  assert.equal(await page.locator('#err').textContent(),'');
  await page.getByRole('tab',{name:'深蹲专项'}).click();
  const chart=page.locator('#squatPressureChart'), status=page.locator('#squatPressureStatus');
  const foot=(side,name)=>page.locator(`[data-foot="${side}"] [data-slot="${name}"]`);
  const repRows=page.locator('#squatPressureRepRows tr'), repCell=(i,key)=>repRows.nth(i).locator(`[data-metric="${key}"]`);
  const pathFor=(side)=>page.locator(`[data-foot="${side}"] .squat-pressure-path`);
  const point=(t)=>chart.evaluate((c,t)=>{
    const box=c.getBoundingClientRect(),a=+document.querySelector('#tA').value,b=+document.querySelector('#tB').value;
    return {clientX:box.x+c.clientLeft+46+(c.clientWidth-16-46)*(t-a)/(b-a),clientY:box.top+50};
  },t);
  const hover=async(t)=>chart.dispatchEvent('pointermove',await point(t));
  const pct=(v)=>Number.isFinite(v) ? `${v.toFixed(1)}%` : '—';
  const ratio=(v)=>{const p=Math.round(v*10);return `左 ${pct(p/10)} / 右 ${pct((1000-p)/10)}`;};
  await chart.waitFor({state:'visible'});
  const packet=await page.evaluate(()=>{
    const {traces,share,reference}=window.shoePressure;
    return {traces:traces.map(({foot,t,linked,values})=>({foot,t:[...t],linked:[...linked],
      values:{peakAdc:[...values.peakAdc],sumAdc:[...values.sumAdc]}})),
      share:share && {t:[...share.t],paired:[...share.paired]},reference};
  });
  const peaks=Object.fromEntries(packet.traces.map(t=>[t.foot,t.values.peakAdc.reduce((max,v)=>Number.isFinite(v)?Math.max(max,v):max,0)]));
  const maxTotal=packet.traces.reduce((max,t)=>t.values.sumAdc.reduce((m,v)=>Number.isFinite(v)?Math.max(m,v):m,max),0);
  assert.deepEqual(packet.reference,{peakAdc:Math.max(peaks.left,peaks.right),sumAdc:maxTotal});
  assert.ok(packet.share?.paired.some(Boolean),'Expected paired dual-foot signals without a pressure calibration.');
  assert.match(await status.textContent(),/不等于实际承重比例/);
  assert.equal(await page.locator('#squatRows tr').count(),0,'Pressure must be available without prediction.');
  assert.match(await chart.getAttribute('aria-label'),/单点峰值.*单脚总信号.*左右信号占比/);
  assert.doesNotMatch(await page.locator('.squat-pressure').textContent(),/kPa|法向载荷|导入鞋垫标定/);
  for(const side of ['left','right']) {
    assert.ok(peaks[side]>0);
    assert.equal(await foot(side,'windowPeak').textContent(),pct(peaks[side]/packet.reference.peakAdc*100));
  }
  assert.match(await page.locator('#squatPressureRatio').textContent(),/^左右信号占比：/);
  assert.equal(await repRows.count(),0,'A per-rep table requires predicted repetitions.');
  const unsegmented=await page.evaluate(()=>{
    const s=window.shoePressure.traces.find(t=>t.foot==='left'),i=s.t.findIndex((t,i)=>t>5 && s.linked[i] && s.linked[i+1]
      && [i-1,i,i+1].every(j=>Number.isFinite(s.values.ap[j])) && Math.abs(s.values.ap[i]-s.values.ap[i-1])>.01);
    const t=s.t[i];
    return {t,maxVertices:2+s.t.filter(v=>v>t-1.002 && v<t+.002).length};
  });
  assert.ok(Number.isFinite(unsegmented.t));
  await hover(unsegmented.t);
  assert.match(await page.locator('#squatPressureTime').textContent(),/COP.*1 s/);
  const recent=await pathFor('left').getAttribute('d');
  assert.match(recent,/L/);
  assert.ok(recent.match(/[ML]/g).length<=unsegmented.maxVertices,'Unsegmented COP should only include the last second.');
  await chart.dispatchEvent('pointerleave');
  await page.locator('#squatRun').click();
  await page.waitForFunction(()=>/^预测完成|^预测失败/.test(document.querySelector('#squatStatus').textContent));
  assert.match(await page.locator('#squatStatus').textContent(),/^预测完成/);
  const reps=await page.evaluate(()=>window.squatPacket.result.reps);
  assert.ok(reps.length>0);
  assert.equal(await page.locator('#squatAngleRows tr').count(),reps.length);
  assert.equal(await repRows.count(),reps.length);
  const stats=reps.map(([a,,,b])=>summarizeSquatPressure(packet,a,b));
  for(const [i,[a,,,b]] of reps.entries()) {
    assert.equal(await repCell(i,'time').textContent(),`${a.toFixed(2)}–${b.toFixed(2)}`);
    for(const side of ['left','right']) {
      const stat=stats[i][side];
      assert.ok(Number.isFinite(stat.peakSignalPct) && Number.isFinite(stat.meanTotalPct));
      assert.equal(await repCell(i,side+'Peak').textContent(),pct(stat.peakSignalPct));
      const total=repCell(i,side+'Total');
      assert.ok((await total.textContent()).startsWith(`${pct(stat.meanTotalPct)} / ${pct(stat.peakTotalPct)}`));
      assert.match(await total.locator('small').textContent(),/^有效覆盖 <?\d+%$/);
    }
    const stat=stats[i],cell=repCell(i,'share');
    assert.ok(Number.isFinite(stat.leftPct));
    assert.ok((await cell.textContent()).startsWith(ratio(stat.leftPct)));
    assert.match(await cell.locator('small').textContent(),/^配对覆盖 <?\d+%$/);
  }
  const layout=()=>page.evaluate(()=>{
    const ids=['squatTimeline','squatAngleChart','squatPressureChart','squatRows'];
    const rects=ids.map(id=>document.getElementById(id).getBoundingClientRect());
    return {ordered:rects.every((r,i)=>i===0 || rects[i-1].bottom<r.top),
      aligned:rects.slice(0,3).every(r=>r.x===rects[0].x && r.width===rects[0].width)};
  });
  assert.deepEqual(await layout(),{ordered:true,aligned:true});
  const footprints=page.locator('#squatPressureFeet');
  const coords=D56_POINTS;
  const footprintGeometry=(checkCoverage=false)=>footprints.evaluate((container,coords)=>Array.from(container.children,card=>{
    const map=card.querySelector('.squat-pressure-footprint'), shape=map.querySelector('.squat-pressure-outline');
    const b=shape.getBoundingClientRect(), bounds=card.getBoundingClientRect(), transform=shape.getCTM();
    const drawingToShape=transform.inverse().multiply(map.getCTM()), pins=coords?.[card.dataset.foot];
    const dot=map.querySelector('.squat-pressure-dot');
    const inside=(x,y)=>shape.isPointInFill(new DOMPoint(x,y).matrixTransform(drawingToShape));
    return {width:b.width,height:b.height,tx:transform.a,contained:b.left>=bounds.left && b.right<=bounds.right,
      onlyTrail:map.querySelectorAll('circle').length===1 && map.querySelectorAll('path').length===2,
      envelope:!pins || pins.every(([x,y],i)=>inside(x,y) && pins.slice(i+1).every(([u,v])=>[.25,.5,.75].every(s=>inside(x+(u-x)*s,y+(v-y)*s)))),
      dot:dot.hasAttribute('hidden') ? null : {x:+dot.getAttribute('cx'),y:+dot.getAttribute('cy'),
        inside:inside(+dot.getAttribute('cx'),+dot.getAttribute('cy'))}, left:bounds.left,right:bounds.right,top:bounds.top,bottom:bounds.bottom};
  }),checkCoverage ? coords : null);
  const desktop=await footprintGeometry(true);
  assert.equal(desktop.length,2);
  for(const {height,width,contained,envelope,onlyTrail} of desktop) {
    assert.ok(height>=245 && width/height>.26 && width/height<.30 && contained,'The D56 insole outlines should keep the actual proportions and stay tall.');
    assert.ok(envelope,'All D56 v3 points and intervening COP positions should fit the slim silhouette.');
    assert.ok(onlyTrail,'Only the outline, current trail and cursor endpoint should be drawn.');
  }
  assert.ok(desktop[0].tx>0 && desktop[1].tx<0 && desktop[0].right<desktop[1].left,'Left and right outlines should mirror without changing D56 coordinates.');
  const aligned=async(kind)=>{
    const a=await page.evaluate(k=>['Timeline','Angle','Pressure'].map(s=>{
      const id=s==='Timeline' && k==='Video' ? 'squatVideoCursor' : `squat${s}${k}Cursor`;
      const c=document.getElementById(id);return c.hidden ? null : parseFloat(c.style.left);
    }),kind);
    assert.ok(a.every(x=>x!==null && Math.abs(x-a[0])<.5),`${kind} cursors out of sync`);
    return a[0];
  };
  const target=await page.evaluate(()=>{
    const reps=window.squatPacket.result.reps,left=window.shoePressure.traces.find(s=>s.foot==='left');
    const off=+document.querySelector('#vidOff').value,dur=document.querySelector('#vid').duration;
    for(const [j,rep] of reps.entries()) for(const [i,t] of left.t.entries())
      if(t>Math.max(rep[0]+.25,off+.2) && t<Math.min(rep[1]-.05,off+dur-.5)
          && left.linked[i] && left.linked[i+1] && [i-1,i,i+1].every(k=>Number.isFinite(left.values.ap[k]))) return {j,t,off};
    return null;
  });
  assert.ok(target,'Need a descent with continuous pressure and video coverage.');
  const {j,t,off}=target;
  await hover(t);
  const allWindowTrail=await pathFor('left').getAttribute('d');
  assert.match(allWindowTrail,/L/);
  assert.match(await page.locator('#squatPressureTime').textContent(),new RegExp(`第 ${j+1} 次 COP`));
  const idle=await page.evaluate(()=>{
    const s=window.shoePressure.traces.find(t=>t.foot==='left');
    return s.t.find((t,i)=>t>.05 && s.linked[i] && s.linked[i+1] && [i-1,i,i+1].every(k=>Number.isFinite(s.values.ap[k]))
      && !window.squatPacket.result.reps.some(([a,,,b])=>t>=a-.01 && t<=b+.01));
  });
  assert.ok(Number.isFinite(idle));
  await hover(idle);
  assert.ok(!(await pathFor('left').getAttribute('d')),'No previous repetition trail should be shown between movements.');
  assert.ok(await page.locator('[data-foot="left"] .squat-pressure-dot').isVisible());
  await repRows.nth(j).getByRole('button').dispatchEvent('click');
  assert.ok((await repRows.nth(j).getAttribute('class')).includes('selected'));
  assert.ok((await page.locator('#squatRows tr').nth(j).getAttribute('class')).includes('selected'));
  await page.waitForFunction(a=>Math.abs(+document.querySelector('#tA').value-Math.max(0,a-.2))<.03,reps[j][0]);
  const hit=await point(t);
  await chart.dispatchEvent('pointermove',hit);
  assert.match(await foot('left','cop').textContent(),/\d+% \/ \d+%/);
  assert.ok(await page.locator('[data-foot="left"] .squat-pressure-dot').isVisible());
  const [{dot}]=await footprintGeometry();
  const currentD=await pathFor('left').getAttribute('d');
  assert.match(currentD,/L/);
  assert.ok(currentD.endsWith(`${dot.x.toFixed(3)} ${dot.y.toFixed(3)}`),'The trail should end at the current COP, never include future frames.');
  // The time-window controls format to milliseconds: compare path lengths rather than exact visual coordinates.
  assert.ok(Math.abs(currentD.match(/[ML]/g).length-allWindowTrail.match(/[ML]/g).length)<=1,
    'Zooming should not change the scope from the current rep to the whole window.');
  const [ap,ml]=(await foot('left','cop').textContent()).match(/\d+/g).map(Number);
  assert.ok(dot.inside && Math.abs(dot.x-D56_CANVAS.width*ml/100)<=2.53 && Math.abs(dot.y-D56_CANVAS.height*(1-ap/100))<=8.71,
    'The live marker must use the same D56 drawing coordinates as the outline.');
  assert.match(await page.locator('#squatAngleValues').textContent(),/前后倾 左 [+-−]\d/);
  await aligned('Hover');
  await chart.dispatchEvent('click',hit);
  await chart.dispatchEvent('pointerleave');
  await page.waitForFunction(t=>Math.abs(document.querySelector('#vid').currentTime-t)<.08,t-off);
  const seekX=await aligned('Video');
  await page.evaluate(()=>document.querySelector('#vid').play());
  await page.waitForFunction(t=>document.querySelector('#vid').currentTime>t+.10,t-off);
  await page.evaluate(()=>document.querySelector('#vid').pause());
  assert.ok(await aligned('Video')>seekX);
  const nPixels=()=>chart.evaluate(c=>{
    const g=c.getContext('2d'), image=g.getImageData(0,0,c.width,c.height).data, css=getComputedStyle(c), colors=[];
    for(const side of ['left','right']) colors.push(css.getPropertyValue(`--squat-angle-${side}`).trim().slice(1).match(/../g).map(x=>parseInt(x,16)));
    return colors.map(([r,g,b])=>{
      let n=0;for(let i=0;i<image.length;i+=4) if(Math.abs(image[i]-r)<10 && Math.abs(image[i+1]-g)<10 && Math.abs(image[i+2]-b)<10) n++;
      return n;
    });
  });
  assert.ok((await nPixels()).every(n=>n>12),'Both insole curves should be painted.');
  // Collapse the floating video so it doesn't obscure the pressure preview.
  await page.locator('#vidClose').click();
  assert.ok(await page.locator('#vidPanel').isHidden());
  const previewTime=screenshot || feetScreenshot ? await page.evaluate(([, ,up,end])=>{
    const traces=window.shoePressure.traces;
    return traces[0].t.find(t=>t>=up && t<end-.03 && traces.every(s=>{
      const i=s.t.findIndex(u=>u>=t);
      return s.linked[i] && s.linked[i+1] && [i-1,i,i+1].every(k=>Number.isFinite(s.values.ap[k]));
    }));
  },reps[j]) ?? t : t;
  if(screenshot) {
    await hover(previewTime);
    await page.locator('.squat-pressure').screenshot({path:screenshot,scale:'css'});
  }

  if(feetScreenshot) {
    await page.emulateMedia({colorScheme:'dark'});
    await footprints.scrollIntoViewIfNeeded();
    await hover(previewTime);
    await footprints.screenshot({path:feetScreenshot,scale:'css'});
    await page.emulateMedia({colorScheme:'light'});
  }

  const paired=await page.evaluate(()=>{
    const {share:s}=window.shoePressure,a=+document.querySelector('#tA').value,b=+document.querySelector('#tB').value;
    const hasPair=(t,i)=>t>a+.1 && t<b-.1 && s.linked[i] && s.linked[i+1] && Number.isFinite(s.leftPct[i]);
    // Pointer events may round to the nearest CSS pixel. Prefer a stable interpolation segment.
    const stable=s.t.findIndex((t,i)=>hasPair(t,i) && Math.abs(s.leftPct[i]-s.leftPct[i-1])<.2
      && Math.abs(s.leftPct[i+1]-s.leftPct[i])<.2);
    const i=stable>=0 ? stable : s.t.findIndex(hasPair);
    return {t:s.t[i],expected:s.leftPct[i]};
  });
  assert.ok(paired.t>0,'Expected locally aligned paired signals in the current repetition.');
  await hover(paired.t);
  assert.match(await foot('left','total').textContent(),/^\d+\.\d%$/);
  assert.match(await foot('right','peak').textContent(),/^\d+\.\d%$/);
  const [,l,r]=(await page.locator('#squatPressureRatio').textContent()).match(/左 (\d+\.\d)% \/ 右 (\d+\.\d)%/) || [];
  assert.equal(Math.round((+l+ +r)*10),1000);
  assert.ok(Math.abs(+l-paired.expected)<.3,`Asked for ${paired.t} s (${paired.expected}%), saw ${l}% at ${await page.locator('#squatPressureTime').textContent()}`);
  assert.equal(await repCell(j,'leftPeak').textContent(),pct(stats[j].left.peakSignalPct),'Zoom must not change the shared recording reference.');
  await chart.dispatchEvent('pointerleave');
  await page.emulateMedia({colorScheme:'dark'});
  await page.setViewportSize({width:550,height:900});
  await page.waitForFunction(()=>{
    const c=document.querySelector('#squatPressureChart');return c.width===Math.round(c.clientWidth*window.devicePixelRatio);
  });
  assert.deepEqual(await layout(),{ordered:true,aligned:true});
  assert.ok((await nPixels()).every(n=>n>12),'Both curves should remain visible in dark mode.');
  assert.ok((await footprintGeometry()).every(f=>f.contained && f.height>=185),'Tablet layout should keep both plots large.');
  await page.setViewportSize({width:390,height:900});
  const mobile=await footprintGeometry();
  assert.ok(mobile[0].bottom<mobile[1].top && mobile.every(f=>f.contained && f.height>=185),'Phone layout should stack the cards without shrinking the footprints.');
  assert.deepEqual(errors,[]);
  console.log(`Offline squat pressure: ${reps.length} repetitions; first ${reps[0][0].toFixed(2)}–${reps[0][3].toFixed(2)} s, peak ${pct(stats[0].left.peakSignalPct)}/${pct(stats[0].right.peakSignalPct)}, ${ratio(stats[0].leftPct)}, paired ${Math.round(stats[0].pairedCoverage*100)}%. D56 v3 COP, cursor/video sync and narrow/dark layout OK.`);
} finally {await browser.close();}
