import assert from 'node:assert/strict';
import fs from 'node:fs';
import {crc32, deflateRawSync} from 'node:zlib';
import {pathToFileURL, fileURLToPath} from 'node:url';
import {chromium} from 'playwright-core';
import {activityEngine} from '../src/activity-engine.ts';
import {APP_VER} from '../src/core/constants.ts';
import {ACTIVITY_WALK_RUN_UNCERTAIN, analyzeGaitWindow} from '../src/gait-window.ts';
const fixtures = new URL('./.activity-fixtures/', import.meta.url);
function wtWalkZip(record) {
  const rows = [{type: 'start', phone_ms: 100000, preprocess: 'activity-wt-receipt-v1'}];
  for (const [foot, s] of [record.left, record.right].entries()) for (let i = 0; i < s.us.length;) {
    const us = s.us[i], packets = [];
    while (i < s.us.length && s.us[i] === us) {
      const values = s.axes[i++], packet = Buffer.alloc(20); packet[0] = 0x55; packet[1] = 0x61;
      values.forEach((v, a) => packet.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(v * (a < 3 ? 2048 : 32768 / 2000)))), 2 + a * 2));
      packets.push(packet);
    }
    rows.push({type: 'notify', foot, connection: 1, receipt_ms: 100000 + us / 1000, bytes: Buffer.concat(packets).toString('base64')});
  }
  const name = Buffer.from('session.jsonl'), raw = Buffer.from(rows.map(r => JSON.stringify(r)).join('\n')), compressed = deflateRawSync(raw);
  const local = Buffer.alloc(30), central = Buffer.alloc(46), end = Buffer.alloc(22), checksum = crc32(raw);
  local.writeUInt32LE(0x04034b50); local.writeUInt16LE(20, 4); local.writeUInt16LE(8, 8); local.writeUInt32LE(checksum, 14);
  local.writeUInt32LE(compressed.length, 18); local.writeUInt32LE(raw.length, 22); local.writeUInt16LE(name.length, 26);
  central.writeUInt32LE(0x02014b50); central.writeUInt16LE(20, 4); central.writeUInt16LE(20, 6); central.writeUInt16LE(8, 10);
  central.writeUInt32LE(checksum, 16); central.writeUInt32LE(compressed.length, 20); central.writeUInt32LE(raw.length, 24); central.writeUInt16LE(name.length, 28);
  end.writeUInt32LE(0x06054b50); end.writeUInt16LE(1, 8); end.writeUInt16LE(1, 10);
  end.writeUInt32LE(central.length + name.length, 12); end.writeUInt32LE(local.length + name.length + compressed.length, 16);
  return {name: 'wt-walk-browser.zip', mimeType: 'application/zip', buffer: Buffer.concat([local, name, compressed, central, name, end])};
}
const browser = await chromium.launch({channel: 'msedge', headless: true});
const page = await browser.newPage({viewport: {width: 1280, height: 900}});
page.setDefaultTimeout(120000); const errors = [], requests = [];
page.on('pageerror', e => errors.push(e.message)); page.on('request', r => { if (/^https?:/.test(r.url())) requests.push(r.url()); });
async function importFile(name) {
  await page.locator('#file').setInputFiles(typeof name === 'string' ? fileURLToPath(new URL(name, fixtures)) : name);
  await page.locator('#busy').waitFor({state: 'hidden'}); await page.locator('#bar').waitFor({state: 'visible'});
  assert.equal(await page.locator('#err').textContent(), ''); await page.locator('[data-tab="activity"]').click();
}
async function analyze() {
  await page.locator('#activityRun').click();
  await page.waitForFunction(() => /分析完成|分析失败|计算失败/.test(document.getElementById('activityStatus').textContent));
  assert.match(await page.locator('#activityStatus').textContent(), /分析完成/);
}
async function checkOverview() {
  const overview = await page.locator('#gaitOverviewRows tr').evaluateAll(rows => rows.map(r => [...r.cells].slice(0, 5).map(c => c.textContent)));
  const events = await page.locator('#activityRows tr').evaluateAll(rows => rows.map(r => [...r.cells].map(c => c.textContent)));
  assert.deepEqual(overview.map(row => row[0]), ['步行', '跑步']);
  for (const [state, duration, steps, cadence, coverage] of overview) {
    const segments = events.filter(r => r[4] === state);
    assert.equal(+duration, Math.round(100 * segments.reduce((x, r) => x + Number(r[3]), 0)) / 100);
    const valid = segments.filter(r => r[7] !== '—');
    if (!valid.length) assert.deepEqual([steps, cadence], ['—', '—']);
    else {
      assert.equal(+steps, valid.reduce((x, r) => x + Number(r[7]), 0));
      const seconds = Number(coverage.split(' · ')[0]);
      assert.ok(seconds > 0 && seconds <= +duration);
      assert.ok(+cadence >= 60 * +steps / (seconds + .005) - .05 && +cadence <= 60 * +steps / (seconds - .005) + .05);
    }
  }
  return overview;
}
// 加速度栏整片的平均颜色：底色开关只改背景，不改波形，平均色差就是底色的贡献
async function imuPatch() {
  return page.evaluate(() => {
    const cv = document.querySelector('#imuPanels canvas');
    const g = cv.getContext('2d'), k = cv.width / cv.clientWidth;
    const d = g.getImageData(Math.round(70 * k), Math.round(34 * k), Math.round((cv.clientWidth - 100) * k), Math.round(112 * k)).data;
    let r = 0, gg = 0, b = 0, n = 0;
    for (let i = 0; i < d.length; i += 4) { r += d[i]; gg += d[i + 1]; b += d[i + 2]; n++; }
    return [r / n, gg / n, b / n];
  });
}
const colorDiff = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));
try {
  if (process.env.WT_RUN31_ZIP) await page.addInitScript(() => {
    const NativeWorker = Worker;
    window.Worker = class extends NativeWorker {
      postMessage(data, transfer) {
        if (data.contract?.reviewGait) window.temporalTestInput = structuredClone({inputs: data.inputs, duration: data.duration});
        return super.postMessage(data, transfer);
      }
    };
  });
  await page.goto(pathToFileURL(process.env.VIEWER_TEST_ENTRY || fileURLToPath(new URL('../index.html', import.meta.url))).href);
  assert.ok((await page.locator('#ver').textContent()).includes(APP_VER));
  assert.equal(await page.locator('#activityModel').inputValue(), 'B1Q 实测候选');
  assert.deepEqual(await page.locator('#activityModel option').evaluateAll(options => options.map(o => o.value)),
    ['B1Q 实测候选', 'Moticon 多域时序模型', 'GaitWindow 优化版', '正常走跑提取', '50 Hz 双脚 IMU 四分类', '50 Hz 四分类 + 步态复核 + 转向']);
  assert.equal(await page.locator('#activityIctoVersion').inputValue(), 'legacy');
  if (process.env.D56_COP_ZIP) {
    await importFile({name: 'd56-cop.zip', mimeType: 'application/zip', buffer: fs.readFileSync(process.env.D56_COP_ZIP)});
    assert.equal(await page.locator('#activityCop').isEnabled(), false);
    await page.locator('[data-tab="cop"]').click();
    assert.match(await page.locator('#copScope').textContent(), /未选中步行或跑步段/);
    assert.equal(await page.locator('#copSeek').count(), 0);
    await page.getByRole('button', {name: '前往活动分段'}).click();
    await analyze();
    const copRows = page.locator('#activityRows tr');
    const all = await copRows.evaluateAll(rows => rows.map(r => [...r.cells].map(c => c.textContent)));
    const longest = all.reduce((best, r, i) => /^(步行|跑步)$/.test(r[4]) && (best < 0 || +r[3] > +all[best][3]) ? i : best, -1);
    assert.ok(longest >= 0, JSON.stringify(all));
    const [start, end] = all[longest].slice(1, 3).map(Number);
    await copRows.nth(longest).click(); await page.locator('#activityCop').click();
    const seek = page.locator('#copSeek'); await seek.waitFor({state: 'visible'});
    assert.equal(Number(await seek.getAttribute('min')), 1000 * start);
    assert.equal(Number(await seek.getAttribute('max')), 1000 * end);
    assert.match(await page.locator('#copScope').textContent(), new RegExp(all[longest][4]));
    let summary = [];
    for (const button of await page.locator('.cop-item.play button').filter({hasText: /^CSV$/}).all()) {
      const [file] = await Promise.all([page.waitForEvent('download'), button.click()]);
      const values = fs.readFileSync(await file.path(), 'utf8').trim().split(/\r?\n/).slice(1).map(r => r.split(',').map(Number));
      assert.ok(values.length > 10 && values.every(([, t, x, y]) => t >= start && t < end && Number.isFinite(x) && Number.isFinite(y)));
      const xs = values.map(r => r[2]).sort((a,b) => a-b);
      summary.push([file.suggestedFilename(), new Set(values.map(r => r[0])).size, xs[Math.floor(xs.length / 2)]]);
    }
    assert.equal(summary.length, 2);
    if (process.env.D56_COP_ZIP.endsWith('鞋垫_CoP优化_v4.zip')) {
      assert.ok(summary.every(([, count]) => count >= 25), JSON.stringify(summary));
    }
    await seek.fill(String(1000 * end)); await page.locator('#copPlay').click();
    assert.ok(Number(await seek.inputValue()) >= 1000 * start && Number(await seek.inputValue()) < 1000 * end);
    await page.locator('#copPlay').click();
    await page.getByRole('button', {name: '前往活动分段'}).click();
    const nonGait = all.findIndex(r => !/^(步行|跑步)$/.test(r[4]));
    assert.ok(nonGait >= 0);
    await copRows.nth(nonGait).click(); assert.equal(await page.locator('#activityCop').isEnabled(), false);
    await page.locator('[data-tab="cop"]').click(); assert.equal(await page.locator('#copSeek').count(), 0);
    console.log('PASS selected gait CoP, exact bounds, both CSVs and stale-selection clearing', start, end, summary);
  }
  await importFile('dual_gap.zip'); await analyze();
  assert.match(await page.locator('#activityStatus').textContent(), /WT-offline-B1Q/);
  const rows = await page.locator('#activityRows tr').allTextContents(); assert.ok(rows.length >= 3);
  assert.ok(rows.some(r => r.includes('无法判断'))); assert.ok(rows.some(r => /步行|跑步|其他/.test(r)));
  const gap = await page.locator('#activityRows tr').evaluateAll(rows => rows.some(r => { const t = r.querySelectorAll('td'); return t[4].textContent === '无法判断' && Number(t[1].textContent) <= 6 && Number(t[2].textContent) >= 7; })); assert.ok(gap);
  const gapRow = page.locator('#activityRows tr').filter({has: page.getByRole('cell', {name: '无法判断', exact: true})}).first();
  await gapRow.press('Enter');
  assert.match(await page.locator('#gaitNote').textContent(), /不计算步态指标/);
  assert.ok(await page.locator('#gaitMetrics').isHidden());
  await page.locator('#activityProbability summary').click(); assert.ok(await page.locator('#activityProb').isVisible());
  await page.locator('#activityProbability summary').click();
  await page.locator('#activityRows tr').last().click(); const a = await page.locator('#tA').inputValue();
  await page.locator('#activityImu').click(); assert.equal(await page.locator('#tA').inputValue(), a);
  await page.locator('[data-tab="activity"]').click();
  const downloadPromise = page.waitForEvent('download'); await page.locator('#activityCsv').click(); const download = await downloadPromise; assert.match(download.suggestedFilename(), /WT-offline-B1Q_状态分段\.csv$/);
  await page.selectOption('#activityModel', 'Moticon 多域时序模型');
  assert.equal(await page.locator('#activityRows tr').count(), 0);
  assert.match(await page.locator('#activitySummary').textContent(), /10 Hz 四类概率/);
  assert.equal(await page.locator('#activityStillLegend').textContent(), '■ 静止');
  await analyze();
  assert.match(await page.locator('#activityStatus').textContent(), /WT-offline-Moticon-TAS-v1/);
  assert.ok(await page.locator('#activityRows tr').evaluateAll(rows => rows.some(r => {
    const t = r.cells; return t[4].textContent === '无法判断' && Number(t[1].textContent) <= 6 && Number(t[2].textContent) >= 7;
  })));
  const tasDownload = page.waitForEvent('download'); await page.locator('#activityCsv').click();
  const tasCsv = await tasDownload;
  assert.match(tasCsv.suggestedFilename(), /WT-offline-Moticon-TAS-v1_状态分段\.csv$/);
  assert.match(fs.readFileSync(await tasCsv.path(), 'utf8'), /,WT-offline-Moticon-TAS-v1\b/);
  assert.equal(await page.locator('#activityTurnLLegend').isHidden(), true);
  await page.selectOption('#activityModel', '50 Hz 双脚 IMU 四分类'); await analyze();
  assert.match(await page.locator('#activityStatus').textContent(), /walkpad-activity4-50hz-v2/);
  assert.equal(await page.locator('#activityStillLegend').textContent(), '■ 静止');
  assert.equal(await page.locator('#activityTurnLLegend').isHidden(), true);
  assert.ok(await page.locator('#activityRows tr').evaluateAll(rows => rows.some(r => {
    const t = r.cells; return t[4].textContent === '无法判断' && +t[1].textContent <= 6 && +t[2].textContent >= 7;
  })));
  const annotatedOption = '50 Hz 四分类 + 步态复核 + 转向';
  async function activityCsv() {
    const pending = page.waitForEvent('download'); await page.locator('#activityCsv').click();
    const download = await pending;
    const [header, ...rows] = fs.readFileSync(await download.path(), 'utf8').trimEnd().split(/\r?\n/);
    return {name: download.suggestedFilename(), header, rows: rows.map(r => r.split(','))};
  }
  function stateSpans(rows) {
    const spans = [];
    for (const [a, b, , state] of rows) {
      const prev = spans.at(-1);
      if (prev?.[2] === state && prev[1] === a) prev[1] = b;
      else spans.push([a, b, state]);
    }
    return spans;
  }
  const originalGap = await activityCsv();
  assert.match(originalGap.name, /walkpad-activity4-50hz-v2_状态分段\.csv$/);
  await page.selectOption('#activityModel', annotatedOption); await analyze();
  const annotatedGap = await activityCsv();
  assert.match(annotatedGap.name, /walkpad-activity4-50hz-v2-review-turn_状态分段\.csv$/);
  assert.match(annotatedGap.header, /,turn,walk_review,gait_eligible$/);
  assert.deepEqual(stateSpans(annotatedGap.rows), stateSpans(originalGap.rows));
  assert.ok(annotatedGap.rows.filter(r => r[3] === '无法判断').every(r => r.slice(7).every(v => !v)));
  if (process.env.ACTIVITY4_WALK_RUN_ZIP) {
    await page.selectOption('#activityModel', '50 Hz 双脚 IMU 四分类');
    await importFile({name: '50hz-walk-run.zip', mimeType: 'application/zip', buffer: fs.readFileSync(process.env.ACTIVITY4_WALK_RUN_ZIP)});
    await analyze();
    const rows = await page.locator('#activityRows tr').evaluateAll(rows => rows.map(r =>
      ({state: r.cells[4].textContent, a: +r.cells[1].textContent, b: +r.cells[2].textContent})));
    const activityRows = rows.filter(r => r.state !== '无法判断');
    assert.deepEqual(activityRows.map(r => r.state), ['静止', '步行', '跑步', '步行']);
    const runs = activityRows.filter(r => r.state === '跑步');
    assert.ok(Math.abs(runs[0].a - 92.3) < .11 && Math.abs(runs[0].b - 99.0) < .11, JSON.stringify(runs));
    for (const [a, b] of [[6, 11], [25.5, 27.5], [41, 48], [81.5, 85.5]]) {
      assert.ok(rows.some(r => r.state === '步行' && r.a <= a && r.b >= b), `walk ${a}–${b}`);
    }
    const base = await activityCsv();
    const totals = await page.locator('#activitySummary').textContent();
    await page.selectOption('#activityModel', annotatedOption); await analyze();
    const tagged = await activityCsv();
    assert.deepEqual(stateSpans(tagged.rows), stateSpans(base.rows));
    assert.equal(await page.locator('#activitySummary').textContent(), totals);
    const turnedWalk = tagged.rows.filter(r => r[3] === '步行' && r[7]);
    assert.ok(turnedWalk.length > 0 && turnedWalk.every(r => r[9] === 'no'));
    assert.ok(tagged.rows.some(r => r[3] === '步行' && !r[7] && r[8] && r[9] === 'no'));
    assert.ok(tagged.rows.some(r => r[3] === '步行' && !r[7] && !r[8] && r[9] === 'yes'));
    const overview = await page.locator('#gaitOverviewRows tr').evaluateAll(rows => rows.map(r => Number(r.cells[1].textContent)));
    for (const [i, state] of ['步行', '跑步'].entries()) {
      const included = tagged.rows.filter(r => r[3] === state && r[9] === 'yes').reduce((sum, r) => sum + Number(r[2]), 0);
      assert.ok(Math.abs(overview[i] - included) < .011);
    }
    const turnRow = page.locator('#activityRows tr').filter({hasText: /步行 (左转|右转|转身)/}).first();
    assert.equal(await turnRow.locator('td').nth(7).textContent(), '—');
    await turnRow.click();
    assert.match(await page.locator('#gaitNote').textContent(), /不纳入常规步态统计/);
    assert.ok(await page.locator('#gaitMetrics').isHidden());
    console.log('PASS Activity4 user recording:', totals, 'running only at', runs, 'turned walk',
      turnedWalk.map(r => r.slice(0, 2)), 'gait overview seconds', overview);
    await importFile('dual_gap.zip');
  }
  if (process.env.ACTIVITY4_UTURN_ZIP) {
    await page.selectOption('#activityModel', '50 Hz 双脚 IMU 四分类');
    await importFile({name: '50hz-uturn.zip', mimeType: 'application/zip', buffer: fs.readFileSync(process.env.ACTIVITY4_UTURN_ZIP)});
    await analyze();
    const base = await activityCsv();
    await page.selectOption('#activityModel', annotatedOption); await analyze();
    const tagged = await activityCsv();
    assert.deepEqual(stateSpans(tagged.rows), stateSpans(base.rows));
    const turns = [];
    for (const [a, b, , state, , , , turn, , eligible] of tagged.rows) {
      if (!turn) continue;
      if (state === '步行' || state === '跑步') assert.equal(eligible, 'no');
      const prev = turns.at(-1);
      if (prev?.[1] === a && prev[2] === turn) prev[1] = b;
      else turns.push([a, b, turn]);
    }
    assert.deepEqual(turns.map(r => r[2]), ['左转', '左转', '左转', '右转', '左转', '左转']);
    assert.ok(turns.every(([a, b]) => Number(b) - Number(a) < 2), JSON.stringify(turns));
    assert.deepEqual(turns[4].slice(0, 2).map(Number), [63.6, 65.1]);
    await importFile('dual_gap.zip');
    console.log('PASS Activity4 straight-path U-turn replay: unchanged base, six compact turns, gait exclusion');
  }
  await page.selectOption('#activityModel', '正常走跑提取'); await analyze();
  assert.match(await page.locator('#activityStatus').textContent(), /WT-gait-extraction-v1/);
  assert.equal(await page.locator('#activityTurnLLegend').isVisible(), true);
  assert.ok(await page.locator('#activityRows tr').evaluateAll(rows => rows.some(r => {
    const t = r.cells; return t[4].textContent === '无法判断' && Number(t[1].textContent) <= 6 && Number(t[2].textContent) >= 7;
  })));
  const selectionCSV = page.waitForEvent('download'); await page.locator('#activityCsv').click();
  assert.match(fs.readFileSync(await (await selectionCSV).path(), 'utf8'), /model,walk_review/);
  for (const [file, abnormal] of [[process.env.D56_NORMAL_WALK_ZIP, false], [process.env.D56_ABNORMAL_WALK_ZIP, true]]) {
    if (!file) continue;
    await importFile({name: 'd56-walk.zip', mimeType: 'application/zip', buffer: fs.readFileSync(file)}); await analyze();
    const rows = await page.locator('#activityRows tr').evaluateAll(rs => rs.map(r => [...r.cells].map(c => c.textContent)));
    const select = label => rows.filter(r => r[4] === label).map(r => r.slice(1, 3).map(Number));
    assert.deepEqual(select('左转'), abnormal ? [[23.2, 26.7]] : [[14.9, 19]]);
    assert.deepEqual(select('右转'), abnormal ? [[60, 83.1]] : []);
    assert.deepEqual(select('步行'), abnormal ? [] : [[1.7, 14.9], [19, 30.5]]);
    assert.deepEqual(select('跑步'), []);
    assert.ok(rows.some(r => r[4] === '其他' && r[5] === '—' && r[8].includes('摆动偏低')));
    const overview = await checkOverview();
    assert.deepEqual(overview[1].slice(1, 4), ['0.00', '—', '—']);
    if (abnormal) assert.deepEqual(overview[0].slice(1, 4), ['0.00', '—', '—']);
    else assert.equal(overview[0][1], '24.70');
    console.log('PASS D56 gait overview', JSON.stringify(overview));
  }
  if (process.env.D56_EXTRACTION_RUN_ZIP) {
    const buffer = fs.readFileSync(process.env.D56_EXTRACTION_RUN_ZIP);
    await importFile({name: 'd56-run.zip', mimeType: 'application/zip', buffer}); await analyze();
    const segments = await page.locator('#activityRows tr').evaluateAll(rows => rows.filter(r => r.cells[4].textContent === '跑步').map(r => Number(r.cells[3].textContent)));
    assert.deepEqual(segments, [60.6]);
  }
  if (process.env.D56_COMPLEX_GAIT_ZIP) {
    const buffer = fs.readFileSync(process.env.D56_COMPLEX_GAIT_ZIP);
    await importFile({name: 'd56-complex-gait.zip', mimeType: 'application/zip', buffer}); await analyze();
    const cases = [[3, '步行', 2.5, '支撑'], [15, '跑步', 72.2, '腾空'], [17, '跑步', 84.5, '腾空']];
    const outputs = [];
    for (const [index, state, start, metric] of cases) {
      const event = page.locator('#activityRows tr').nth(index - 1);
      assert.equal(await event.locator('td').nth(4).textContent(), state);
      assert.equal(Number(await event.locator('td').nth(1).textContent()), start);
      await event.click();
      const cells = await event.locator('td').allTextContents();
      assert.deepEqual((await page.locator('#gaitCompare strong').allTextContents()).slice(0, 3), [cells[3], cells[7], cells[6]]);
      const states = await page.locator('#gaitPhaseRows th').allTextContents();
      assert.deepEqual(states, ['仅左脚支撑', '仅右脚支撑', '双脚支撑', '双脚腾空']);
      const values = await page.locator('#gaitPhaseRows td').allTextContents();
      const coverage = await page.locator('#gaitPhaseCoverage').textContent();
      const known = Number(/双脚可判定 ([\d.]+)/.exec(coverage)[1]);
      const totals = values.map(v => /^([\d.]+) s · ([\d.]+)%$/.exec(v).slice(1).map(Number));
      assert.ok(Math.abs(totals.reduce((sum, [s]) => sum + s, 0) - known) <= .026);
      assert.ok(Math.abs(totals.reduce((sum, [, pct]) => sum + pct, 0) - 100) <= .201);
      assert.equal(cells[9], `双脚${metric} ${values[metric === '支撑' ? 2 : 3]}`);
      outputs.push({index, coverage, values});
    }
    console.log('PASS D56 complex segment phase estimates', JSON.stringify(outputs));
    const legacy = await checkOverview();
    assert.deepEqual(legacy.map(r => r[1]), ['22.20', '21.00']);
    await page.selectOption('#activityIctoVersion', 'running');
    const experimental = await checkOverview();
    assert.deepEqual(experimental[0], legacy[0]);
    assert.equal(experimental[1][1], legacy[1][1]);
    await page.selectOption('#activityIctoVersion', 'legacy');
    console.log('PASS D56 mixed walking/running overview', JSON.stringify(legacy));
  }
  await importFile('dual_gap.zip');
  console.log('PASS raw Moticon TAS and gait extraction options, gap isolation, model-tagged CSV and optional D56 replay');
  await page.selectOption('#activityModel', 'GaitWindow 优化版');
  assert.ok(await page.locator('#activityStillLegend').isVisible()); await analyze();
  assert.match(await page.locator('#activityStatus').textContent(), /WT-offline-GaitWindow-Refined/);
  assert.ok(await page.locator('#activityRows tr').evaluateAll(rows => rows.some(r => {
    const t = r.cells; return t[4].textContent === '无法判断' && Number(t[1].textContent) <= 6 && Number(t[2].textContent) >= 7;
  })));
  const refinedCsv = page.waitForEvent('download'); await page.locator('#activityCsv').click();
  const downloadRefined = await refinedCsv;
  assert.match(downloadRefined.suggestedFilename(), /WT-offline-GaitWindow-Refined_状态分段\.csv$/);
  assert.match(fs.readFileSync(await downloadRefined.path(), 'utf8'), /model,state_source/);
  await page.selectOption('#activityModel', 'B1Q 实测候选'); await analyze();
  assert.ok(await page.locator('#activityStillLegend').isHidden());
  assert.deepEqual(await page.locator('#activityRows tr').allTextContents(), rows);
  console.log('PASS frozen GaitWindow via Blob worker, full-window gap protection, model CSV and B1Q restoration');
  await page.locator('[data-tab="imu"]').click();
  assert.match(await page.locator('#imuSegNote').textContent(), /B1Q · ■步行/);
  assert.ok(await page.locator('#imuSegOn').isEnabled());
  const tinted = await imuPatch();
  await page.locator('#imuSegOn').click();
  assert.equal(await page.locator('#imuSegOn').evaluate(b => b.classList.contains('on')), false);
  assert.match(await page.locator('#imuSegNote').textContent(), /底色已关/);
  const plain = await imuPatch();
  assert.ok(colorDiff(tinted, plain) > 2, `背景色没画上：${tinted} vs ${plain}`);
  await page.locator('#imuSegOn').click();
  assert.ok(colorDiff(tinted, await imuPatch()) < .5, '关掉再打开应回到同一张图');
  console.log('PASS IMU waveform background comes from the shared segmentation result and can be switched off');
  await page.locator('[data-tab="activity"]').click();
  await page.selectOption('#activityModel', 'Moticon 多域时序模型');
  await page.locator('[data-tab="imu"]').click();
  assert.equal(await page.locator('#imuSegNote').textContent(), '未分析');
  assert.equal(await page.locator('#imuSegOn').isEnabled(), false);
  await page.locator('#imuRun').click();
  await page.waitForFunction(() => /Moticon-TAS-v1 · ■步行/.test(document.getElementById('imuSegNote').textContent));
  await page.locator('[data-tab="activity"]').click();
  assert.match(await page.locator('#activityStatus').textContent(), /WT-offline-Moticon-TAS-v1/); assert.ok(await page.locator('#activityRows tr').count() > 0);
  await page.selectOption('#activityModel', 'B1Q 实测候选'); await analyze();
  console.log('PASS analyze from the IMU tab fills the same result into the segmentation table, no second computation');
  await page.setViewportSize({width: 390, height: 844}); assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true); await page.setViewportSize({width: 1280, height: 900});
  console.log('PASS file:// without browser security flags: Blob worker inference, gap isolation, timeline/table, shared window, CSV, mobile layout');
  await importFile('single.zip'); assert.equal(await page.locator('#activityRows tr').count(), 0); await page.locator('#activityRun').click(); assert.match(await page.locator('#activityStatus').textContent(), /必须同时/);
  await importFile('dual.parquet'); await analyze(); assert.ok(await page.locator('#activityRows tr').count() > 0);
  console.log('PASS missing foot refusal, file-change reset, legacy physical parquet');
  await importFile('wt_local.zip'); await analyze(); assert.match(await page.locator('#activityStatus').textContent(), /手机接收时钟/); assert.ok(await page.locator('#activityRows tr').count() > 0);
  console.log('PASS independent WT local Android ZIP + JSONL raw-notification decoding and receipt-timing disclaimer');
  await importFile('batched.zip'); await analyze();
  const batched = await page.locator('#activityStatus').textContent();
  assert.ok(Number(/有效 ([\d.]+) s/.exec(batched)[1]) > 15, batched);
  assert.ok(await page.locator('#activityRows tr').evaluateAll(rows => rows.some(r => r.querySelectorAll('td')[4].textContent !== '无法判断')));
  console.log('PASS two inertial frames sharing one notification timestamp stay one continuous signal');
  await importFile('long_dual.zip'); await analyze();
  const end = await page.locator('#activityRows tr').last().locator('td').nth(2).textContent(); assert.ok(Number(end) > 600, end);
  await page.locator('#activityRun').click(); await page.locator('#activityCancel').click(); assert.match(await page.locator('#activityStatus').textContent(), /取消/);
  await page.locator('#activityRun').click(); await importFile('single.zip'); await page.waitForTimeout(500); assert.equal(await page.locator('#activityRows tr').count(), 0);
  assert.equal(await page.locator('#activityRun').isEnabled(), true);
  console.log('PASS long lazy package full-duration inference beyond initial 60s, cancellation and stale-result isolation');
  const walk = new URL('wt_dev_walk_frames.json', fixtures);
  if (fs.existsSync(walk)) {
    await importFile(wtWalkZip(JSON.parse(fs.readFileSync(walk))['64步.zip']));
    await page.locator('[data-tab="axis"]').click();
    const axes = page.locator('#axisPanels .side');
    await axes.last().locator('.meta').getByText(/双侧状态检测.*段步行候选/).waitFor();
    assert.equal(await axes.count(), 2);
    for (const card of await axes.all()) {
      assert.match(await card.locator('.meta').textContent(), /双侧状态检测.*[1-9]\d* 段步行候选/);
      assert.equal(await card.getByRole('combobox').count(), 0);
    }
    assert.equal(await page.locator('#activityRows tr').count(), 0);
    console.log('PASS offline axis uses existing dual-side walk extraction without updating the activity page:', await axes.locator('.axsum').allTextContents());
    await page.locator('[data-tab="activity"]').click(); await analyze();
    const heads = await page.locator('#activityRows').evaluate(tb => [...tb.closest('table').tHead.rows[0].cells].map(c => c.textContent));
    assert.equal(heads.length, 10);
    assert.match(heads[6], /平均步频/); assert.match(heads[7], /有效步数/); assert.match(heads[9], /双脚支撑 \/ 腾空/);
    const walkRows = page.locator('#activityRows tr').filter({has: page.getByRole('cell', {name: '步行', exact: true})});
    assert.ok(await walkRows.evaluateAll(rs => rs.some(r => Number(r.cells[6].textContent) > 0 && /^\d+$/.test(r.cells[7].textContent) && Number(r.cells[7].textContent) > 0)), '步行段应显示平均步频与有效步数');
    const overviewText = await page.locator('#activityGaitOverview').textContent();
    for (const token of ['步行', '有效计步', '步/分', '接地均值']) assert.ok(overviewText.includes(token), `步行概览缺少「${token}」：${overviewText}`);
    assert.match(await page.locator('#gaitTitle').textContent(), /步行/);
    const metricText = await page.locator('#gaitMetricRows').textContent();
    assert.match(metricText, /接地均值[（(]ms[）)]/);
    assert.match(metricText, /平均三轴加速度合量峰值[（(]g[）)]/);
    const contact = await page.locator('#gaitMetricRows tr').filter({hasText: /接地均值/}).locator('td, th').allTextContents();
    assert.ok(contact.slice(1).some(v => /\d/.test(v)), `接地均值缺少数值：${contact}`);
    assert.ok(contact.slice(1).every(v => /\d/.test(v) || /—|未提供/.test(v)), `接地均值列应为数值或未提供：${contact}`);
    const impact = await page.locator('#gaitMetricRows tr').filter({hasText: /平均三轴加速度合量峰值/}).locator('.gait-sample').allTextContents();
    assert.ok(impact.some(v => /^[1-9]\d* \/ \d+ 步$/.test(v)), `实际步行应有可报告的均值：${impact}`);
    assert.match(await page.locator('#gaitPhaseCoverage').textContent(), /双脚只有手机接收时间/);
    assert.deepEqual(await page.locator('#gaitPhaseRows td').allTextContents(), ['—', '—', '—', '—']);
    await page.locator('#gaitCoverage summary').click();
    assert.ok(await page.locator('#gaitCountable').isVisible());
    assert.match((await page.locator('#gaitCountable').textContent()).trim(), /^可计步区间/);
    await page.locator('#gaitSteps summary').click();
    assert.ok(await page.locator('#gaitSteps').evaluate(el => el.open));
    const stepHeads = await page.locator('#gaitSteps thead tr:last-child th:visible').allTextContents();
    assert.equal(stepHeads.length, 9);
    assert.equal(await page.locator('#gaitSteps [data-gait-device]:visible').count(), 0);
    assert.match(stepHeads[0], /序号|步次/);
    for (const token of ['脚', 'IC', 'TO', '接地', '三轴加速度合量', '峰值时刻']) assert.ok(stepHeads.join('|').includes(token), `逐步明细缺少「${token}」列：${stepHeads}`);
    assert.match(stepHeads.join('|'), /触地.*俯仰/); assert.match(stepHeads.join('|'), /离地.*俯仰/);
    assert.doesNotMatch(stepHeads.join('|'), /COP/);
    const steps = page.locator('#gaitStepRows tr');
    await steps.first().waitFor({state: 'visible'});
    assert.ok(await steps.evaluateAll(rs => rs.every(r => r.cells[2].querySelector('button'))), '每一步都应有 IC 时间按钮');
    assert.ok(await steps.evaluateAll(rs => rs.every(r => r.cells[3].querySelector('button') || r.cells[3].textContent.trim() === '—')), 'TO 列应为按钮或 —');
    const completeRows = steps.filter({has: page.locator('td:nth-child(4) button')});
    assert.ok(await completeRows.count(), '步行段应有带 TO 的完整步');
    const icTexts = await completeRows.locator('td:nth-child(3) button').allTextContents();
    const completeIndex = icTexts.findIndex(v => Number(v.replace(/[^\d.]/g, '')) > 500);
    const complete = completeRows.nth(completeIndex >= 0 ? completeIndex : 0);
    const cells = complete.locator('td, th');
    assert.equal(await cells.count(), 9);
    assert.match(await cells.nth(0).textContent(), /^\d+$/);
    const side = (await cells.nth(1).textContent()).trim(); assert.match(side, /^(左脚|右脚)$/);
    assert.equal(await cells.nth(2).locator('button').count(), 1);
    assert.equal(await cells.nth(3).locator('button').count(), 1, '只对带 TO 的完整步要求接地 ms');
    const grounded = parseFloat(await cells.nth(4).textContent());
    assert.ok(grounded > 20, `接地 ms 应为毫秒量级：${await cells.nth(4).textContent()}`);
    assert.ok(await steps.evaluateAll(rs => [5, 6].every(i => rs.some(r => /^-?\d+\.\d$/.test(r.cells[i].textContent.trim())))), '应显示有参考的触地和离地俯仰');
    assert.ok(await steps.evaluateAll(rs => rs.every(r => {
      const peak = r.cells[8].querySelector('button'), value = r.cells[7].textContent.trim();
      return peak ? Number(value) >= 0 && Number(peak.textContent) >= Number(r.cells[2].textContent) &&
        Number(peak.textContent) - Number(r.cells[2].textContent) <= 100 : value.startsWith('— ');
    })), '三轴合量峰与峰时必须成对，且处于触地窗口');
    const icMs = Number((await cells.nth(2).locator('button').textContent()).replace(/[^\d.]/g, '')); assert.ok(icMs > 0);
    await cells.nth(2).locator('button').click();
    assert.ok(await page.locator('#viewImu').isVisible());
    assert.match(await page.locator('#gaitWaveNote').textContent(), new RegExp(side));
    assert.ok(await page.locator('#gaitWaveFocus').isVisible());
    const icT = Number(await page.locator('#tA').inputValue()), icB = Number(await page.locator('#tB').inputValue()), icSec = icMs / 1000;
    assert.ok(Math.abs(icT - (icSec - .45)) < .12 && Math.abs(icB - (icSec + .45)) < .12, `IC ${icMs} ms 未按 ±0.45 s 定位：${icT}–${icB}`);
    await page.locator('#gaitWaveClear').click(); assert.ok(await page.locator('#gaitWaveFocus').isHidden());
    await page.locator('[data-tab="activity"]').click();
    const toButton = cells.nth(3).locator('button');
    const toMs = Number((await toButton.textContent()).replace(/[^\d.]/g, '')); assert.ok(toMs > 0);
    await toButton.click();
    assert.ok(await page.locator('#viewImu').isVisible());
    assert.match(await page.locator('#gaitWaveNote').textContent(), new RegExp(side));
    assert.ok(await page.locator('#gaitWaveFocus').isVisible());
    const toT = Number(await page.locator('#tA').inputValue()), toB = Number(await page.locator('#tB').inputValue()), toSec = toMs / 1000;
    assert.ok(Math.abs(toT - (toSec - .45)) < .12 && Math.abs(toB - (toSec + .45)) < .12, `TO ${toMs} ms 未按 ±0.45 s 定位：${toT}–${toB}`);
    await page.locator('#gaitWaveClear').click(); assert.ok(await page.locator('#gaitWaveFocus').isHidden());
    await page.locator('[data-tab="activity"]').click();
    const peakButton = steps.locator('td:nth-child(9) button').first();
    assert.ok(await peakButton.count(), '步行应有可定位的三轴合量峰');
    const peakMs = Number(await peakButton.textContent());
    await peakButton.click();
    assert.ok(await page.locator('#viewImu').isVisible());
    assert.match(await page.locator('#gaitWaveNote').textContent(), /三轴加速度合量峰值/);
    assert.ok(Math.abs(Number(await page.locator('#tA').inputValue()) - (peakMs / 1000 - .45)) < .12);
    await page.locator('#gaitWaveClear').click(); await page.locator('[data-tab="activity"]').click();
    const before = {
      title: await page.locator('#gaitTitle').textContent(), overview: await page.locator('#activityGaitOverview').textContent(),
      metrics: await page.locator('#gaitMetricRows').textContent(), steps: await page.locator('#gaitStepRows').innerText(),
      countable: await page.locator('#gaitCountable').textContent()
    };
    await page.selectOption('#activityIctoVersion', 'running');
    assert.match(await page.locator('#activityStatus').textContent(), /分析完成/);
    assert.match(await page.locator('#activityIctoHint').textContent(), /步行沿用原版/);
    assert.ok(await page.locator('#gaitSteps').evaluate(el => el.open), '切换版本不应收起步行逐步明细');
    assert.ok(await page.locator('#gaitCoverage').evaluate(el => el.open), '切换版本不应收起可计步覆盖');
    assert.equal(await page.locator('#gaitTitle').textContent(), before.title);
    assert.equal(await page.locator('#activityGaitOverview').textContent(), before.overview);
    assert.equal(await page.locator('#gaitMetricRows').textContent(), before.metrics);
    assert.equal(await page.locator('#gaitStepRows').innerText(), before.steps);
    assert.equal(await page.locator('#gaitCountable').textContent(), before.countable);
    await page.selectOption('#gaitStepSide', 'left');
    assert.ok(await steps.count());
    assert.ok(await steps.evaluateAll(rs => rs.every(r => r.cells[1].textContent === '左脚')));
    await page.selectOption('#gaitStepSide', 'right');
    assert.ok(await steps.count());
    assert.ok(await steps.evaluateAll(rs => rs.every(r => r.cells[1].textContent === '右脚')));
    await page.selectOption('#gaitStepSide', 'both');
    await page.selectOption('#activityIctoVersion', 'legacy');
    await page.setViewportSize({width: 390, height: 844});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.setViewportSize({width: 1280, height: 900});
    await page.selectOption('#activityModel', 'Moticon 多域时序模型');
    assert.ok(await page.locator('#gaitDetail').isHidden()); assert.ok(await page.locator('#activityGaitOverview').isHidden());
    console.log('PASS real WT channels via ZIP, walking overview and step rows, IC/TO waveform focus, running-version isolation, mobile and model reset');
  } else console.log('SKIP local WT gait fixture not available');
  if (process.env.WT_RUN_ZIP) {
    await page.setViewportSize({width: 1280, height: 900});
    await page.selectOption('#activityModel', 'B1Q 实测候选');
    await importFile({
      name: 'wt-run.zip', mimeType: 'application/zip', buffer: fs.readFileSync(process.env.WT_RUN_ZIP)
    });
    await analyze();
    const runOverview = await page.locator('#activityGaitOverview').textContent();
    for (const token of ['步行', '有效计步', '步/分', '接地均值']) assert.ok(runOverview.includes(token), `跑步录制中的步行概览缺少「${token}」：${runOverview}`);
    const overview = page.locator('#gaitOverviewRows tr');
    assert.equal(await overview.count(), 2);
    const runCells = overview.nth(1).locator('td');
    assert.ok(Number(await runCells.nth(1).textContent()) > 0, '跑步概览应有有效步数');
    assert.ok(Number(await runCells.nth(2).textContent()) > 0, '跑步概览应有平均步频');
    assert.equal(await overview.locator('progress').count(), 2);
    const overviewWalk = await overview.first().innerText(), overviewRun = await overview.nth(1).innerText();
    const runs = page.locator('#activityRows tr').filter({has: page.getByRole('cell', {name: '跑步', exact: true})});
    assert.ok(await runs.evaluateAll(rs => rs.some(r => Number(r.cells[6].textContent) > 0 && Number(r.cells[7].textContent) > 0)));
    assert.ok(await page.locator('#activityRows tr').evaluateAll(rs => rs.filter(r => !['步行', '跑步'].includes(r.cells[4].textContent)).every(r => r.cells[6].textContent === '—' && r.cells[7].textContent === '—')), '非步态段不显示步频或步数');
    const walkRows = page.locator('#activityRows tr').filter({has: page.getByRole('cell', {name: '步行', exact: true})});
    assert.ok(await walkRows.evaluateAll(rs => rs.some(r => Number(r.cells[6].textContent) > 0 && /^\d+$/.test(r.cells[7].textContent) && Number(r.cells[7].textContent) > 0)), '步行段应显示平均步频与有效步数');
    const lengths = await runs.locator('td:nth-child(4)').allTextContents();
    assert.ok(lengths.length, '录制应有跑步段');
    await runs.nth(lengths.map(Number).indexOf(Math.max(...lengths.map(Number)))).click();
    const title = await page.locator('#gaitTitle').textContent(), duration = await page.locator('#activitySummary').textContent();
    assert.match(title, /跑步/);
    assert.ok(await page.locator('#gaitSteps').isVisible());
    assert.ok(await page.locator('#gaitMetrics').isVisible());
    assert.match(await page.locator('#gaitNote').textContent(), /原版 IC\/TO/);
    const metrics = page.locator('#gaitMetricRows');
    for (const label of ['接地均值', '触地设备俯仰', '离地设备俯仰', '三轴加速度合量', '峰值延迟']) assert.ok((await metrics.innerText()).includes(label));
    const legacyStats = await metrics.innerText();
    await page.locator('#gaitSteps summary').click();
    const runSteps = page.locator('#gaitStepRows tr');
    await runSteps.first().waitFor();
    await page.selectOption('#gaitStepSide', 'right');
    assert.ok(await runSteps.count());
    assert.ok(await runSteps.evaluateAll(rs => rs.every(r => r.cells[1].textContent === '右脚')));
    const originalSteps = await page.locator('#gaitStepRows').innerText();
    const reportedAngles = await runSteps.evaluateAll(rs => rs.flatMap(r => [5, 6].map(i => r.cells[i].textContent.trim())));
    assert.ok(reportedAngles.every(v => /^-?\d+\.\d$|^— .+/.test(v)));
    assert.ok(reportedAngles.some(v => /^— .+/.test(v)), '缺测角度应显示具体原因');
    const angleCounts = async () => runSteps.evaluateAll(rs => [5, 6, 7].map(i => rs.filter(r => /^-?\d/.test(r.cells[i].textContent.trim())).length));
    console.log('WT running original first right-foot page: IC pitch / TO pitch / impact counts', await angleCounts());
    await page.locator('#gaitPeriods summary').click(); await page.selectOption('#gaitSide', 'right');
    assert.ok(await page.locator('#gaitPeriods').evaluate(el => el.open));
    const events = page.locator('#gaitPeriodRows'), original = await events.innerText();
    assert.match(original, /\d/);
    await page.selectOption('#activityIctoVersion', 'running');
    assert.equal(await page.locator('#gaitTitle').textContent(), title);
    assert.equal(await page.locator('#activitySummary').textContent(), duration);
    assert.equal(await page.locator('#gaitSide').inputValue(), 'right');
    assert.ok(await page.locator('#gaitMetrics').isVisible());
    assert.match(await page.locator('#gaitNote').textContent(), /跑步实验版 IC\/TO/);
    assert.equal(await overview.first().innerText(), overviewWalk);
    assert.notEqual(await overview.nth(1).innerText(), overviewRun);
    assert.notEqual(await metrics.innerText(), legacyStats);
    assert.notEqual(await page.locator('#gaitStepRows').innerText(), originalSteps);
    assert.equal(await page.locator('#gaitStepSide').inputValue(), 'right');
    const updated = await events.innerText(); assert.notEqual(updated, original);
    const changed = page.locator('#gaitPeriodRows tr').filter({hasText: /TO 改选/}).first();
    await changed.getByRole('button').click();
    assert.match(await page.locator('#gaitWaveNote').textContent(), /右脚.*跑步实验版.*TO 改选/);
    await page.locator('[data-tab="activity"]').click();
    await page.selectOption('#activityIctoVersion', 'legacy'); assert.equal(await events.innerText(), original);
    assert.ok(await page.locator('#gaitWaveFocus').isHidden());
    assert.equal(await metrics.innerText(), legacyStats);
    console.log(`PASS real mixed gait: ${title}\n步行概览 ${overviewWalk}\n跑步概览 ${overviewRun}`);
    await page.selectOption('#activityModel', 'Moticon 多域时序模型');
    await analyze();
    const rows = await page.locator('#activityRows tr').evaluateAll(rs => rs.map(r => [...r.cells].map(td => td.textContent)));
    const seconds = (state, from, to) => rows.filter(r => r[4] === state).reduce((sum, r) =>
      sum + Math.max(0, Math.min(to, Number(r[2])) - Math.max(from, Number(r[1]))), 0);
    assert.equal(seconds('跑步', 99, 139), 40);
    assert.equal(seconds('步行', 40, 90), 50);
    assert.equal(seconds('步行', 153, 160), 7);
    assert.equal(seconds('静止', 24, 29), 5);
    const still = page.locator('#activityRows tr').filter({has: page.getByRole('cell', {name: '静止', exact: true})}).first();
    assert.match(await still.locator('td').nth(5).textContent(), /%$/);
    assert.equal(await still.locator('td').nth(6).textContent(), '—'); assert.equal(await still.locator('td').nth(7).textContent(), '—');
    await still.click(); assert.match(await page.locator('#gaitNote').textContent(), /模型判为静止.*不计算步态/);
    console.log('PASS Moticon TAS real mixed recording: steady run 40/40 s, walks 50/50 s and 7/7 s, quiet 5/5 s');
  }
  if (process.env.WT_RUN51_ZIP || process.env.WT_WALK_STOP_WALK_ZIP || fs.existsSync(walk)) {
    await page.setViewportSize({width: 1280, height: 900});
    const labels = () => page.locator('#activityRows tr').evaluateAll(rows => rows.map(row => [...row.cells].map(c => c.textContent)));
    const seconds = (rows, label, a, z) => rows.filter(r => r[4] === label).reduce((n, r) => n + Math.max(0, Math.min(z, Number(r[2])) - Math.max(a, Number(r[1]))), 0);
    const upload = (file, name) => ({name, mimeType: 'application/zip', buffer: fs.readFileSync(file)});
    const tas = async input => {
      await page.selectOption('#activityModel', 'Moticon 多域时序模型');
      await importFile(input); await analyze();
      assert.match(await page.locator('#activityStatus').textContent(), /WT-offline-Moticon-TAS-v1/);
      return labels();
    };
    const opt = async input => {
      await page.selectOption('#activityModel', 'GaitWindow 优化版');
      await importFile(input); await analyze();
      assert.match(await page.locator('#activityStatus').textContent(), /WT-offline-GaitWindow-Refined/);
      return labels();
    };
    if (process.env.WT_RUN51_ZIP) {
      const run51 = upload(process.env.WT_RUN51_ZIP, 'run51.zip');
      assert.equal(seconds(await opt(run51), '跑步', 40, 70), 30);
      const result = await tas(run51);
      console.log('WT Run51 mixed replay: Moticon TAS run in 40–70 s', seconds(result, '跑步', 40, 70));
    }
    if (process.env.WT_WALK_STOP_WALK_ZIP) {
      const input = upload(process.env.WT_WALK_STOP_WALK_ZIP, 'walk-stop-walk.zip');
      const improved = await opt(input), still = '静止（IMU 判据）';
      assert.equal(seconds(improved, '跑步', 0, Infinity), 0);
      assert.ok(seconds(improved, still, 46.2, 77.3) > 30);
      assert.ok(seconds(improved, '步行', 77.5, 81.1) > 0);
      assert.ok(seconds(improved, '无法判断', 0, Infinity) < .7);
      const row = page.locator('#activityRows tr').filter({has: page.getByRole('cell', {name: still, exact: true})}).first();
      await row.click(); assert.match(await page.locator('#gaitNote').textContent(), /不计算步态指标/);
      assert.equal(await row.locator('td').nth(5).textContent(), '—');
      const sent = page.waitForEvent('download'); await page.locator('#activityCsv').click();
      const text = fs.readFileSync(await (await sent).path(), 'utf8');
      assert.match(text, /静止（IMU 判据）,,no,WT-offline-GaitWindow-Refined,imu_still/);
      assert.match(text, /步行,,no,WT-offline-GaitWindow-Refined,local_review/);
      await page.locator('#activityImu').click(); assert.match(await page.locator('#imuSegNote').textContent(), /静止（IMU 判据）/);
      await page.locator('[data-tab="activity"]').click();
      console.log('PASS GaitWindow walk-stop-walk: zero false running, short walk preserved, still in table, CSV and IMU');
      const candidate = await tas(input);
      console.log('WT walk-stop-walk: Moticon TAS running/quiet seconds', seconds(candidate, '跑步', 0, Infinity).toFixed(1), seconds(candidate, '静止', 46.2, 77.3).toFixed(1));
    }
    if (fs.existsSync(walk)) for (const [key, record] of Object.entries(JSON.parse(fs.readFileSync(walk)))) {
      const input = wtWalkZip(record), baseline = seconds(await opt(input), '跑步', 0, Infinity);
      const candidate = seconds(await tas(input), '跑步', 0, Infinity);
      console.log('WT walking control:', key, 'seconds flagged running (GaitWindow / Moticon TAS)', baseline.toFixed(1), candidate.toFixed(1));
    }
  }
  if (process.env.WT_RUN31_ZIP) {
    const engine = activityEngine();
    const select = async input => {
      await page.selectOption('#activityModel', 'GaitWindow 优化版');
      await importFile(input); await analyze();
      const {inputs, duration} = await page.evaluate(() => window.temporalTestInput);
      const [old, candidate] = await Promise.all([true, 'temporal'].map(mode =>
        analyzeGaitWindow(engine, 'comparison', inputs, duration, undefined, mode)));
      assert.deepEqual(candidate.probs, old.probs);
      for (let i = 0; i < old.ticks; i++) if (old.labels[i] === -1 || old.labels[i] === 4) assert.equal(candidate.labels[i], old.labels[i]);
      return [old.labels, candidate.labels];
    };
    const zip = file => ({name: 'temporal-test.zip', mimeType: 'application/zip', buffer: fs.readFileSync(file)});
    const count = (y, c, a = 0, b = y.length) => y.slice(a, b).filter(v => v === c).length;
    const runs = y => Array.from(y).reduce((n, v, i) => n + (i === 0 || v !== y[i - 1] ? 1 : 0), 0);
    for (const [name, file] of Object.entries({Run31: process.env.WT_RUN31_ZIP, Run51: process.env.WT_RUN51_ZIP,
      mixed: process.env.WT_RUN_ZIP, walkStopWalk: process.env.WT_WALK_STOP_WALK_ZIP})) {
      if (!file) continue;
      const [old, candidate] = await select(zip(file));
      const changed = [...candidate.entries()].filter(([i, c]) => c !== old[i]).map(([i, c]) => `${(i / 10).toFixed(1)}:${old[i]}→${c}`);
      console.log(`Temporal ${name}: segments ${runs(old)}→${runs(candidate)}; changed ${changed.join(', ') || 'none'}`);
      if (name === 'Run31') {
        assert.equal(runs(old.slice(289, 301)), 5);
        assert.equal(runs(candidate.slice(289, 301)), 1);
        assert.ok([1, ACTIVITY_WALK_RUN_UNCERTAIN].includes(candidate[289]));
        assert.ok(count(candidate, 1, 275, 281) <= count(old, 1, 275, 281));
        assert.equal(count(candidate, 1, 310, 380), 70);
      } else if (name === 'Run51') assert.equal(count(candidate, 1, 400, 700), 300);
      else if (name === 'mixed') {
        assert.equal(count(candidate, 1, 990, 1390), 400);
        assert.equal(count(candidate, 1, 293, 297), 0);
      } else {
        assert.equal(count(candidate, 1), 0);
        assert.ok(count(candidate, 0, 775, 811) >= count(old, 0, 775, 811));
      }
    }
    if (fs.existsSync(walk)) for (const [name, record] of Object.entries(JSON.parse(fs.readFileSync(walk)))) {
      const [old, candidate] = await select(wtWalkZip(record));
      assert.ok(count(candidate, 1) <= count(old, 1), name);
      console.log(`Temporal walking control ${name}: run ${(count(old, 1) / 10).toFixed(1)}→${(count(candidate, 1) / 10).toFixed(1)} s`);
    }
  }
  for (const [file, cls, count] of [[process.env.APP_WALK_ZIP, '步行', 83], [process.env.APP_RUN_ZIP, '跑步', 144]]) {
    if (!file) continue;
    await page.selectOption('#activityModel', 'B1Q 实测候选');
    await page.selectOption('#activityIctoVersion', 'legacy');
    await importFile({name: 'app-gait.zip', mimeType: 'application/zip', buffer: fs.readFileSync(file)});
    await analyze();
    const segment = page.locator('#activityRows tr').filter({has: page.getByRole('cell', {name: cls, exact: true})})
      .filter({has: page.locator(`td:nth-child(8):text-is("${count}")`)});
    await segment.click();
    await page.locator('#gaitSteps summary').click();
    const source = page.locator('#gaitStepAngleSource');
    const heads = () => page.locator('#gaitSteps thead tr:last-child th').allTextContents();
    const snapshot = () => page.locator('#gaitStepRows tr').evaluateAll(rs => rs.map(r => [...r.cells].map(c => c.textContent.trim())));
    await page.locator('#gaitStepRows tr').first().waitFor({state: 'visible'});
    assert.ok(await source.isVisible());
    assert.equal(await source.inputValue(), 'device');
    assert.equal((await heads()).length, 9);
    assert.match((await heads())[5], /触地设备俯仰/);
    assert.match((await heads())[6], /离地设备俯仰/);
    const device = await snapshot();
    assert.ok(device.length && device.every(r => r.length === 9 && [5, 6, 7].every(i => /^-?\d+\.\d+$/.test(r[i]))));
    if (process.env.APP_GAIT_REFERENCE) {
      const {recordings} = JSON.parse(fs.readFileSync(process.env.APP_GAIT_REFERENCE, 'utf8'));
      const ref = recordings.find(r => r.sample === (cls === '步行' ? 'walking' : 'running'));
      const actual = [...device], next = page.locator('#gaitStepNext'), prev = page.locator('#gaitStepPrev');
      while (!(await next.isDisabled())) { await next.click(); actual.push(...await snapshot()); }
      while (!(await prev.isDisabled())) await prev.click();
      assert.deepEqual(actual, ref.result.segments[0].steps.map(x => [String(x.step_index), x.side === 'left' ? '左脚' : '右脚',
        String(x.ic_ms), String(x.to_ms), String(x.to_ms - x.ic_ms), x.ic_pitch_deg.toFixed(1), x.to_pitch_deg.toFixed(1),
        x.impact_peak_g.toFixed(2), String(x.impact_peak_ms)]));
      for (const [key, label, digits] of [['ic_pitch_deg', '触地设备俯仰', 1], ['to_pitch_deg', '离地设备俯仰', 1], ['impact_peak_g', '平均三轴加速度合量峰值', 2]]) {
        const values = page.locator('#gaitMetricRows tr').filter({hasText: label}).locator('td');
        for (const [i, side] of ['left', 'right'].entries()) {
          const {mean, valid_steps: n} = ref.summaries[0][side][key];
          assert.equal(await values.nth(i).textContent(), `${mean.toFixed(digits)}${n} / ${n} 步`);
        }
      }
    }
    const metrics = await page.locator('#gaitMetricRows').textContent();
    await source.selectOption('sideTilt');
    assert.match((await heads())[5], /触地设备侧倾/);
    assert.ok((await snapshot()).every(r => /^-?\d+\.\d$/.test(r[5])));
    const status = await page.locator('#activityStatus').textContent();
    await source.selectOption('foot');
    const foot = await snapshot();
    assert.match((await heads())[5], /触地俯仰/);
    assert.match((await heads())[6], /离地俯仰/);
    assert.ok(foot.every(r => [5, 6].every(i => /^-?\d+\.\d$|^—/.test(r[i]))));
    const nonAngles = rows => rows.map(r => [...r.slice(0, 5), ...r.slice(7)]);
    assert.deepEqual(nonAngles(foot), nonAngles(device));
    assert.equal(await page.locator('#gaitMetricRows').textContent(), metrics);
    assert.equal(await page.locator('#activityStatus').textContent(), status);
    await source.selectOption('device');
    assert.deepEqual(await snapshot(), device);
    if (cls === '跑步') {
      await page.selectOption('#activityIctoVersion', 'running');
      await page.waitForFunction(() => /跑步实验版 IC\/TO/.test(document.getElementById('gaitNote').textContent));
      let incomplete = 0;
      const next = page.locator('#gaitStepNext');
      for (;;) {
        const rows = await snapshot();
        for (const r of rows.filter(r => r[3] === '—')) { assert.equal(r[6], '—'); incomplete++; }
        if (await next.isDisabled()) break;
        await next.click();
      }
      assert.equal(incomplete, 5);
    }
    await source.selectOption('foot');
    console.log(`PASS App ${cls}: ${count} steps, device angle default, separate estimate, unchanged metrics and peak timing`);
  }
  if ((process.env.APP_WALK_ZIP || process.env.APP_RUN_ZIP) && fs.existsSync(walk)) {
    await importFile(wtWalkZip(JSON.parse(fs.readFileSync(walk))['64步.zip'])); await analyze();
    await page.locator('#gaitSteps summary').click();
    await page.locator('#gaitStepRows tr').first().waitFor({state: 'visible'});
    assert.ok(await page.locator('#gaitStepAngleSource').isHidden());
    assert.match(await page.locator('#gaitStepIcAngle').textContent(), /触地俯仰/);
    assert.equal(await page.locator('#gaitStepRows tr').first().locator('td').count(), 9);
    console.log('PASS switching from an App package to a legacy recording leaves no device angle columns');
  }
  assert.deepEqual(errors, []); assert.deepEqual(requests, []); console.log('PASS zero page errors, zero network requests');
} finally { await browser.close(); }
