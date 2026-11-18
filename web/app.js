'use strict';
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (v, d = 2) => (v == null || isNaN(v)) ? '–' : Number(v).toFixed(d);
const api = (path, body) => fetch(path, body === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)})
  .then(async r => { if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText); return r.json(); });

let CFG = null, RES = null, CLASSES = {}, curShot = null, liveUrl = null, LIVE_N = 0;
function renderLegend() {
  const sc = RES?.summary?.counts;
  $('liveLegend').innerHTML = Object.entries(CLASSES).filter(([k]) => k !== 'edge')
    .map(([k, v]) => `<span><i style="background:${v.color}"></i>${esc(v.label)}${sc ? ` <b>${sc[k] || 0}</b>` : ''}</span>`).join('')
    + `<span class="muted">${sc ? 'counts = analysed sample · ' : ''}${LIVE_N} objects in view (live preview)</span>`;
}

/* ---------------- config / controls ---------------- */
async function loadConfig() {
  CFG = await api('/api/config');
  setCommodityUI(CFG.commodity);
  $('grade').innerHTML = CFG.grades.map(g => `<option>${esc(g)}</option>`).join('');
  $('grade').value = CFG.rice.grade;
  $('lot').value = CFG.lot.lot_id; $('officer').value = CFG.lot.officer;
  $('sampleG').value = CFG.commodity === 'rice' ? CFG.rice.sample_g : CFG.sugar.sample_g;
  $('auto').checked = CFG.auto_capture; $('useStill').checked = CFG.use_still;
  $('t-broken').value = Math.round(CFG.rice.broken_frac * 100);
  $('t-chalky').value = Math.round(CFG.rice.chalky_frac * 100);
  $('t-margin').value = CFG.rice.chalk_margin;
  $('t-red').value = Math.round(CFG.rice.red_frac * 100);
  $('t-ref').value = CFG.rice.ref_length_mm;
  $('t-lump').value = CFG.sugar.lump_factor; $('t-speck').value = CFG.sugar.speck_L;
  fillPresets(); syncOut();
  if (CFG.source.kind === 'sim') $('preset').value = CFG.source.preset;
}
function fillPresets() {
  const p = CFG.presets[CFG.commodity];
  $('preset').innerHTML = Object.entries(p).map(([k, v]) => `<option value="${esc(k)}">${esc(v)}</option>`).join('');
}
function setCommodityUI(c) {
  document.querySelectorAll('#commodity button').forEach(b => b.classList.toggle('on', b.dataset.c === c));
  $('gradeWrap').hidden = c !== 'rice';
  $('thrRice').hidden = c !== 'rice'; $('thrSugar').hidden = c !== 'sugar';
}
function syncOut() {
  $('o-broken').textContent = $('t-broken').value + '%';
  $('o-chalky').textContent = $('t-chalky').value + '%';
  $('o-margin').textContent = $('t-margin').value;
  $('o-red').textContent = $('t-red').value + '%';
  $('o-ref').textContent = +$('t-ref').value ? $('t-ref').value + ' mm' : 'auto';
  $('o-lump').textContent = $('t-lump').value; $('o-speck').textContent = $('t-speck').value;
}
let setT;
function pushSettings(extra = {}) {
  clearTimeout(setT);
  setT = setTimeout(async () => {
    const d = {
      lot_id: $('lot').value, officer: $('officer').value, grade: $('grade').value,
      sample_g: +$('sampleG').value || 0, auto_capture: $('auto').checked,
      broken_frac: $('t-broken').value / 100, chalky_frac: $('t-chalky').value / 100,
      chalk_margin: +$('t-margin').value, red_frac: $('t-red').value / 100, ref_length_mm: +$('t-ref').value,
      lump_factor: +$('t-lump').value, speck_L: +$('t-speck').value, ...extra };
    try { CFG = await api('/api/settings', d); } catch (e) { showErr(e.message); }
  }, 150);
}
document.querySelectorAll('#commodity button').forEach(b => b.addEventListener('click', async () => {
  const c = b.dataset.c;
  CFG = await api('/api/settings', {commodity: c});
  setCommodityUI(c); fillPresets();
  $('sampleG').value = c === 'rice' ? CFG.rice.sample_g : CFG.sugar.sample_g;
  $('markerMm').value = c === 'rice' ? 20 : 8;
  if (CFG.source.kind === 'sim') $('preset').value = CFG.source.preset;
}));
$('useStill').addEventListener('change', () => pushSettings({use_still: $('useStill').checked}));
['lot', 'officer', 'sampleG', 'grade', 'auto'].forEach(id => $(id).addEventListener('change', () => pushSettings()));
['t-broken', 't-chalky', 't-margin', 't-red', 't-ref', 't-lump', 't-speck'].forEach(id => $(id).addEventListener('input', () => { syncOut(); pushSettings(); }));

function showSrc() {
  const k = $('srcKind').value;
  $('srcSim').hidden = k !== 'sim'; $('srcUsb').hidden = k !== 'usb'; $('srcUrl').hidden = k !== 'url';
  $('srcFile').hidden = k !== 'file'; $('srcPhoto').hidden = k !== 'photo';
  $('shake').hidden = k !== 'sim';
}
$('srcKind').addEventListener('change', showSrc);
$('preset').addEventListener('change', () => $('connect').click());
$('findCams').addEventListener('click', async () => {
  $('camList').textContent = 'Searching…';
  const r = await api('/api/cameras');
  $('camList').textContent = r.usb.length ? 'Found camera number(s): ' + r.usb.join(', ') : 'No USB camera found.';
  if (r.usb.length) $('usbIdx').value = r.usb[0];
});
$('connect').addEventListener('click', async () => {
  const k = $('srcKind').value, mm = +$('markerMm').value || 20;
  const commodity = document.querySelector('#commodity button.on').dataset.c;
  try {
    if (k === 'photo') {
      const f = $('photo').files[0]; if (!f) return showErr('Choose a photo first');
      const r = await fetch('/api/upload?marker_mm=' + mm, {method: 'POST', body: f});
      if (!r.ok) throw new Error((await r.json()).detail); CFG = await r.json();
    } else {
      const value = k === 'usb' ? $('usbIdx').value : k === 'url' ? $('url').value : k === 'file' ? $('filePath').value : null;
      CFG = await api('/api/source', {kind: k, value, commodity, preset: $('preset').value, marker_mm: mm});
    }
    showErr(null);
  } catch (e) { showErr('Could not connect: ' + e.message); }
});
$('shake').addEventListener('click', () => api('/api/shake', {}));
$('capture').addEventListener('click', () => api('/api/capture', {}));
$('newSample').addEventListener('click', () => api('/api/new-sample', {}));
function showErr(m) { $('err').hidden = !m; $('err').textContent = m || ''; }

/* ---------------- live stream ---------------- */
const STATE = {moving: ['p-warn', 'Moving'], settling: ['p-info', 'Settling…'], still: ['p-info', 'Still'],
  analysing: ['p-warn', 'Analysing…'], analysed: ['p-ok', 'Analysed · re-spread for next shot'], 'no signal': ['p-bad', 'No signal'], starting: ['p-info', 'Starting']};
function onLive(m) {
  const [c, t] = STATE[m.state] || ['p-info', m.state];
  $('state').className = 'pill ' + c; $('state').textContent = t;
  $('srcLabel').textContent = m.source || '–';
  const cal = m.calibration || {};
  $('fps').textContent = `${fmt(m.fps, 1)} fps · ${m.resolution ? m.resolution.join('×') : ''} · ${fmt(cal.px_per_mm, 1)} px/mm`;
  $('calInfo').innerHTML = cal.source === 'markers'
    ? `Calibration: <b>${fmt(cal.px_per_mm, 2)} px/mm</b> from ${cal.markers} marker(s) ✓`
    : `Calibration: <b>no markers found</b> – using manual ${fmt(cal.px_per_mm, 1)} px/mm. Place the calibration sheet in view.`;
  LIVE_N = m.n; renderLegend();
}
function connectWs() {
  const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
  ws.binaryType = 'blob';
  ws.onopen = () => { $('conn').className = 'pill p-ok'; $('conn').textContent = 'connected'; };
  ws.onclose = () => { $('conn').className = 'pill p-bad'; $('conn').textContent = 'disconnected – retrying'; setTimeout(connectWs, 1500); };
  ws.onmessage = ev => {
    if (ev.data instanceof Blob) {
      const u = URL.createObjectURL(ev.data);
      $('live').onload = () => { if (liveUrl) URL.revokeObjectURL(liveUrl); liveUrl = u; };
      $('live').src = u;
      return;
    }
    const m = JSON.parse(ev.data);
    if (m.type === 'live') onLive(m);
    else if (m.type === 'result') onResult(m);
    else if (m.type === 'error') showErr(m.error);
  };
}

/* ---------------- results ---------------- */
function onResult(r) {
  RES = r; CLASSES = r.classes; renderLegend();
  const s = r.summary, rice = r.commodity === 'rice';
  $('resEyebrow').textContent = `Result · sample ${r.sample_id}` + (rice ? ` · ${r.grade}` : ' · IS 498');
  if (!s) {
    $('banner').innerHTML = `<div class="banner info"><span>Waiting for a still tray…</span><small>Lot ${esc(r.lot.lot_id)}</small></div>`;
    ['k1', 'k2', 'k3', 'k4'].forEach(k => $(k).textContent = '–');
    $('pHead').innerHTML = ''; $('pBody').innerHTML = '<tr><td class="muted">Waiting for the first still frame…</td></tr>';
    $('dist').innerHTML = ''; $('foot').textContent = ''; renderShots(); return;
  }
  const ok = s.verdict.code === 'WITHIN';
  $('banner').innerHTML = `<div class="banner ${ok ? 'ok' : 'bad'}"><span>${esc(s.verdict.text)}</span><small>${
    s.verdict.params.length ? 'Beyond: ' + esc(s.verdict.params.join(', ')) : 'All assessed parameters within limits'} · lot ${esc(r.lot.lot_id)}</small></div>`;
  $('k1').textContent = s.n_kernels; $('k1l').textContent = rice ? 'Kernels analysed' : 'Crystals analysed';
  $('k2').textContent = r.shots.length;
  if (rice) {
    $('k3').textContent = `${fmt(s.mean_length_mm)} mm`; $('k3l').textContent = `Mean length (whole) · L:B ${fmt(s.mean_lb)}`;
    $('k4').textContent = s.size_class; $('k4l').textContent = 'Size class';
    $('pHead').innerHTML = '<tr><th>Parameter</th><th class="n">Count</th><th class="n">% count</th><th class="n">% weight</th><th class="n">Limit %</th><th>Status</th></tr>';
    $('pBody').innerHTML = s.params.map(p => {
      const st = p.status === 'within' ? '<span class="pill p-ok">Within</span>' : p.status === 'beyond' ? '<span class="pill p-bad">Beyond</span>' : '<span class="pill p-info">n/a</span>';
      return `<tr><td>${esc(p.name)}</td><td class="n">${p.count}</td><td class="n">${fmt(p.count_pct)}</td><td class="n"><b>${fmt(p.wt_pct)}</b></td><td class="n">${p.limit ?? '–'}</td><td>${st}</td></tr>`;
    }).join('');
    $('dist').innerHTML = '';
  } else {
    $('k3').textContent = `${fmt(s.ma_mm, 2)} mm`; $('k3l').textContent = `Mean aperture MA · CV ${fmt(s.cv_pct, 0)}%`;
    $('k4').textContent = s.grade; $('k4l').textContent = 'Grade (size-colour)';
    $('pHead').innerHTML = '<tr><th>Parameter</th><th class="n">Value</th><th>Status</th></tr>';
    $('pBody').innerHTML = s.params.map(p => {
      const st = p.status === 'within' ? '<span class="pill p-ok">OK</span>' : p.status === 'beyond' ? '<span class="pill p-bad">Check</span>' : '<span class="pill p-info">info</span>';
      return `<tr><td>${esc(p.name)}</td><td class="n"><b>${esc(p.value)}</b></td><td>${st}</td></tr>`;
    }).join('');
    const chk = (s.group_checks || []).map(g => `<tr><td><b>${g.group}</b></td>${g.rules.map(x =>
      `<td class="n">${x.actual_pct}% on ${x.aperture_mm} mm <span class="muted">(≥${x.min_pct})</span></td>`).join('')}<td>${g.ok ? '<span class="pill p-ok">meets</span>' : '<span class="pill p-info">no</span>'}</td></tr>`).join('');
    $('dist').innerHTML = `<h2 style="margin-top:16px">Virtual sieve (% retained by mass, cumulative)</h2><div class="bars">${
      s.distribution.map(d => `<span class="mono">${d.aperture_mm.toFixed(3)} mm</span><span class="bar"><i style="width:${d.retained_pct}%"></i></span><span class="mono">${fmt(d.retained_pct, 1)}%</span>`).join('')
    }</div><h2 style="margin-top:16px">IS 498 size-group test</h2><div class="tw"><table><tbody>${chk}</tbody></table></div>`;
  }
  const share = s.sample_g ? ` = ${fmt(100 * s.imaged_g / s.sample_g, 0)}% of the ${fmt(s.sample_g, 1)} g sample` : '';
  $('foot').innerHTML = `Weight basis: each grain's mass estimated from its measured volume × density; ≈ <b>${fmt(s.imaged_g, 2)} g imaged</b>${share}. Add shots until the whole sample is covered.
    ${s.n_review ? `<b>${s.n_review}</b> borderline grain(s) flagged for officer review (yellow box).` : ''}<br>Not assessed by camera: ${esc((s.not_assessed || []).join(' · '))}`;
  renderShots();
}

function renderShots() {
  const shots = RES?.shots || [];
  $('thumbs').innerHTML = shots.map(s => `<button data-n="${s.n}" class="${s.n === curShot ? 'on' : ''}">Shot ${s.n + 1}</button>`).join('');
  $('thumbs').querySelectorAll('button').forEach(b => b.addEventListener('click', () => showShot(+b.dataset.n)));
  if (!shots.length) { $('shotImg').removeAttribute('src'); curShot = null; $('inspector').innerHTML = '<p class="muted">Click a grain in the image.</p>'; return; }
  if (curShot == null || !shots.find(s => s.n === curShot) || curShot !== shots[shots.length - 1].n) showShot(shots[shots.length - 1].n);
}
function showShot(n) {
  curShot = n; $('hl').style.display = 'none';
  $('shotImg').src = `/api/shot/${n}.jpg?v=${RES.ver}`;
  $('thumbs').querySelectorAll('button').forEach(b => b.classList.toggle('on', +b.dataset.n === n));
}
$('shotImg').addEventListener('click', ev => {
  const shot = RES?.shots.find(s => s.n === curShot); if (!shot) return;
  const img = $('shotImg'), r = img.getBoundingClientRect(), sc = r.width / shot.image_size[0];
  const x = (ev.clientX - r.left) / sc, y = (ev.clientY - r.top) / sc;
  let best = null, bd = 1e9;
  for (const k of shot.kernels) {
    const [bx, by, bw, bh] = k.bbox;
    const inside = x >= bx - 3 && x <= bx + bw + 3 && y >= by - 3 && y <= by + bh + 3;
    const d = Math.hypot(k.cx - x, k.cy - y);
    if (inside && d < bd) { best = k; bd = d; }
  }
  if (!best) return;
  const [bx, by, bw, bh] = best.bbox, hl = $('hl');
  Object.assign(hl.style, {display: 'block', left: (bx * sc - 3) + 'px', top: (by * sc - 3) + 'px', width: (bw * sc + 6) + 'px', height: (bh * sc + 6) + 'px'});
  inspect(best);
});
function inspect(k) {
  const c = CLASSES[k.cls] || {label: k.cls, color: '#888'}, rice = RES.commodity === 'rice';
  const rows = rice
    ? [['Length', fmt(k.length_mm) + ' mm'], ['Breadth', fmt(k.breadth_mm) + ' mm'], ['L : B', fmt(k.lb)], ['Of whole length', fmt(k.rel_length * 100, 0) + '%'],
       ['Opaque (chalky) area', fmt(k.opaque_pct, 0) + '%'], ['Red surface', fmt(k.red_pct, 0) + '%'], ['Dark spots', fmt(k.dark_pct, 1) + '%'],
       ['Colour L* a* b*', `${fmt(k.L, 0)} / ${fmt(k.a, 0)} / ${fmt(k.b, 0)}`], ['Est. mass', k.mass_g != null ? fmt(k.mass_g * 1000, 1) + ' mg' : '–']]
    : [['Sieve size (minor)', fmt(k.sieve_mm, 3) + ' mm'], ['Length', fmt(k.length_mm, 3) + ' mm'], ['Area', fmt(k.area_mm2, 3) + ' mm²'],
       ['Colour L* b*', `${fmt(k.L, 0)} / ${fmt(k.b, 1)}`], ['Dark inclusion', fmt(k.dark_pct, 1) + '%']];
  $('inspector').innerHTML = `<div class="eyebrow">Grain #${k.id + 1} · shot ${k.shot + 1}</div>
    <h2 style="display:flex;gap:8px;align-items:center"><i style="width:12px;height:12px;border-radius:3px;background:${c.color};display:inline-block"></i>${esc(c.label)}</h2>
    ${k.also?.length ? `<p class="small muted">Also shows: ${k.also.map(a => esc(CLASSES[a]?.label || a)).join(', ')} (counted once, by FCI picking order)</p>` : ''}
    <div class="small"><b>Why</b><ul>${k.why.map(w => `<li>${esc(w)}</li>`).join('')}</ul></div>
    ${k.review?.length ? `<p class="pill p-warn">Officer review: ${esc(k.review.join('; '))}</p>` : ''}
    <dl>${rows.map(([a, b]) => `<dt>${a}</dt><dd>${b}</dd>`).join('')}</dl>`;
}

showSrc();
loadConfig().then(connectWs).catch(e => showErr('Server not reachable: ' + e.message));
