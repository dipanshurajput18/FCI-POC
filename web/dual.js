'use strict';
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (v, d = 2) => (v == null || isNaN(v)) ? '–' : Number(v).toFixed(d);
const api = (path, body) => fetch(path, body === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)})
  .then(async r => { if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText); return r.json(); });

let CFG = null, RES = null, CLASSES = {}, curShot = null, liveUrl = null, LIVE_N = '';
const lbl = c => CLASSES[c]?.label || c;
const sw = c => `<i style="background:${CLASSES[c]?.color || '#888'}"></i>`;

/* ---------- 3D rig: load the iframe only when opened ---------- */
$('rigPanel').addEventListener('toggle', () => { if ($('rigPanel').open && !$('rig').src) $('rig').src = '/rig3d'; });

/* ---------- config ---------- */
async function loadConfig() {
  CFG = await api('/api/dual/config');
  $('preset').innerHTML = Object.entries(CFG.presets).map(([k, v]) => `<option value="${esc(k)}">${esc(v)}</option>`).join('');
  $('preset').value = CFG.source.preset;
  $('srcKind').value = CFG.source.kind === 'sim' ? 'sim' : 'url';
  $('grade').innerHTML = CFG.grades.map(g => `<option>${esc(g)}</option>`).join('');
  $('grade').value = CFG.rice.grade; $('sampleG').value = CFG.rice.sample_g;
  $('lot').value = CFG.lot.lot_id; $('officer').value = CFG.lot.officer;
  $('flip').checked = CFG.flip_bottom; $('chalky').value = CFG.chalky_rule;
  $('auto').checked = CFG.auto_capture; $('useStill').checked = CFG.use_still;
  showSrc();
}
function showSrc() {
  const sim = $('srcKind').value === 'sim';
  $('srcSim').hidden = !sim; $('srcUrl').hidden = sim; $('shake').hidden = !sim;
}
$('srcKind').addEventListener('change', showSrc);
$('preset').addEventListener('change', () => $('connect').click());
let setT;
function pushSettings(extra = {}) {
  clearTimeout(setT);
  setT = setTimeout(async () => {
    try {
      CFG = await api('/api/dual/settings', {lot_id: $('lot').value, officer: $('officer').value, grade: $('grade').value,
        sample_g: +$('sampleG').value || 0, chalky_rule: $('chalky').value, flip_bottom: $('flip').checked,
        auto_capture: $('auto').checked, use_still: $('useStill').checked, ...extra});
    } catch (e) { showErr(e.message); }
  }, 150);
}
['lot', 'officer', 'sampleG', 'grade', 'chalky', 'flip', 'auto', 'useStill'].forEach(id => $(id).addEventListener('change', () => pushSettings()));
$('connect').addEventListener('click', async () => {
  const sim = $('srcKind').value === 'sim';
  $('connect').disabled = true; $('connect').textContent = sim ? 'Connecting…' : 'Connecting to both phones…';
  try {
    CFG = await api('/api/dual/source', sim ? {kind: 'sim', preset: $('preset').value}
      : {kind: 'url', top: $('topUrl').value, bottom: $('botUrl').value, marker_mm: +$('markerMm').value || 20});
    showErr(null);
  } catch (e) { showErr('Could not connect: ' + e.message); }
  $('connect').disabled = false; $('connect').textContent = 'Connect';
});
$('shake').addEventListener('click', () => api('/api/dual/shake', {}));
$('capture').addEventListener('click', () => api('/api/dual/capture', {}));
$('newSample').addEventListener('click', () => api('/api/dual/new-sample', {}));
function showErr(m) { $('err').hidden = !m; $('err').textContent = m || ''; }

/* ---------- live ---------- */
const STATE = {moving: ['p-warn', 'Moving'], settling: ['p-info', 'Settling…'], still: ['p-info', 'Still'],
  analysing: ['p-warn', 'Analysing both photos…'], analysed: ['p-ok', 'Analysed · re-spread for next shot'],
  'no signal': ['p-bad', 'No signal'], starting: ['p-info', 'Starting']};
function calTxt(c) { return c?.source === 'markers' ? `<b>${fmt(c.px_per_mm, 1)} px/mm</b> (${c.markers} markers ✓)` : `<b>no markers</b> (guessing ${fmt(c?.px_per_mm, 1)} px/mm)`; }
function onLive(m) {
  const [c, t] = STATE[m.state] || ['p-info', m.state];
  $('state').className = 'pill ' + c;
  $('state').textContent = m.state === 'no signal' && m.missing ? `No signal from the ${m.missing} phone` : t;
  $('srcLabel').textContent = m.source || '–';
  if (m.top) {
    $('fps').textContent = `${fmt(m.fps, 1)} fps · top ${m.top.resolution.join('×')} · bottom ${m.bottom.resolution.join('×')}`;
    $('calInfo').innerHTML = `Top: ${calTxt(m.top.calibration)}<br>Bottom: ${calTxt(m.bottom.calibration)}`;
    LIVE_N = `${m.top.n} grains in top view · ${m.bottom.n} in bottom view (live preview)`;
  }
  renderLegend();
}
function renderLegend() {
  const sc = RES?.summary?.counts;
  $('liveLegend').innerHTML = Object.entries(CLASSES).filter(([k]) => k !== 'edge')
    .map(([k, v]) => `<span><i style="background:${v.color}"></i>${esc(v.label)}${sc ? ` <b>${sc[k] || 0}</b>` : ''}</span>`).join('')
    + `<span class="muted">${sc ? 'counts = combined sample · ' : ''}${esc(LIVE_N)}</span>`;
}
function connectWs() {
  const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/dual');
  ws.binaryType = 'blob';
  ws.onopen = () => { $('conn').className = 'pill p-ok'; $('conn').textContent = 'connected'; };
  ws.onclose = () => { $('conn').className = 'pill p-bad'; $('conn').textContent = 'disconnected – retrying'; setTimeout(connectWs, 1500); };
  ws.onmessage = ev => {
    if (ev.data instanceof Blob) {
      const u = URL.createObjectURL(ev.data);
      $('live').onload = () => { if (liveUrl) URL.revokeObjectURL(liveUrl); liveUrl = u; };
      $('live').src = u; return;
    }
    const m = JSON.parse(ev.data);
    if (m.type === 'live') onLive(m);
    else if (m.type === 'result') onResult(m);
    else if (m.type === 'error') showErr(m.error);
  };
}

/* ---------- results ---------- */
function onResult(r) {
  RES = r; CLASSES = r.classes; renderLegend();
  const s = r.summary, st = r.stats;
  $('resEyebrow').textContent = `Result · sample ${r.sample_id} · ${r.grade} · both faces`;
  if (!s) {
    $('banner').innerHTML = `<div class="banner info"><span>Waiting for a still plate…</span><small>Lot ${esc(r.lot.lot_id)}</small></div>`;
    ['k1', 'k2', 'k3', 'k4', 's1', 's2', 's3', 's4'].forEach(k => $(k).textContent = '–');
    $('pHead').innerHTML = ''; $('pBody').innerHTML = '<tr><td class="muted">Waiting for the first still plate…</td></tr>';
    $('foot').textContent = ''; $('statNote').textContent = ''; renderShots(); return;
  }
  const ok = s.verdict.code === 'WITHIN';
  $('banner').innerHTML = `<div class="banner ${ok ? 'ok' : 'bad'}"><span>${esc(s.verdict.text)}</span><small>${
    s.verdict.params.length ? 'Beyond: ' + esc(s.verdict.params.join(', ')) : 'All assessed parameters within limits'} · lot ${esc(r.lot.lot_id)}</small></div>`;
  $('k1').textContent = s.n_kernels; $('k2').textContent = r.shots.length;
  $('k3').textContent = `${fmt(s.mean_length_mm)} mm`; $('k3l').textContent = `Mean length (whole) · L:B ${fmt(s.mean_lb)}`;
  $('k4').textContent = s.size_class;
  $('pHead').innerHTML = '<tr><th>Parameter</th><th class="n">Count</th><th class="n">% count</th><th class="n">% weight</th><th class="n">Limit %</th><th>Status</th></tr>';
  $('pBody').innerHTML = s.params.map(p => {
    const stt = p.status === 'within' ? '<span class="pill p-ok">Within</span>' : p.status === 'beyond' ? '<span class="pill p-bad">Beyond</span>' : '<span class="pill p-info">n/a</span>';
    return `<tr><td>${esc(p.name)}</td><td class="n">${p.count}</td><td class="n">${fmt(p.count_pct)}</td><td class="n"><b>${fmt(p.wt_pct)}</b></td><td class="n">${p.limit ?? '–'}</td><td>${stt}</td></tr>`;
  }).join('');
  const last = r.shots[r.shots.length - 1];
  $('s1').textContent = st.paired;
  $('s2').textContent = st.hidden_kernels;
  $('s2l').textContent = st.hidden_kernels ? 'Defects seen only from below: ' + Object.entries(st.hidden).map(([k, v]) => `${v} ${lbl(k).toLowerCase()}`).join(', ') : 'Defects seen only from below';
  $('s3').textContent = st.top_only + st.bottom_only;
  $('s4').textContent = last?.align_err_mm != null ? `±${fmt(last.align_err_mm)} mm` : '–';
  $('s4l').textContent = 'Alignment · ' + (last?.align || '');
  $('statNote').innerHTML = `${st.changed} grain(s) got a different result than the top camera alone would give.
    ${st.top_only + st.bottom_only ? `${st.top_only} seen only from the top, ${st.bottom_only} only from the bottom (flagged for review).` : ''}`;
  const share = s.sample_g ? ` = ${fmt(100 * s.imaged_g / s.sample_g, 0)}% of the ${fmt(s.sample_g, 1)} g sample` : '';
  const clf = last ? (last.classifier_t?.used === 'model' ? 'AI model on both faces' : 'rules: ' + esc(last.classifier_t?.note || '')) : '';
  $('foot').innerHTML = `Weight from the top view: ≈ <b>${fmt(s.imaged_g, 2)} g imaged</b>${share}. Chalky rule: <b>${esc($('chalky').selectedOptions[0].text)}</b>. Condition: ${clf}.
    ${s.n_review ? `<b>${s.n_review}</b> grain(s) flagged for officer review.` : ''}<br>Not assessed by camera: ${esc((s.not_assessed || []).join(' · '))}`;
  renderShots();
}

function renderShots() {
  const shots = RES?.shots || [];
  $('thumbs').innerHTML = shots.map(s => `<button data-n="${s.n}" class="${s.n === curShot ? 'on' : ''}">Shot ${s.n + 1}</button>`).join('');
  $('thumbs').querySelectorAll('button').forEach(b => b.addEventListener('click', () => showShot(+b.dataset.n)));
  if (!shots.length) { $('imgT').removeAttribute('src'); $('imgB').removeAttribute('src'); curShot = null;
    $('inspector').innerHTML = '<p class="muted">Click a grain in either picture.</p>'; return; }
  if (curShot == null || !shots.find(s => s.n === curShot) || curShot !== shots[shots.length - 1].n) showShot(shots[shots.length - 1].n);
}
function showShot(n) {
  curShot = n; $('hlT').style.display = $('hlB').style.display = 'none';
  $('imgT').src = `/api/dual/shot/${n}/top.jpg?v=${RES.ver}`;
  $('imgB').src = `/api/dual/shot/${n}/bottom.jpg?v=${RES.ver}`;
  $('thumbs').querySelectorAll('button').forEach(b => b.classList.toggle('on', +b.dataset.n === n));
}
function mark(imgId, hlId, bb, size) {
  const hl = $(hlId);
  if (!bb) { hl.style.display = 'none'; return; }
  const sc = $(imgId).getBoundingClientRect().width / size[0], [x, y, w, h] = bb;
  Object.assign(hl.style, {display: 'block', left: (x * sc - 3) + 'px', top: (y * sc - 3) + 'px', width: (w * sc + 6) + 'px', height: (h * sc + 6) + 'px'});
}
function pick(ev, side) {
  const shot = RES?.shots.find(s => s.n === curShot); if (!shot) return;
  const img = $(side === 't' ? 'imgT' : 'imgB'), size = side === 't' ? shot.size_t : shot.size_b;
  const r = img.getBoundingClientRect(), sc = r.width / size[0];
  const x = (ev.clientX - r.left) / sc, y = (ev.clientY - r.top) / sc;
  let best = null, bd = 1e9;
  for (const k of shot.kernels) {
    const bb = side === 't' ? k.bbox_t : k.bbox_b; if (!bb) continue;
    const [bx, by, bw, bh] = bb;
    if (x < bx - 3 || x > bx + bw + 3 || y < by - 3 || y > by + bh + 3) continue;
    const d = Math.hypot(bx + bw / 2 - x, by + bh / 2 - y);
    if (d < bd) { best = k; bd = d; }
  }
  if (!best) return;
  mark('imgT', 'hlT', best.bbox_t, shot.size_t); mark('imgB', 'hlB', best.bbox_b, shot.size_b);
  inspect(best);
}
$('imgT').addEventListener('click', ev => pick(ev, 't'));
$('imgB').addEventListener('click', ev => pick(ev, 'b'));

function face(title, f) {
  if (!f) return `<div class="face"><div class="eyebrow">${title}</div><p class="muted small">Not matched in this view.</p></div>`;
  const bars = f.model ? `<div class="bars">${Object.entries(f.model).map(([n, p]) =>
    `<span class="mono">${esc(n)}</span><span class="bar"><i style="width:${(p * 100).toFixed(0)}%"></i></span><span class="mono">${(p * 100).toFixed(0)}%</span>`).join('')}</div>` : '';
  return `<div class="face"><div class="eyebrow">${title}</div><h3>${sw(f.cls)}${esc(lbl(f.cls))}</h3>
    ${f.also?.length ? `<p class="small muted">Also: ${f.also.map(a => esc(lbl(a))).join(', ')}</p>` : ''}
    <ul class="small">${f.why.map(w => `<li>${esc(w)}</li>`).join('')}</ul>${bars}
    <dl><dt>Opaque area</dt><dd>${fmt(f.opaque_pct, 0)}%</dd><dt>Dark spots</dt><dd>${fmt(f.dark_pct, 1)}%</dd>
    <dt>Red surface</dt><dd>${fmt(f.red_pct, 0)}%</dd><dt>L* a* b*</dt><dd>${fmt(f.L, 0)} / ${fmt(f.a, 0)} / ${fmt(f.b, 0)}</dd></dl></div>`;
}
function inspect(k) {
  const where = {both: 'matched in both cameras', top: 'top camera only', bottom: 'bottom camera only'}[k.side] || '';
  $('inspector').innerHTML = `<div class="eyebrow">Grain #${k.fid + 1} · shot ${k.shot + 1} · ${where}</div>
    <div class="faces">
      <div class="face"><div class="eyebrow">Final result</div><h3>${sw(k.cls)}${esc(lbl(k.cls))}</h3>
        ${k.hidden?.length ? `<p class="pill p-warn">Only visible from below: ${k.hidden.map(h => esc(lbl(h))).join(', ')}</p>` : ''}
        ${k.review?.length ? `<p class="small muted">Review: ${esc(k.review.join('; '))}</p>` : ''}
        <dl><dt>Length</dt><dd>${fmt(k.length_mm)} mm</dd><dt>Breadth</dt><dd>${fmt(k.breadth_mm)} mm</dd>
        <dt>Of whole length</dt><dd>${fmt(k.rel_length * 100, 0)}%</dd><dt>Est. mass</dt><dd>${k.mass_g != null ? fmt(k.mass_g * 1000, 1) + ' mg' : '–'}</dd></dl></div>
      ${face('Top face', k.top)}
      ${face('Bottom face (through glass)', k.bottom)}
    </div>`;
}

loadConfig().then(connectWs).catch(e => showErr('Server not reachable: ' + e.message));
