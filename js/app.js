/* =============================================================
 * app.js —— 页面交互：概率曲线 / 蒙特卡洛 / MLP 实验台 / 玩具实验
 * ============================================================= */
(function () {
  'use strict';

  const $ = id => document.getElementById(id);
  const G = window.GachaKit;
  const { MLP, makeDataset } = window.MLPKit;
  const C = window.Charts;
  const P5 = G.GACHA.P5_BASE;

  const fmt = (v, d) => v.toFixed(d == null ? 2 : d);
  const pct = (v, d) => (v * 100).toFixed(d == null ? 1 : d) + '%';
  const el = (tag, cls, html) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  };

  /* =========================================================
   * 0. Hero 小图
   * ========================================================= */
  function drawHero() {
    const ns = [], ps = [];
    for (let n = 1; n <= 90; n++) { ns.push(n); ps.push(G.p5(n)); }
    C.plot($('heroCurve'), [
      { type: 'area', x: ns, y: ps, color: 'rgba(47,95,208,0.10)' },
      { type: 'line', x: ns, y: ps, color: C.P.blue, width: 1.8 },
      { type: 'vline', x: 74, color: 'rgba(201,138,30,0.7)', dash: [3, 3], label: '74' }
    ], {
      box: { l: 34, r: 6, t: 10, b: 20 },
      xDomain: [0, 91], yDomain: [0, 1],
      xticks: [1, 30, 60, 90], yticks: [0, 0.5, 1],
      yFmt: t => Math.round(t * 100) + '%',
      xFmt: t => String(Math.round(t))
    });
  }

  /* =========================================================
   * 1. 机制：概率曲线（带两个开关）
   * ========================================================= */
  function pOf(n, soft, hard) {
    if (!soft) return (hard && n >= 90) ? 1 : P5;
    const p = Math.min(1, P5 + G.GACHA.SOFT_STEP * (n - G.GACHA.SOFT_START + 1));
    if (n < G.GACHA.SOFT_START) return hard && n >= 90 ? 1 : P5;
    return hard && n >= 90 ? 1 : p;
  }

  function drawCurve() {
    const soft = $('tgSoft').checked, hard = $('tgHard').checked;
    const ns = [], ps = [];
    for (let n = 1; n <= 90; n++) { ns.push(n); ps.push(pOf(n, soft, hard)); }

    C.plot($('curveP5'), [
      { type: 'area', x: ns, y: ps, color: 'rgba(47,95,208,0.09)' },
      { type: 'line', x: ns, y: ps, color: C.P.blue, width: 2 },
      soft ? { type: 'vline', x: 74, color: 'rgba(201,138,30,0.75)', dash: [3, 3], label: '74 抽' } : null
    ].filter(Boolean), {
      box: { l: 44, r: 10, t: 14, b: 30 },
      xDomain: [0, 91], yDomain: [0, 1],
      xticks: [1, 20, 40, 60, 74, 90], yticks: [0, 0.25, 0.5, 0.75, 1],
      yFmt: t => Math.round(t * 100) + '%',
      xFmt: t => String(Math.round(t)),
      xLabel: '垫抽数 n', yLabel: '单抽五星概率'
    });

    // 读数
    let mean = 0, survive = 1, before = 0;
    if (!soft && !hard) {
      mean = 1 / P5;
    } else {
      for (let n = 1; n <= 400; n++) {
        const p = pOf(n, soft, hard);
        mean += n * survive * p;
        survive *= (1 - p);
        if (survive < 1e-9) break;
      }
      survive = 1;
      for (let n = 1; n <= 73; n++) { before += survive * pOf(n, soft, hard); survive *= 1 - pOf(n, soft, hard); }
    }
    $('curveReadout').innerHTML =
      `<span class="k">期望出货</span> <b>${fmt(mean, 1)} 抽</b>　` +
      `<span class="k">综合概率</span> <b>${pct(1 / mean, 2)}</b>　` +
      `<span class="k">74 抽前出货</span> <b>${pct(before)}</b>`;
  }

  /* =========================================================
   * 2. 蒙特卡洛
   * ========================================================= */
  let simSeed = 20261115;

  function sampleTTH(rng, soft, hard) {
    let n = 0;
    for (;;) {
      n++;
      const p = n >= hard ? 1 : (soft ? G.p5(n) : P5);
      if (rng() < p) return n;
      if (n > 3000) return n;
    }
  }

  function runSim() {
    const N = +$('simN').value;
    const noPity = $('simMode').value === 'nopity';
    const soft = !noPity;
    const hard = noPity ? Infinity : G.GACHA.HARD;
    const rng = G.makeRNG(simSeed++);
    const xs = [];
    for (let i = 0; i < N; i++) xs.push(sampleTTH(rng, soft, hard));

    const maxN = noPity ? 300 : 91;
    const bins = new Array(maxN + 1).fill(0);
    for (const v of xs) bins[Math.min(maxN, v)]++;
    const x0 = [], x1 = [], y = [];
    for (let n = 1; n <= maxN; n++) { x0.push(n - 0.45); x1.push(n + 0.45); y.push(bins[n] / N); }

    // 解析分布
    const distOpt = noPity ? { softPity: false, hardPity: false, maxN: 300 } : { maxN: 91 };
    const dist = G.pityDist(distOpt);
    const tx = dist.filter(d => d.n <= maxN).map(d => d.n);
    const ty = dist.filter(d => d.n <= maxN).map(d => d.pmf);

    C.plot($('histChart'), [
      { type: 'bars', x0, x1, y, color: 'rgba(47,95,208,0.42)' },
      { type: 'line', x: tx, y: ty, color: C.P.gold, width: 1.6, dash: [4, 3] }
    ], {
      box: { l: 46, r: 12, t: 12, b: 30 },
      xDomain: [0, maxN + 1],
      yDomain: [0, Math.max.apply(null, y) * 1.12],
      xticks: noPity ? [1, 50, 100, 150, 200, 250, 300] : [1, 20, 40, 60, 74, 90],
      yFmt: t => (t * 100).toFixed(0) + '%',
      xLabel: '出货抽数', yLabel: '占比'
    });

    // CDF
    const sorted = xs.slice().sort((a, b) => a - b);
    const cx = [], cy = [];
    for (let n = 1; n <= maxN; n++) {
      const idx = sorted.findIndex(v => v > n);
      const cnt = idx === -1 ? N : idx;
      cx.push(n); cy.push(cnt / N);
    }
    let cum = 0;
    const tcx = [], tcy = [];
    for (const d of dist) { cum += d.pmf; tcx.push(d.n); tcy.push(cum); if (d.n >= maxN) break; }

    C.plot($('cdfChart'), [
      { type: 'line', x: cx, y: cy, color: C.P.blue, width: 2 },
      { type: 'line', x: tcx, y: tcy, color: C.P.gold, width: 1.4, dash: [4, 3] },
      { type: 'hline', y: 0.5, color: 'rgba(28,26,23,0.22)', dash: [2, 3] },
      { type: 'hline', y: 0.9, color: 'rgba(28,26,23,0.22)', dash: [2, 3] }
    ], {
      box: { l: 44, r: 12, t: 12, b: 28 },
      xDomain: [0, maxN + 1], yDomain: [0, 1],
      xticks: noPity ? [1, 50, 100, 150, 200, 250, 300] : [1, 20, 40, 60, 74, 90],
      yticks: [0, 0.25, 0.5, 0.75, 1],
      yFmt: t => Math.round(t * 100) + '%',
      xLabel: '抽数', yLabel: '累计出货比例'
    });

    // 统计量
    const st = G.stats(noPity ? { softPity: false, hardPity: false } : {});
    const meanSim = xs.reduce((a, b) => a + b, 0) / N;
    const rows = [
      ['均值', meanSim, st.mean, 1],
      ['中位数', G.quantile(sorted, 0.5), st.q50, 0],
      ['90% 分位', G.quantile(sorted, 0.9), st.q90, 0],
      ['99% 分位', G.quantile(sorted, 0.99), st.q99, 0]
    ];
    const tb = $('simStats').querySelector('tbody');
    tb.innerHTML = '';
    for (const [name, a, b, d] of rows) {
      const tr = el('tr');
      tr.appendChild(el('td', null, name));
      tr.appendChild(el('td', null, fmt(a, d)));
      tr.appendChild(el('td', null, fmt(b, d)));
      tb.appendChild(tr);
    }

    const diff = Math.abs(meanSim - st.mean) / st.mean * 100;
    $('simVerdict').innerHTML = noPity
      ? `关掉保底后均值直接涨到 <b>${fmt(st.mean, 0)} 抽</b>，而且 <b>${pct(1 - Math.pow(1 - P5, 90), 0)}</b> 的人在 90 抽内出不了金——这就是保底存在的意义。`
      : `模拟 ${N.toLocaleString()} 次的均值与解析值相差 <b>${fmt(diff, 1)}%</b>，说明两套独立算法互相印证。分布没有长尾：90% 的样本挤在 74~80 抽之间。`;
  }

  /* =========================================================
   * 3. UP 成本分布
   * ========================================================= */
  function drawUpHist() {
    const xs = G.simulateUntilUp(15000, 4321);
    const bins = new Array(21).fill(0);
    for (const v of xs) bins[Math.min(20, Math.floor(v / 10))]++;
    const x0 = [], x1 = [], y = [];
    for (let i = 0; i < 20; i++) { x0.push(i * 10); x1.push(i * 10 + 10); y.push(bins[i] / xs.length); }
    C.plot($('upHist'), [
      { type: 'bars', x0, x1, y, color: 'rgba(201,138,30,0.45)' },
      { type: 'vline', x: 93.5, color: C.P.blue, dash: [4, 3], label: '期望 93.5' }
    ], {
      box: { l: 46, r: 12, t: 14, b: 30 },
      xDomain: [0, 200], yDomain: [0, Math.max.apply(null, y) * 1.15],
      xticks: [0, 40, 80, 120, 160, 200],
      yFmt: t => (t * 100).toFixed(0) + '%',
      xLabel: '抽到当期 UP 的抽数', yLabel: '占比'
    });
  }

  /* =========================================================
   * 4. MLP 实验台
   * ========================================================= */
  const TOTAL_EPOCHS = 600;
  const mlp = {
    hidden: [16, 16], act: 'tanh', init: 'xavier', opt: 'adam', lr: 0.05, trials: 6000,
    model: null, X: null, y: null, freq: null, epoch: 0, hist: [], running: false,
    order: [], seedShift: 0
  };

  /* 分层采样：每个垫抽数独立做 trials 次伯努利试验 */
  function buildFreqData(trials, seed) {
    const rng = G.makeRNG(seed);
    const X = [], y = [], freq = [];
    for (let n = 1; n <= 90; n++) {
      const p = G.p5(n);
      let hits = 0;
      for (let i = 0; i < trials; i++) if (rng() < p) hits++;
      const f = hits / trials;
      X.push([n / 90]); y.push([1 - f, f]); freq.push(f);
    }
    return { X, y, freq };
  }

  function resetMLP(keepData) {
    if (!keepData || !mlp.X) {
      const d = buildFreqData(mlp.trials, 99);
      mlp.X = d.X; mlp.y = d.y; mlp.freq = d.freq;
    }
    mlp.model = new MLP({
      sizes: [1, ...mlp.hidden, 2],
      activation: mlp.act, init: mlp.init, optimizer: mlp.opt, lr: mlp.lr,
      seed: 5 + (mlp.seedShift * 17)
    });
    mlp.epoch = 0;
    mlp.hist = [];
    mlp.running = false;
    mlp.order = [...Array(90).keys()];
    $('btnTrain').textContent = '开始训练';
    $('trainBar').style.width = '0%';
    buildHeatLayerOptions();
    buildUnitOptions();
    renderMLPStatic();
    renderMLPDynamic();
  }

  function trainEpoch() {
    const rng = G.makeRNG(mlp.epoch * 7919 + 13);
    for (let i = mlp.order.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const t = mlp.order[i]; mlp.order[i] = mlp.order[j]; mlp.order[j] = t;
    }
    for (let s = 0; s < 90; s += 32) {
      mlp.model.trainBatch(mlp.X, mlp.y, mlp.order.slice(s, s + 32));
    }
    mlp.epoch++;
    const ev = mlp.model.evaluate(mlp.X, mlp.y);
    mlp.hist.push({ e: mlp.epoch, loss: ev.loss });
  }

  function curveMAE() {
    const out = new Float64Array(2);
    let mae = 0;
    for (let n = 1; n <= 90; n++) {
      mlp.model.predict([n / 90], out);
      mae += Math.abs(out[1] - mlp.freq[n - 1]);
    }
    return mae / 90;
  }

  /** 用学到的曲线反推期望抽数 */
  function expectFromModel() {
    const out = new Float64Array(2);
    let survive = 1, E = 0;
    for (let n = 1; n <= 90; n++) {
      mlp.model.predict([n / 90], out);
      const p = Math.min(1, Math.max(0, out[1]));
      E += n * survive * p;
      survive *= 1 - p;
    }
    return E;
  }

  function renderMLPDynamic() {
    const m = mlp.model;
    const ns = [], pred = [], truth = [];
    const out = new Float64Array(m.K);
    for (let n = 1; n <= 90; n++) {
      ns.push(n); truth.push(G.p5(n));
      m.predict([n / 90], out);
      pred.push(out[1]);
    }
    C.plot($('fitChart'), [
      { type: 'points', x: ns, y: mlp.freq, color: 'rgba(28,26,23,0.42)', r: 2.4 },
      { type: 'line', x: ns, y: truth, color: 'rgba(201,138,30,0.85)', width: 1.4, dash: [5, 3] },
      { type: 'line', x: ns, y: pred, color: C.P.blue, width: 2 },
      { type: 'vline', x: 74, color: 'rgba(28,26,23,0.14)', dash: [2, 3] }
    ], {
      box: { l: 46, r: 12, t: 12, b: 30 },
      xDomain: [0, 91], yDomain: [0, 1.04],
      xticks: [1, 20, 40, 60, 74, 90], yticks: [0, 0.25, 0.5, 0.75, 1],
      yFmt: t => Math.round(t * 100) + '%',
      xLabel: '垫抽数 n', yLabel: '出货概率'
    });

    // 损失曲线
    const lx = mlp.hist.map(h => h.e), ly = mlp.hist.map(h => h.loss);
    C.plot($('lossChart'), [
      { type: 'line', x: lx, y: ly, color: C.P.blue, width: 1.6 },
      { type: 'area', x: lx, y: ly, color: 'rgba(47,95,208,0.07)' }
    ], {
      box: { l: 44, r: 12, t: 10, b: 26 },
      xDomain: [0, TOTAL_EPOCHS],
      yDomain: [0, Math.max(0.02, (ly.length ? Math.max.apply(null, ly) : 0.7) * 1.05)],
      xTickCount: 5,
      yFmt: t => t.toFixed(2),
      xLabel: 'epoch', yLabel: '交叉熵'
    });

    $('mEpoch').textContent = mlp.epoch + ' / ' + TOTAL_EPOCHS;
    $('mLoss').textContent = mlp.hist.length ? fmt(mlp.hist[mlp.hist.length - 1].loss, 4) : '—';
    $('mMAE').textContent = mlp.hist.length ? fmt(curveMAE() * 100, 2) + ' pp' : '—';
    $('mExpect').textContent = mlp.hist.length ? fmt(expectFromModel(), 2) + ' 抽' : '—';
    $('mParams').textContent = m.nParams;
    $('trainBar').style.width = (mlp.epoch / TOTAL_EPOCHS * 100) + '%';
  }

  /** 参数面板：热图、单元响应、网络图、数值明细 */
  function renderMLPStatic() {
    const m = mlp.model;

    // 权重热图
    const li = +$('heatLayer').value || 0;
    const nIn = m.sizes[li], nOut = m.sizes[li + 1];
    let rows = nOut, cols = nIn, W = m.W[li], extra = '';
    if (nIn === 1) { rows = 1; cols = nOut; extra = '（列向量，横向展开）'; W = Array.from(m.W[li]); }
    const r = C.heatmap($('heatW'), W, rows, cols, { showValues: cols <= 8 && rows <= 18, pad: 3 });
    $('heatTitle').textContent = `W${['⁽¹⁾', '⁽²⁾', '⁽³⁾', '⁽⁴⁾'][li] || ''} ${rows}×${cols}${extra}`;
    $('heatScale').textContent = `|w|max = ${fmt(r.maxAbs, 2)}`;

    // 单元响应 + 参数明细
    const sel = $('unitSel').value || '0:0';
    const [ul, uu] = sel.split(':').map(Number);
    C.responseGrid($('respGrid'), m, { layer: ul, selected: uu });
    drawUnitWeights($('unitBars'), m, ul, uu);
    const nInU = m.sizes[ul];
    if (nInU === 1) {
      const z = unitSplit(m, ul, uu);
      $('unitInfo').innerHTML = `b = <b>${fmt(m.b[ul][uu], 3)}</b>　w = <b>${fmt(m.W[ul][uu], 3)}</b>　` +
        `折点 x ≈ <b>${fmt(z, 2)}</b> → n ≈ <b>${Math.round(z * 90)}</b> 抽`;
    } else {
      let sw = 0;
      for (let i = 0; i < nInU; i++) sw += Math.abs(m.W[ul][uu * nInU + i]);
      $('unitInfo').innerHTML = `接收 ${nInU} 个输入　|w| 之和 = <b>${fmt(sw, 2)}</b>　` +
        `b = <b>${fmt(m.b[ul][uu], 3)}</b>`;
    }

    // 网络图
    const means = m.meanActivations(mlp.X);
    C.network($('netGraph'), m, means);
  }

  /** 单元在输入轴上的转折点：w·x + b = 0 → x = -b/w */
  function unitSplit(model, layer, unit) {
    const w = model.W[layer][unit * model.sizes[layer]];
    if (Math.abs(w) < 1e-9) return 0.5;
    return -model.b[layer][unit] / w;
  }

  /** 选中单元的参数条形：输入权重若干行 + 偏置一行 */
  function drawUnitWeights(canvas, model, layer, unit) {
    const nIn = model.sizes[layer];
    const W = model.W[layer];
    const vals = [];
    let maxAbs = 1e-6;
    for (let i = 0; i < nIn; i++) {
      const v = W[unit * nIn + i];
      vals.push(v);
      maxAbs = Math.max(maxAbs, Math.abs(v));
    }
    const b = model.b[layer][unit];
    maxAbs = Math.max(maxAbs, Math.abs(b));
    const rows = nIn + 1;

    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const rowH = h / rows;
    const midX = 66;
    const scale = (w - midX - 58) / maxAbs;
    ctx.strokeStyle = 'rgba(28,26,23,0.2)';
    ctx.beginPath(); ctx.moveTo(midX, 2); ctx.lineTo(midX, h - 2); ctx.stroke();
    ctx.font = '9px "Cascadia Mono", Consolas, monospace';

    const drawRow = (label, v, cy, isBias) => {
      const len = v * scale;
      ctx.fillStyle = isBias ? 'rgba(28,26,23,0.6)'
        : (v >= 0 ? 'rgba(47,95,208,0.75)' : 'rgba(201,138,30,0.75)');
      const barH = Math.max(1.5, rowH * 0.55);
      ctx.fillRect(len >= 0 ? midX : midX + len, cy - barH / 2, Math.abs(len), barH);
      ctx.fillStyle = 'rgba(125,118,107,0.95)';
      ctx.textAlign = 'right';
      ctx.fillText(label, midX - 6, cy);
      ctx.fillStyle = 'rgba(28,26,23,0.75)';
      ctx.textAlign = 'left';
      ctx.fillText((v >= 0 ? '+' : '') + v.toFixed(2), midX + 5, cy);
    };

    for (let i = 0; i < nIn; i++) drawRow(nIn === 1 ? 'w·x' : 'a' + i, vals[i], i * rowH + rowH / 2, false);
    drawRow('b（偏置）', b, nIn * rowH + rowH / 2, true);
  }

  function buildHeatLayerOptions() {
    const m = mlp.model;
    const sel = $('heatLayer');
    const prev = sel.value;
    sel.innerHTML = '';
    for (let l = 0; l < m.L; l++) {
      const o = document.createElement('option');
      o.value = l;
      o.textContent = `W${['⁽¹⁾', '⁽²⁾', '⁽³⁾', '⁽⁴⁾'][l] || ''}　${m.sizes[l + 1]}×${m.sizes[l]}`;
      sel.appendChild(o);
    }
    // 默认看最大的那个矩阵
    let best = 0, bestN = -1;
    for (let l = 0; l < m.L; l++) {
      const sz = m.sizes[l] * m.sizes[l + 1];
      if (sz > bestN) { bestN = sz; best = l; }
    }
    sel.value = (prev !== '' && +prev < m.L) ? prev : best;
  }

  function buildUnitOptions() {
    const m = mlp.model;
    const sel = $('unitSel');
    sel.innerHTML = '';
    for (let l = 0; l < m.L - 1; l++) {
      const g = document.createElement('optgroup');
      g.label = `隐藏层 ${l + 1}（${m.sizes[l + 1]} 个单元）`;
      for (let u = 0; u < m.sizes[l + 1]; u++) {
        const o = document.createElement('option');
        o.value = l + ':' + u;
        o.textContent = `L${l + 1} #${u}`;
        g.appendChild(o);
      }
      sel.appendChild(g);
    }
    if (!sel.options.length) {
      const o = document.createElement('option');
      o.value = '0:0'; o.textContent = '无隐藏层';
      sel.appendChild(o);
      sel.disabled = true;
    } else {
      sel.disabled = false;
    }
  }

  /* =========================================================
   * 5. 玩具实验：二维决策边界
   * ========================================================= */
  const toy = { key: 'xor', noise: 0.1, running: false, model: null, ds: null, epoch: 0 };

  function resetToy() {
    toy.ds = makeDataset(toy.key, 240, toy.noise, 2026);
    toy.model = new MLP({
      sizes: [2, 8, 8, toy.ds.K], activation: 'tanh',
      optimizer: 'momentum', lr: 0.1, seed: 31
    });
    toy.epoch = 0;
    toy.running = false;
    $('btnToy').textContent = '开始训练';
    $('toyBar').style.width = '0%';
    renderToy(true);
  }

  function renderToy(full) {
    C.boundary($('toyBoundary'), toy.model, toy.ds, { res: 80 });
    const ev = toy.model.evaluate(toy.ds.X, toy.ds.y);
    $('toyStat').textContent = `epoch ${toy.epoch}　loss ${fmt(ev.loss, 3)}　acc ${pct(ev.acc)}`;
    if (full) {
      C.heatmap($('toyW1'), toy.model.W[0], toy.model.sizes[1], toy.model.sizes[0], { pad: 2 });
      C.heatmap($('toyW2'), toy.model.W[1], toy.model.sizes[2], toy.model.sizes[1], { pad: 2 });
      C.heatmap($('toyW3'), toy.model.W[2], toy.model.sizes[3], toy.model.sizes[2], { pad: 2 });
    }
  }

  /* =========================================================
   * 6. 动态过程：一次迭代的四个阶段
   * ========================================================= */
  const dynData = buildFreqData(6000, 99);
  const SUP = ['', '¹', '²', '³', '⁴'];
  const DYN_NOTES = {
    forward: '信号从左边进来，每层先算净输入 <b>z = Wa + b</b>，再过多一次激活函数。蓝色光点走完一列，那一列就“算好了”。',
    loss: '<b>p̂</b> 与标签 <b>t</b> 的差距被压成一个数。这里的 t 是 6000 次抽样得到的经验频率，所以是软标签。',
    backward: '<b>δ</b> 从输出层往回走：先乘 <b>Wᵀ</b> 把误差分摊给上一层，再乘 σ′(z)——净输入离 0 越远、梯度越小，这个单元就少担一点责任。',
    update: '前面三个阶段参数一动没动，只有这一下真的写了进去。绿色越亮，说明这个权重被改得越多。'
  };

  let dyn = null, dynLastKey = '', dynFitKey = '', dynAuto = false;

  function newDynModel(lr) {
    return new MLP({
      sizes: [1, 16, 16, 2], activation: 'tanh', optimizer: 'adam',
      lr: lr == null ? 0.02 : lr, seed: 11
    });
  }

  /** 数值账本：单元 × 每层的 z / a / δ + 输出概率 */
  function renderDynTrace(st) {
    const m = st.model, pass = st.pass, L = m.L;
    const box = $('traceTable');
    box.innerHTML = '';
    box.style.gridTemplateColumns = '62px ' + `repeat(${3 * (L - 1) + 2}, minmax(0, 1fr))`;

    const h = (txt, on) => el('div', 'hd' + (on ? ' on' : ''), txt);
    box.appendChild(h('单元'));
    for (let l = 1; l < L; l++) {
      const f = st.focus === l;
      box.appendChild(h('z' + SUP[l], f && st.key === 'forward'));
      box.appendChild(h('a' + SUP[l], f && st.key === 'forward'));
      box.appendChild(h('δ' + SUP[l], f && st.key === 'backward'));
    }
    box.appendChild(h('p 出金', st.key === 'loss'));
    box.appendChild(h('输入 x'));

    const rows = Math.max.apply(null, m.sizes.slice(1, L).concat([1]));
    for (let j = 0; j < rows; j++) {
      box.appendChild(el('div', 'hd', '#' + j));
      for (let l = 1; l < L; l++) {
        const z = pass.Z[l], a = pass.A[l], d = pass.delta[l];
        const fz = st.focus === l && st.key === 'forward';
        const fb = st.focus === l && st.key === 'backward';
        box.appendChild(el('div', fz ? 'on' : null, j < a.length ? fmt(z[j], 2) : '·'));
        box.appendChild(el('div', fz ? 'on' : null, j < a.length ? fmt(a[j], 2) : '·'));
        box.appendChild(el('div', fb ? 'on' : null, j < a.length ? fmt(d[j], 2) : '·'));
      }
      box.appendChild(el('div', null, j < m.K ? fmt(pass.p[j], 3) : ''));
      box.appendChild(el('div', null, j === 0 ? fmt(pass.x[0], 3) : ''));
    }

    box.appendChild(el('div', 'hd', '合计'));
    for (let l = 1; l < L; l++) {
      box.appendChild(el('div', null, ''));
      box.appendChild(el('div', null, ''));
      box.appendChild(el('div', 'hd', '‖δ‖ ' + fmt(pass.deltaNorm[l], 2)));
    }
    box.appendChild(el('div', 'val final' + (st.key === 'loss' ? ' on' : ''), 'L ' + fmt(pass.loss, 3)));
    box.appendChild(el('div', null, ''));
  }

  /** p̂(n) 曲线随每次更新被改写 */
  function drawDynFit(st) {
    const key = st.iter + '|' + st.key;
    if (key === dynFitKey) return;
    dynFitKey = key;

    const m = st.model;
    const ns = [], pred = [], truth = [];
    const out = new Float64Array(m.K);
    for (let n = 1; n <= 90; n++) {
      ns.push(n); truth.push(G.p5(n));
      m.predict([n / 90], out);
      pred.push(out[1]);
    }
    $('dynFitTag').textContent = '更新 ' + st.iter + ' 次';

    C.plot($('dynFit'), [
      { type: 'points', x: ns, y: dynData.freq, color: 'rgba(28,26,23,0.32)', r: 2.1 },
      { type: 'line', x: ns, y: truth, color: 'rgba(201,138,30,0.8)', width: 1.3, dash: [5, 3] },
      { type: 'line', x: ns, y: pred, color: C.P.blue, width: 1.9 },
      { type: 'vline', x: st.n, color: 'rgba(192,57,43,0.35)', dash: [2, 4] },
      { type: 'points', x: [st.n], y: [pred[st.n - 1]], color: C.P.red, r: 3.8 }
    ], {
      box: { l: 46, r: 12, t: 10, b: 26 },
      xDomain: [0, 91], yDomain: [0, 1.05],
      xticks: [1, 20, 40, 60, 74, 90], yticks: [0, 0.5, 1],
      yFmt: t => Math.round(t * 100) + '%',
      xLabel: '垫抽数 n', yLabel: 'p̂(n)'
    });
  }

  function onDynFrame(st) {
    const key = st.key + '|' + st.n + '|' + st.focus + '|' + st.iter + '|' + (st.dW ? 'u' : '');
    if (key !== dynLastKey) {
      dynLastKey = key;
      $('dynN').textContent = st.n + ' 抽';
      $('dynX').textContent = fmt(st.pass.x[0], 3);
      $('dynP').textContent = pct(st.pass.p[1], 1);
      $('dynT').textContent = pct(dynData.y[st.n][1], 1);
      $('dynLoss').textContent = fmt(st.pass.loss, 3);
      $('dynDelta').textContent = st.pass.deltaNorm.slice(1).map(v => fmt(v, 2)).join(' / ');
      $('dynDw').textContent = st.dW ? st.dwMax.toExponential(1) : '—';
      $('traceTag').textContent = `n = ${st.n}　第 ${st.round} 轮　迭代 ${st.iter + 1}`;
      renderDynTrace(st);
      for (const c of $('phaseTrack').children) c.classList.toggle('on', c.dataset.k === st.key);
      if (DYN_NOTES[st.key]) $('dynNote').innerHTML = DYN_NOTES[st.key];
      $('dynBar').style.width = ((st.iter % 90) / 90 * 100).toFixed(1) + '%';
    }
    drawDynFit(st);
  }

  function restartDyn(lr) {
    dyn.playing = false;
    $('btnDyn').textContent = '播放';
    dyn.reset(newDynModel(lr));
    dynLastKey = ''; dynFitKey = '';
    onDynFrame(dyn.state());
  }


  /* =========================================================
   * 7. 全局动画循环
   * ========================================================= */
  let frameCount = 0, lastTs = 0;
  function loop(ts) {
    const dt = lastTs ? Math.min(64, (ts || 0) - lastTs) : 16;
    if (ts) lastTs = ts;
    frameCount++;
    if (dyn) dyn.tick(dt);
    if (mlp.running) {
      for (let k = 0; k < 2 && mlp.epoch < TOTAL_EPOCHS; k++) trainEpoch();
      if (mlp.epoch >= TOTAL_EPOCHS) {
        mlp.running = false;
        $('btnTrain').textContent = '重新训练';
        $('trainHint').innerHTML = `训练完成：600 个 epoch，最终交叉熵 <b>${fmt(mlp.hist[mlp.hist.length - 1].loss, 4)}</b>，` +
          `平均绝对误差 <b>${fmt(curveMAE() * 100, 2)} pp</b>，反推期望 <b>${fmt(expectFromModel(), 2)} 抽</b>（理论 62.30）。`;
      }
      renderMLPDynamic();
      if (frameCount % 4 === 0) {
        const sel = $('unitSel').value || '0:0';
        const [ul, uu] = sel.split(':').map(Number);
        C.responseGrid($('respGrid'), mlp.model, { layer: ul, selected: uu });
      }
      if (frameCount % 12 === 0) renderMLPStatic();
    }
    if (toy.running) {
      for (let e = 0; e < 3; e++) {
        const idx = [...Array(toy.ds.X.length).keys()];
        for (let i = idx.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          const t = idx[i]; idx[i] = idx[j]; idx[j] = t;
        }
        for (let s = 0; s < idx.length; s += 32) toy.model.trainBatch(toy.ds.X, toy.ds.y, idx.slice(s, s + 32));
        toy.epoch++;
      }
      $('toyBar').style.width = Math.min(100, toy.epoch / 600 * 100) + '%';
      if (frameCount % 2 === 0) renderToy(toy.epoch % 18 === 0);
      if (toy.epoch >= 600) {
        toy.running = false;
        $('btnToy').textContent = '重新训练';
        renderToy(true);
      }
    }
    requestAnimationFrame(loop);
  }

  /* =========================================================
   * 8. 事件绑定
   * ========================================================= */
  function bind() {
    // 机制
    $('tgSoft').addEventListener('change', drawCurve);
    $('tgHard').addEventListener('change', drawCurve);

    // 模拟
    $('btnSim').addEventListener('click', runSim);
    $('simN').addEventListener('change', runSim);
    $('simMode').addEventListener('change', runSim);

    // MLP 结构
    $('cfgHidden').addEventListener('change', e => {
      const v = e.target.value;
      mlp.hidden = v ? v.split(',').map(Number) : [];
      resetMLP(false);
    });
    $('cfgAct').addEventListener('change', e => { mlp.act = e.target.value; resetMLP(true); });
    $('cfgInit').addEventListener('change', e => { mlp.init = e.target.value; resetMLP(true); });
    $('cfgOpt').addEventListener('change', e => { mlp.opt = e.target.value; resetMLP(true); });
    $('cfgLR').addEventListener('input', e => {
      mlp.lr = Math.pow(10, +e.target.value);
      $('outLR').textContent = mlp.lr.toFixed(3);
      if (mlp.model) mlp.model.setLR(mlp.lr);
    });
    $('cfgTrials').addEventListener('change', e => { mlp.trials = +e.target.value; resetMLP(false); });

    $('btnTrain').addEventListener('click', () => {
      if (mlp.epoch >= TOTAL_EPOCHS) { mlp.seedShift++; resetMLP(true); }
      mlp.running = !mlp.running;
      $('btnTrain').textContent = mlp.running ? '暂停' : '继续训练';
    });
    $('btnReset').addEventListener('click', () => { mlp.seedShift++; resetMLP(true); });

    $('heatLayer').addEventListener('change', renderMLPStatic);
    $('unitSel').addEventListener('change', renderMLPStatic);

    // 玩具
    $('toyData').addEventListener('change', e => { toy.key = e.target.value; resetToy(); });
    $('toyNoise').addEventListener('input', e => {
      toy.noise = +e.target.value;
      $('toyNoiseOut').textContent = toy.noise.toFixed(2);
      resetToy();
    });
    $('btnToy').addEventListener('click', () => {
      if (toy.epoch >= 600) resetToy();
      toy.running = !toy.running;
      $('btnToy').textContent = toy.running ? '暂停' : '继续训练';
    });
    $('btnToyReset').addEventListener('click', resetToy);

    // 动态过程
    $('btnDyn').addEventListener('click', () => {
      dyn.playing = !dyn.playing;
      $('btnDyn').textContent = dyn.playing ? '暂停' : '继续播放';
    });
    $('btnDynStep').addEventListener('click', () => {
      dyn.playing = false;
      $('btnDyn').textContent = '继续播放';
      dyn.step();
    });
    $('btnDynReset').addEventListener('click', () => restartDyn(+$('dynLR').value));
    $('dynSpeed').addEventListener('change', e => { dyn.speed = +e.target.value; });
    $('dynLR').addEventListener('change', e => restartDyn(+e.target.value));

    // 滚到 MLP 实验区就开始播
    if ('IntersectionObserver' in window) {
      const io = new IntersectionObserver(es => {
        for (const e of es) {
          if (e.isIntersecting && !dynAuto) {
            dynAuto = true;
            dyn.playing = true;
            $('btnDyn').textContent = '暂停';
            io.disconnect();
          }
        }
      }, { threshold: 0.35 });
      io.observe($('dynNet'));
    }

    // 尺寸变化
    let rt;
    window.addEventListener('resize', () => {
      clearTimeout(rt);
      rt = setTimeout(() => {
        drawHero(); drawCurve(); runSim(); drawUpHist();
        renderMLPStatic(); renderMLPDynamic();
        renderToy(true);
        dynLastKey = ''; dynFitKey = '';
        dyn.draw(); onDynFrame(dyn.state());
      }, 180);
    });
  }

  /* =========================================================
   * 9. 启动
   * ========================================================= */
  function init() {
    drawHero();
    drawCurve();
    runSim();
    drawUpHist();
    resetMLP(false);
    resetToy();
    // 让 MLP 先有一个可见的"未训练"曲线
    renderMLPDynamic();

    // 动态过程
    dyn = new window.Dyn({
      canvas: $('dynNet'),
      model: newDynModel(0.02),
      data: dynData,
      onFrame: onDynFrame
    });
    dyn.speed = +$('dynSpeed').value;
    dyn.draw();
    onDynFrame(dyn.state());

    // 便于测试/演示的自动启动：index.html?autostart
    if (location.search.indexOf('autostart') >= 0) {
      mlp.running = true; toy.running = true;
      dyn.playing = true;
      $('btnTrain').textContent = '暂停';
      $('btnToy').textContent = '暂停';
      $('btnDyn').textContent = '暂停';
    }
    bind();
    requestAnimationFrame(loop);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
