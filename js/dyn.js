/* =============================================================
 * dyn.js —— 把一次训练迭代做成可播放的动画
 *
 * 一次迭代 = 一个样本，四个阶段：
 *   ① 前向传播  z = Wa + b, a = σ(z)，信号从左往右逐层点亮
 *   ② 计算损失  L = −Σ t·log p̂，输出层两节点向外扩散
 *   ③ 反向传播  δ 从输出层逐层回传，误差从右往左
 *   ④ 更新参数  w ← w − η·∂L/∂w，边的颜色按 |Δw| 闪烁
 *
 * 所有中间量都来自 MLP.gradSample()，画出来的数字与训练时用的完全一致。
 * ============================================================= */
(function (root) {
  'use strict';

  const C = root.Charts;
  const FONT = '"Segoe UI","Microsoft YaHei",system-ui,sans-serif';
  const MONO = '"Cascadia Mono","Consolas",monospace';

  const PHASES = [
    { key: 'forward',  label: '① 前向传播', dur: 1.30, eq: 'z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾　a⁽ˡ⁾ = σ(z⁽ˡ⁾)' },
    { key: 'loss',     label: '② 计算损失', dur: 0.70, eq: 'L = −[ t·log p̂ + (1−t)·log(1−p̂) ]' },
    { key: 'backward', label: '③ 反向传播', dur: 1.25, eq: 'δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀδ⁽ˡ⁺¹⁾ ⊙ σ′(z⁽ˡ⁾)' },
    { key: 'update',   label: '④ 更新参数', dur: 0.95, eq: 'w ← w − η·∂L/∂w　（Adam 自适应步长）' }
  ];

  const COL = {
    blue: [47, 95, 208],
    gold: [201, 138, 30],
    red: [192, 57, 43],
    green: [46, 139, 110],
    ink: [28, 26, 23]
  };
  const rgba = (c, a) => 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + Math.max(0, Math.min(1, a)).toFixed(3) + ')';
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);

  function Dyn(opts) {
    this.canvas = opts.canvas;
    this.model = opts.model;
    this.data = opts.data;              // { X, y }
    this.onFrame = opts.onFrame || function () {};
    this.speed = 0.6;
    this.playing = false;
    this.iter = 0;                      // 已完成的参数更新次数
    this.round = 1;                     // 第几轮（epoch）
    this.order = [];
    this.oi = 0;
    this.pi = 0;
    this.t = 0;
    this.dirty = true;
    this.shuffle();
    this.enter();
  }

  Dyn.prototype.shuffle = function () {
    const a = this.order = this.order.length === this.data.X.length
      ? this.order : [...Array(this.data.X.length).keys()];
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const t = a[i]; a[i] = a[j]; a[j] = t;
    }
  };

  /** 进入一个新样本：算好这次迭代的全部中间量（不改参数） */
  Dyn.prototype.enter = function () {
    const m = this.model;
    const n = this.order[this.oi];
    this.n = n;
    this.pass = m.gradSample(this.data.X[n], this.data.y[n]);
    this.snapW = m.W.map(w => Float64Array.from(w));
    this.dW = null;
    this.dwMax = 0;
    this.pi = 0;
    this.t = 0;
    this.dirty = true;
    this.emit();
  };

  /** 阶段 ④ 开始时真正改一次参数，并记下每个权重被改了多少 */
  Dyn.prototype.applyUpdate = function () {
    const m = this.model;
    m.trainBatch(this.data.X, this.data.y, [this.n]);
    const dW = [];
    let mx = 0;
    for (let l = 0; l < m.L; l++) {
      const d = new Float64Array(m.W[l].length);
      for (let k = 0; k < d.length; k++) {
        const v = m.W[l][k] - this.snapW[l][k];
        d[k] = v;
        if (Math.abs(v) > mx) mx = Math.abs(v);
      }
      dW.push(d);
    }
    this.dW = dW;
    this.dwMax = mx;
  };

  Dyn.prototype.advance = function () {
    if (PHASES[this.pi].key === 'update') {
      this.oi++;
      if (this.oi >= this.order.length) { this.oi = 0; this.round++; this.shuffle(); }
      this.iter++;
      this.enter();
      return;
    }
    this.pi++;
    if (PHASES[this.pi].key === 'update') this.applyUpdate();
    this.t = 0;
    this.dirty = true;
  };

  Dyn.prototype.step = function () {
    this.advance();
    this.draw();
    this.emit();
  };

  Dyn.prototype.tick = function (dt) {
    if (!this.playing) {
      if (this.dirty) { this.draw(); this.dirty = false; }
      return;
    }
    let budget = (dt / 1000) * this.speed;
    let guard = 0;
    while (budget > 0 && guard++ < 60) {
      const ph = PHASES[this.pi];
      const remain = (1 - this.t) * ph.dur;
      if (budget >= remain) {
        budget -= remain;
        this.advance();
      } else {
        this.t += budget / ph.dur;
        budget = 0;
      }
    }
    this.draw();
    this.emit();
  };

  Dyn.prototype.reset = function (model) {
    if (model) this.model = model;
    this.iter = 0;
    this.round = 1;
    this.oi = 0;
    this.order = [];
    this.shuffle();
    this.enter();
    this.draw();
  };

  /* ---------------- 状态：给 DOM 用 ---------------- */
  Dyn.prototype.state = function () {
    const ph = PHASES[this.pi];
    const L = this.model.L;
    let focus = 0;
    if (ph.key === 'forward') focus = clamp(1 + Math.floor(this.t * L), 1, L);
    else if (ph.key === 'loss') focus = L;
    else if (ph.key === 'backward') focus = clamp(L - Math.floor(this.t * L), 1, L);
    return {
      model: this.model,
      key: ph.key, label: ph.label, eq: ph.eq, t: this.t,
      iter: this.iter, round: this.round, n: this.n, focus: focus,
      pass: this.pass, dW: this.dW, dwMax: this.dwMax,
      lr: this.model.opt.lr, playing: this.playing
    };
  };

  Dyn.prototype.emit = function () { this.onFrame(this.state()); };

  /* ---------------- 绘制 ---------------- */
  Dyn.prototype.draw = function () {
    const { ctx, w, h } = C.setup(this.canvas);
    const m = this.model, L = m.L;
    const padL = 56, padR = 78, padT = 58, padB = 46;
    const iw = Math.max(40, w - padL - padR), ih = Math.max(40, h - padT - padB);
    const colX = i => padL + iw * i / L;
    const cy = padT + ih / 2;
    const pos = m.sizes.map(nn => {
      const gap = Math.min(26, ih / Math.max(1, nn - 1 || 1));
      const tot = gap * (nn - 1);
      return Array.from({ length: nn }, (_, j) => cy - tot / 2 + j * gap);
    });

    let maxAbs = 1e-6;
    for (let l = 0; l < L; l++) for (const v of m.W[l]) maxAbs = Math.max(maxAbs, Math.abs(v));

    const ph = PHASES[this.pi].key;
    const t = this.t;
    const pass = this.pass;
    const fwdP = l => clamp(t * L - l, 0, 1);
    const bwdP = l => clamp(t * L - (L - 1 - l), 0, 1);

    /* ---- 顶部信息 ---- */
    ctx.textBaseline = 'middle';
    ctx.font = '12px ' + MONO;
    ctx.textAlign = 'left';
    ctx.fillStyle = rgba(COL.ink, 0.75);
    ctx.fillText('第 ' + this.round + ' 轮 · 迭代 ' + (this.iter + 1), padL - 10, 18);

    ctx.textAlign = 'right';
    const phColor = ph === 'forward' ? COL.blue : ph === 'loss' ? COL.gold
      : ph === 'backward' ? COL.red : COL.green;
    ctx.fillStyle = rgba(phColor, 0.95);
    ctx.font = '600 12px ' + FONT;
    ctx.fillText(PHASES[this.pi].label, w - padR + 10, 18);

    ctx.font = '11px ' + MONO;
    ctx.fillStyle = rgba(COL.ink, 0.4);
    ctx.textAlign = 'center';
    ctx.fillText(PHASES[this.pi].eq, w / 2, h - 8);

    /* ---- 边 ---- */
    for (let l = 0; l < L; l++) {
      const nIn = m.sizes[l], nOut = m.sizes[l + 1];
      const fp = fwdP(l), bp = bwdP(l);
      const actF = ph === 'forward' && fp > 0 && fp < 1;
      const doneF = ph === 'forward' && fp >= 1;
      const actB = ph === 'backward' && bp > 0 && bp < 1;
      const doneB = ph === 'backward' && bp >= 1;

      for (let j = 0; j < nOut; j++) {
        for (let i = 0; i < nIn; i++) {
          const v = m.W[l][j * nIn + i];
          const mag = Math.min(1, Math.abs(v) / maxAbs);
          let color = v >= 0 ? COL.blue : COL.gold;
          let alpha = 0.03 + 0.26 * mag;
          let width = 0.45 + 1.0 * mag;

          if (actF) { color = COL.blue; alpha = 0.22 + 0.62 * (0.3 + 0.7 * mag); width += 0.35; }
          else if (doneF) { alpha = 0.05 + 0.18 * mag; }

          if (actB) { color = COL.red; alpha = 0.28 + 0.62 * (0.25 + 0.75 * mag); width += 0.3; }
          else if (doneB) { alpha = 0.04 + 0.1 * mag; }

          if (ph === 'update' && this.dW) {
            const dl = Math.abs(this.dW[l][j * nIn + i]) / (this.dwMax || 1);
            const pulse = 0.55 + 0.45 * Math.sin(Math.PI * t);
            color = COL.green;
            alpha = 0.02 + 0.5 * Math.pow(dl, 0.7) * pulse;
            width = 0.5 + 2.4 * Math.pow(dl, 0.45);
          }

          if (ph === 'loss') alpha *= 0.4;
          if (alpha < 0.012) continue;

          ctx.strokeStyle = rgba(color, alpha);
          ctx.lineWidth = width;
          ctx.beginPath();
          ctx.moveTo(colX(l), pos[l][i]);
          ctx.lineTo(colX(l + 1), pos[l + 1][j]);
          ctx.stroke();
        }
      }
    }

    /* ---- 沿边流动的光点 ---- */
    if (ph === 'forward' || ph === 'backward') {
      for (let l = 0; l < L; l++) {
        const s = ph === 'forward' ? fwdP(l) : 1 - bwdP(l);
        if (s <= 0.02 || s >= 0.98) continue;
        const nIn = m.sizes[l], nOut = m.sizes[l + 1];
        const cc = ph === 'forward' ? COL.blue : COL.red;
        for (let j = 0; j < nOut; j++) {
          for (let i = 0; i < nIn; i++) {
            const mag = Math.abs(m.W[l][j * nIn + i]) / maxAbs;
            if (mag < 0.42) continue;
            const x0 = colX(l), y0 = pos[l][i], x1 = colX(l + 1), y1 = pos[l + 1][j];
            const px = x0 + (x1 - x0) * s, py = y0 + (y1 - y0) * s;
            ctx.beginPath();
            ctx.arc(px, py, 1.4 + 1.2 * mag, 0, Math.PI * 2);
            ctx.fillStyle = rgba(cc, 0.55 + 0.4 * mag);
            ctx.fill();
          }
        }
      }
    }

    /* ---- 节点 ---- */
    for (let l = 0; l <= L; l++) {
      const isOut = l === L;
      const r = l === 0 || isOut ? 8.5 : 6.2;
      let lit;
      if (ph === 'forward') lit = l === 0 ? 1 : fwdP(l - 1);
      else if (ph === 'loss') lit = isOut ? 1 : 0.4;
      else if (ph === 'backward') lit = isOut ? 1 : bwdP(l);
      else lit = 0.5;

      const dArr = pass.delta[l];
      let dMax = 1e-9;
      if (dArr) for (let k = 0; k < dArr.length; k++) dMax = Math.max(dMax, Math.abs(dArr[k]));

      for (let j = 0; j < m.sizes[l]; j++) {
        const x = colX(l), y = pos[l][j];
        const a = pass.A[l][j];
        const s = isOut ? a : Math.abs(a) / (m.act.name === 'ReLU' ? Math.max(0.5, 1) : 1);
        const glow = clamp(lit, 0, 1) * clamp(s, 0, 1);

        // 底
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fillStyle = isOut ? rgba(COL.blue, 0.1 + 0.75 * glow) : rgba([247, 244, 236], 1);
        ctx.fill();
        // 激活染色
        if (!isOut) {
          ctx.beginPath();
          ctx.arc(x, y, r, 0, Math.PI * 2);
          ctx.fillStyle = rgba(COL.blue, 0.06 + 0.62 * glow * (a >= 0 ? 1 : 0.5));
          ctx.fill();
        }
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.strokeStyle = rgba(COL.ink, 0.25 + 0.25 * glow);
        ctx.lineWidth = 1;
        ctx.stroke();

        // 反向传播：δ 大小画成外环
        if (ph === 'backward' && dArr) {
          const dn = Math.abs(dArr[j]) / dMax;
          ctx.beginPath();
          ctx.arc(x, y, r + 3 + 3.2 * dn, 0, Math.PI * 2);
          ctx.strokeStyle = rgba(COL.red, 0.2 + 0.65 * dn);
          ctx.lineWidth = 1 + 1.6 * dn;
          ctx.stroke();
        }
      }
    }

    /* ---- 损失阶段：输出层向外扩散的涟漪 ---- */
    if (ph === 'loss') {
      const ox = colX(L);
      for (let k = 0; k < m.K; k++) {
        for (let q = 0; q < 2; q++) {
          const rr = 9 + (t + q * 0.5) % 1 * 24;
          ctx.beginPath();
          ctx.arc(ox, pos[L][k], rr, 0, Math.PI * 2);
          ctx.strokeStyle = rgba(COL.gold, 0.45 * (1 - ((t + q * 0.5) % 1)));
          ctx.lineWidth = 1.2;
          ctx.stroke();
        }
      }
      ctx.font = '600 12px ' + MONO;
      ctx.fillStyle = rgba(COL.gold, 0.9);
      ctx.textAlign = 'left';
      ctx.fillText('L = ' + pass.loss.toFixed(3), ox + 16, pos[L][0] - 14);
    }

    /* ---- 输出层概率标签 ---- */
    if (ph !== 'backward') {
      ctx.font = '11px ' + MONO;
      ctx.textAlign = 'left';
      for (let k = 0; k < m.K; k++) {
        const p = pass.p[k];
        ctx.fillStyle = rgba(COL.blue, 0.85);
        ctx.fillText((p * 100).toFixed(1) + '%', colX(L) + 14, pos[L][k]);
      }
    }

    /* ---- 输入值标签（画在节点下方，避免被左边缘裁掉） ---- */
    ctx.font = '11px ' + MONO;
    ctx.textAlign = 'center';
    ctx.fillStyle = rgba(COL.ink, 0.6);
    ctx.fillText('x = ' + pass.x[0].toFixed(3), colX(0), pos[0][0] + 24);
    ctx.fillStyle = rgba(COL.ink, 0.42);
    ctx.font = '10px ' + FONT;
    ctx.fillText('n = ' + this.n, colX(0), pos[0][0] + 38);

    /* ---- 层标签 ---- */
    ctx.font = '10px ' + FONT;
    ctx.textAlign = 'center';
    ctx.fillStyle = rgba(COL.ink, 0.42);
    const names = ['输入 · 1', ...Array.from({ length: L - 1 }, (_, i) => '隐藏 ' + (i + 1) + ' · ' + m.sizes[i + 1]), '输出 · ' + m.K];
    for (let l = 0; l <= L; l++) {
      ctx.fillStyle = ph === 'forward' && l > 0 && fwdP(l - 1) >= 1 ? rgba(COL.blue, 0.7) : rgba(COL.ink, 0.42);
      ctx.fillText(names[l], colX(l), h - 26);
    }

    /* ---- 左下角：参数更新幅度 ---- */
    if (this.dW) {
      ctx.textAlign = 'left';
      ctx.font = '10px ' + MONO;
      ctx.fillStyle = rgba(COL.green, 0.85);
      ctx.fillText('|Δw|max = ' + this.dwMax.toExponential(1), padL - 10, h - 26);
    }
  };

  root.Dyn = Dyn;
  root.DynPhases = PHASES;
})(window);
