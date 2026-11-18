/* =============================================================
 * charts.js —— 所有图表都用 Canvas 手写，不依赖任何图表库
 * 这样离线也能打开，且每个像素都由我们控制
 * ============================================================= */
(function (root) {
  'use strict';

  const P = {
    ink: '#1c1a17',
    muted: '#8a8378',
    grid: '#e6e2da',
    axis: '#d6d0c6',
    paper: '#fffdf9',
    blue: '#2f5fd0',
    blueSoft: 'rgba(47,95,208,0.12)',
    gold: '#c98a1e',
    goldSoft: 'rgba(201,138,30,0.16)',
    green: '#2e8b6e',
    red: '#c0392b',
    purple: '#7d54b4'
  };

  const FONT = '"Segoe UI", "Microsoft YaHei", system-ui, sans-serif';
  const MONO = '"Cascadia Mono", "Consolas", monospace';

  function setup(canvas) {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = canvas.clientWidth || canvas.parentNode.clientWidth || 400;
    const h = canvas.clientHeight || 240;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.textBaseline = 'middle';
    return { ctx, w, h };
  }

  function niceTicks(min, max, count) {
    const span = max - min;
    if (span <= 0) return [min];
    const raw = span / count;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    let step = mag;
    if (norm > 5) step = 10 * mag;
    else if (norm > 2) step = 5 * mag;
    else if (norm > 1) step = 2 * mag;
    const ticks = [];
    for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) ticks.push(+v.toFixed(10));
    return ticks;
  }

  /**
   * 通用折线/柱状图
   * layers: [{type:'line'|'area'|'bars'|'points'|'vline'|'hline', ...}]
   */
  function plot(canvas, layers, opts) {
    opts = opts || {};
    const { ctx, w, h } = setup(canvas);
    const box = Object.assign({ l: 46, r: 14, t: 16, b: 30 }, opts.box);
    const iw = w - box.l - box.r;
    const ih = h - box.t - box.b;

    // ---- 计算坐标范围 ----
    let xMin = opts.xDomain ? opts.xDomain[0] : Infinity;
    let xMax = opts.xDomain ? opts.xDomain[1] : -Infinity;
    let yMin = opts.yDomain ? opts.yDomain[0] : 0;
    let yMax = opts.yDomain ? opts.yDomain[1] : -Infinity;
    if (!opts.xDomain || !opts.yDomain) {
      for (const L of layers) {
        if (L.x) for (const v of L.x) { if (v < xMin) xMin = v; if (v > xMax) xMax = v; }
        if (L.x0) for (const v of L.x0) if (v < xMin) xMin = v;
        if (L.x1) for (const v of L.x1) if (v > xMax) xMax = v;
        if (L.y) for (const v of L.y) { if (v < yMin) yMin = v; if (v > yMax) yMax = v; }
      }
    }
    if (!opts.yDomain) { yMin = Math.min(0, yMin); if (yMax <= yMin) yMax = yMin + 1; }
    if (!opts.xDomain) { if (xMax <= xMin) xMax = xMin + 1; }

    const mx = v => box.l + (v - xMin) / (xMax - xMin) * iw;
    const my = v => box.t + ih - (v - yMin) / (yMax - yMin) * ih;

    // ---- 网格与刻度 ----
    const yticks = opts.yticks || niceTicks(yMin, yMax, opts.yTickCount || 4);
    const xticks = opts.xticks || niceTicks(xMin, xMax, opts.xTickCount || 6);
    ctx.font = '11px ' + FONT;
    ctx.lineWidth = 1;

    for (const t of yticks) {
      const y = Math.round(my(t)) + 0.5;
      if (y < box.t - 1 || y > box.t + ih + 1) continue;
      ctx.strokeStyle = P.grid;
      ctx.beginPath(); ctx.moveTo(box.l, y); ctx.lineTo(box.l + iw, y); ctx.stroke();
      ctx.fillStyle = P.muted;
      ctx.textAlign = 'right';
      ctx.fillText(opts.yFmt ? opts.yFmt(t) : String(+t.toFixed(3)), box.l - 8, y);
    }
    for (const t of xticks) {
      const x = Math.round(mx(t)) + 0.5;
      if (x < box.l - 1 || x > box.l + iw + 1) continue;
      ctx.strokeStyle = P.grid;
      ctx.beginPath(); ctx.moveTo(x, box.t); ctx.lineTo(x, box.t + ih); ctx.stroke();
      ctx.fillStyle = P.muted;
      ctx.textAlign = 'center';
      ctx.fillText(opts.xFmt ? opts.xFmt(t) : String(+t.toFixed(3)), x, box.t + ih + 12);
    }

    // ---- 图层 ----
    ctx.save();
    ctx.beginPath();
    ctx.rect(box.l, box.t - 2, iw, ih + 4);
    ctx.clip();

    for (const L of layers) {
      if (L.type === 'bars') {
        ctx.fillStyle = L.color || P.blueSoft;
        for (let i = 0; i < L.y.length; i++) {
          const x0 = mx(L.x0[i]), x1 = mx(L.x1[i]);
          const y = my(L.y[i]);
          const base = my(Math.max(yMin, 0));
          ctx.fillRect(x0 + 0.6, y, Math.max(1, x1 - x0 - 1.2), base - y);
        }
        continue;
      }
      if (L.type === 'vline') {
        ctx.strokeStyle = L.color || P.muted;
        ctx.setLineDash(L.dash || [4, 4]);
        ctx.lineWidth = L.width || 1;
        const x = mx(L.x);
        ctx.beginPath(); ctx.moveTo(x, box.t); ctx.lineTo(x, box.t + ih); ctx.stroke();
        ctx.setLineDash([]);
        if (L.label) {
          ctx.fillStyle = L.color || P.muted;
          ctx.font = '10px ' + FONT;
          ctx.textAlign = 'left';
          ctx.fillText(L.label, x + 4, box.t + 8);
        }
        continue;
      }
      if (L.type === 'hline') {
        ctx.strokeStyle = L.color || P.muted;
        ctx.setLineDash(L.dash || [4, 4]);
        ctx.lineWidth = L.width || 1;
        const y = my(L.y);
        ctx.beginPath(); ctx.moveTo(box.l, y); ctx.lineTo(box.l + iw, y); ctx.stroke();
        ctx.setLineDash([]);
        continue;
      }
      if (L.type === 'area') {
        ctx.beginPath();
        ctx.moveTo(mx(L.x[0]), my(Math.max(yMin, 0)));
        for (let i = 0; i < L.x.length; i++) ctx.lineTo(mx(L.x[i]), my(L.y[i]));
        ctx.lineTo(mx(L.x[L.x.length - 1]), my(Math.max(yMin, 0)));
        ctx.closePath();
        ctx.fillStyle = L.color || P.blueSoft;
        ctx.fill();
        continue;
      }
      if (L.type === 'points') {
        for (let i = 0; i < L.x.length; i++) {
          if (L.skip && L.skip[i]) continue;
          ctx.beginPath();
          ctx.arc(mx(L.x[i]), my(L.y[i]), L.r || 3, 0, Math.PI * 2);
          ctx.fillStyle = L.color || P.blue;
          ctx.globalAlpha = L.alpha == null ? 1 : L.alpha;
          ctx.fill();
          ctx.globalAlpha = 1;
          if (L.stroke) { ctx.strokeStyle = L.stroke; ctx.lineWidth = 1; ctx.stroke(); }
        }
        continue;
      }
      // 默认 line
      ctx.beginPath();
      for (let i = 0; i < L.x.length; i++) {
        const px = mx(L.x[i]), py = my(L.y[i]);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.strokeStyle = L.color || P.blue;
      ctx.lineWidth = L.width || 2;
      ctx.setLineDash(L.dash || []);
      ctx.lineJoin = 'round';
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.restore();

    // ---- 坐标轴 ----
    ctx.strokeStyle = P.axis;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(box.l, box.t + ih + 0.5);
    ctx.lineTo(box.l + iw, box.t + ih + 0.5);
    ctx.stroke();

    if (opts.xLabel) {
      ctx.fillStyle = P.muted;
      ctx.font = '11px ' + FONT;
      ctx.textAlign = 'right';
      ctx.fillText(opts.xLabel, box.l + iw, box.t + ih + 12);
    }
    if (opts.yLabel) {
      ctx.save();
      ctx.translate(12, box.t + ih / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center';
      ctx.fillStyle = P.muted;
      ctx.font = '11px ' + FONT;
      ctx.fillText(opts.yLabel, 0, 0);
      ctx.restore();
    }
    return { mx, my, box };
  }

  /* ---------------- 权重热图 ---------------- */
  function diverging(v, maxAbs) {
    const t = Math.max(-1, Math.min(1, v / (maxAbs || 1)));
    const mix = (a, b, k) => Math.round(a + (b - a) * k);
    const neg = [47, 95, 208], pos = [201, 138, 30], zero = [244, 241, 234];
    if (t >= 0) return `rgb(${mix(zero[0], pos[0], t)},${mix(zero[1], pos[1], t)},${mix(zero[2], pos[2], t)})`;
    const k = -t;
    return `rgb(${mix(zero[0], neg[0], k)},${mix(zero[1], neg[1], k)},${mix(zero[2], neg[2], k)})`;
  }

  function heatmap(canvas, W, rows, cols, opts) {
    opts = opts || {};
    const { ctx, w, h } = setup(canvas);
    const pad = opts.pad || 4;
    const cw = (w - pad * 2) / cols;
    const ch = (h - pad * 2) / rows;
    let maxAbs = opts.maxAbs;
    if (!maxAbs) {
      maxAbs = 1e-6;
      for (let i = 0; i < W.length; i++) maxAbs = Math.max(maxAbs, Math.abs(W[i]));
    }
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = W[r * cols + c];
        ctx.fillStyle = diverging(v, maxAbs);
        ctx.fillRect(pad + c * cw, pad + r * ch, Math.ceil(cw) - 0.5, Math.ceil(ch) - 0.5);
        if (opts.showValues && cw > 26 && ch > 13) {
          ctx.fillStyle = Math.abs(v / maxAbs) > 0.62 ? '#fff' : 'rgba(28,26,23,0.62)';
          ctx.font = '9px ' + MONO;
          ctx.textAlign = 'center';
          ctx.fillText(v.toFixed(2), pad + c * cw + cw / 2, pad + r * ch + ch / 2);
        }
      }
    }
    return { maxAbs };
  }

  /* ---------------- 二维决策边界 + 散点 ---------------- */
  function classColor(i) {
    return [P.blue, P.gold, P.green, P.purple][i % 4];
  }

  function boundary(canvas, model, ds, opts) {
    opts = opts || {};
    const { ctx, w, h } = setup(canvas);
    const ext = ds.ext || 1.6;
    const res = opts.res || 84;
    const off = document.createElement('canvas');
    off.width = res; off.height = res;
    const octx = off.getContext('2d');
    const img = octx.createImageData(res, res);
    const out = new Float64Array(model.K);
    const K = model.K;

    // 类别基色
    const base = [];
    for (let k = 0; k < K; k++) {
      const c = classColor(k).replace('#', '');
      base.push([parseInt(c.slice(0, 2), 16), parseInt(c.slice(2, 4), 16), parseInt(c.slice(4, 6), 16)]);
    }

    for (let gy = 0; gy < res; gy++) {
      for (let gx = 0; gx < res; gx++) {
        const x = -ext + (gx + 0.5) / res * 2 * ext;
        const y = ext - (gy + 0.5) / res * 2 * ext;
        model.predict([x, y], out);
        let r = 0, g = 0, b = 0, mxp = 0;
        for (let k = 0; k < K; k++) {
          const p = out[k];
          mxp = Math.max(mxp, p);
          r += p * base[k][0]; g += p * base[k][1]; b += p * base[k][2];
        }
        // 置信度越低越接近纸色，于是边界自然淡出
        const conf = K > 1 ? (mxp - 1 / K) / (1 - 1 / K) : 1;
        const t = 0.16 + 0.40 * Math.pow(conf, 0.9);
        const paper = [253, 251, 246];
        const idx = (gy * res + gx) * 4;
        img.data[idx] = Math.round(paper[0] + (r - paper[0]) * t);
        img.data[idx + 1] = Math.round(paper[1] + (g - paper[1]) * t);
        img.data[idx + 2] = Math.round(paper[2] + (b - paper[2]) * t);
        img.data[idx + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(off, 0, 0, w, h);

    // 坐标轴
    ctx.strokeStyle = 'rgba(28,26,23,0.16)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2);
    ctx.moveTo(w / 2, 0); ctx.lineTo(w / 2, h);
    ctx.stroke();
    ctx.strokeStyle = 'rgba(28,26,23,0.22)';
    ctx.strokeRect(0.5, 0.5, w - 1, h - 1);

    // 散点
    for (let i = 0; i < ds.X.length; i++) {
      const px = (ds.X[i][0] + ext) / (2 * ext) * w;
      const py = (ext - ds.X[i][1]) / (2 * ext) * h;
      ctx.beginPath();
      ctx.arc(px, py, opts.dotR || 3.1, 0, Math.PI * 2);
      ctx.fillStyle = classColor(ds.y[i]);
      ctx.fill();
      ctx.strokeStyle = 'rgba(28,26,23,0.45)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }

  /* ---------------- 网络结构图（边=权重，点=平均激活） ---------------- */
  function network(canvas, model, means, opts) {
    opts = opts || {};
    const { ctx, w, h } = setup(canvas);
    const L = model.L;
    const pad = 34;
    const colX = i => pad + (w - pad * 2) * (L === 0 ? 0.5 : i / L);
    const nodeR = Math.max(4, Math.min(9, (h - 40) / 26));

    // 每层节点纵坐标
    const pos = model.sizes.map((n, li) => {
      const arr = [];
      const gap = Math.min(24, (h - 44) / Math.max(1, n - 1 || 1));
      const total = gap * (n - 1);
      for (let j = 0; j < n; j++) arr.push(h / 2 - total / 2 + j * gap);
      return arr;
    });

    let maxAbs = 1e-6;
    for (let l = 0; l < L; l++) for (const v of model.W[l]) maxAbs = Math.max(maxAbs, Math.abs(v));

    // 边
    for (let l = 0; l < L; l++) {
      const nIn = model.sizes[l], nOut = model.sizes[l + 1];
      for (let j = 0; j < nOut; j++) {
        for (let i = 0; i < nIn; i++) {
          const v = model.W[l][j * nIn + i];
          const a = 0.03 + 0.42 * Math.min(1, Math.abs(v) / maxAbs);
          ctx.strokeStyle = v >= 0
            ? `rgba(47,95,208,${a.toFixed(3)})`
            : `rgba(201,138,30,${a.toFixed(3)})`;
          ctx.lineWidth = 0.4 + 1.1 * Math.min(1, Math.abs(v) / maxAbs);
          ctx.beginPath();
          ctx.moveTo(colX(l), pos[l][i]);
          ctx.lineTo(colX(l + 1), pos[l + 1][j]);
          ctx.stroke();
        }
      }
    }

    // 节点：颜色深浅表示该层平均激活
    for (let l = 0; l <= L; l++) {
      const m = means[l];
      let mn = Infinity, mx = -Infinity;
      for (const v of m) { mn = Math.min(mn, v); mx = Math.max(mx, v); }
      for (let j = 0; j < model.sizes[l]; j++) {
        const t = mx - mn < 1e-9 ? 0.4 : (m[j] - mn) / (mx - mn);
        const c = 244 - 100 * t;
        ctx.beginPath();
        ctx.arc(colX(l), pos[l][j], nodeR, 0, Math.PI * 2);
        ctx.fillStyle = l === L
          ? `rgba(47,95,208,${(0.18 + 0.7 * t).toFixed(2)})`
          : `rgba(${Math.round(c)},${Math.round(c + 2)},${Math.round(c + 8)},1)`;
        ctx.fill();
        ctx.strokeStyle = 'rgba(28,26,23,0.28)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }

    // 层标签
    ctx.fillStyle = P.muted;
    ctx.font = '10px ' + FONT;
    ctx.textAlign = 'center';
    const names = ['输入', ...Array.from({ length: L - 1 }, (_, i) => '隐藏 ' + (i + 1)), '输出'];
    for (let l = 0; l <= L; l++) ctx.fillText(names[l] + ' · ' + model.sizes[l], colX(l), h - 8);
  }

  /* ---------------- 隐藏单元响应曲线阵列 ---------------- */
  function responseGrid(canvas, model, opts) {
    opts = opts || {};
    const { ctx, w, h } = setup(canvas);
    const layer = opts.layer == null ? 0 : opts.layer;
    const n = model.sizes[layer + 1];
    const cols = Math.min(8, n);
    const rows = Math.ceil(n / cols);
    const cw = w / cols, ch = h / rows;
    const sel = opts.selected == null ? -1 : opts.selected;

    for (let j = 0; j < n; j++) {
      const cx = (j % cols) * cw, cy = Math.floor(j / cols) * ch;
      const isSel = j === sel;
      ctx.fillStyle = isSel ? 'rgba(47,95,208,0.08)' : 'rgba(0,0,0,0.015)';
      ctx.fillRect(cx + 1, cy + 1, cw - 2, ch - 2);
      if (isSel) {
        ctx.strokeStyle = P.blue;
        ctx.lineWidth = 1.4;
        ctx.strokeRect(cx + 1.5, cy + 1.5, cw - 3, ch - 3);
      }

      const out = [];
      for (let i = 1; i <= 90; i++) {
        const act = model.activations([i / 90], layer);
        out.push(act[j]);
      }
      let mn = Infinity, mx = -Infinity;
      for (const v of out) { mn = Math.min(mn, v); mx = Math.max(mx, v); }
      // 用激活函数的天然值域做纵轴，避免初始阶段把噪声放大成锯齿
      const name = model.act.name;
      let lo, hi;
      if (name === 'ReLU') { lo = 0; hi = Math.max(0.5, mx); }
      else if (name === 'Sigmoid') { lo = 0; hi = 1; }
      else { lo = -1; hi = 1; }
      if (mx - mn < 1e-9) { mn = lo; mx = hi; }
      else { mn = Math.min(mn, lo); mx = Math.max(mx, hi); }

      const px = i => cx + 4 + (i - 1) / 89 * (cw - 8);
      const py = v => cy + ch - 5 - (v - mn) / (mx - mn) * (ch - 10);
      // 74 抽参考线
      ctx.strokeStyle = 'rgba(201,138,30,0.4)';
      ctx.setLineDash([2, 2]);
      ctx.beginPath(); ctx.moveTo(px(74), cy + 3); ctx.lineTo(px(74), cy + ch - 3); ctx.stroke();
      ctx.setLineDash([]);

      ctx.beginPath();
      for (let i = 1; i <= 90; i++) {
        const X = px(i), Y = py(out[i - 1]);
        if (i === 1) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
      }
      ctx.strokeStyle = isSel ? P.blue : 'rgba(28,26,23,0.55)';
      ctx.lineWidth = isSel ? 1.6 : 1;
      ctx.stroke();

      ctx.fillStyle = isSel ? P.blue : P.muted;
      ctx.font = '9px ' + MONO;
      ctx.textAlign = 'left';
      ctx.fillText('#' + j, cx + 5, cy + 9);
    }
    return { cols, rows, cw, ch };
  }

  /* ---------------- 参数条形图（一层里每个单元的 w 与 b） ---------------- */
  function paramBars(canvas, model, layer, opts) {
    opts = opts || {};
    const { ctx, w, h } = setup(canvas);
    const nIn = model.sizes[layer], nOut = model.sizes[layer + 1];
    const rows = nOut;
    const rowH = h / rows;
    let maxAbs = 1e-6;
    for (const v of model.W[layer]) maxAbs = Math.max(maxAbs, Math.abs(v));
    for (const v of model.b[layer]) maxAbs = Math.max(maxAbs, Math.abs(v));

    const midX = w * 0.42;
    const scale = (w * 0.5) / maxAbs;
    ctx.strokeStyle = 'rgba(28,26,23,0.25)';
    ctx.beginPath(); ctx.moveTo(midX, 0); ctx.lineTo(midX, h); ctx.stroke();

    for (let j = 0; j < rows; j++) {
      const cy = j * rowH + rowH / 2;
      // 权重条（向上偏置避免重叠）
      for (let i = 0; i < nIn; i++) {
        const v = model.W[layer][j * nIn + i];
        const bh = Math.max(1.6, (rowH - 3) / (nIn + 1));
        const y = cy - (nIn * bh) / 2 + i * bh;
        const len = v * scale;
        ctx.fillStyle = v >= 0 ? 'rgba(47,95,208,0.78)' : 'rgba(201,138,30,0.78)';
        ctx.fillRect(len >= 0 ? midX : midX + len, y, Math.abs(len), bh - 0.8);
      }
      // 偏置：小方块
      const bv = model.b[layer][j];
      ctx.beginPath();
      ctx.arc(midX + bv * scale, cy, Math.min(3.2, rowH * 0.28), 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(28,26,23,0.55)';
      ctx.fill();
      if (opts.label !== false && rowH > 11) {
        ctx.fillStyle = P.muted;
        ctx.font = '9px ' + MONO;
        ctx.textAlign = 'right';
        ctx.fillText('#' + j, 26, cy);
      }
    }
    return { maxAbs };
  }

  function fmtPct(v, digits) { return (v * 100).toFixed(digits == null ? 1 : digits) + '%'; }

  root.Charts = {
    P, plot, heatmap, boundary, network, responseGrid, paramBars,
    setup, classColor, fmtPct, niceTicks
  };
})(window);
