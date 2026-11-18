/* =============================================================
 * mlp.js —— 多层感知机（MLP）学习参数的最小实现
 * 纯 JavaScript，不依赖任何框架；浏览器与 Node 均可运行。
 * 只依赖 Float64Array 完成矩阵运算，好处是每一行都能和公式对上。
 * ============================================================= */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.MLPKit = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  /* ---------------- 随机数（可复现） ---------------- */
  function makeRNG(seed) {
    let s = (seed >>> 0) || 1;
    return function () {
      // 线性同余，够用且完全可复现
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      return s / 4294967296;
    };
  }
  function gauss(rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /* ---------------- 激活函数 ---------------- */
  // 注意：df 接收的是「净输入 z」，不是激活后的值，这样可以避免回传时再用错变量
  const ACTIVATIONS = {
    tanh: {
      name: 'tanh',
      f: x => Math.tanh(x),
      df: z => { const t = Math.tanh(z); return 1 - t * t; }
    },
    relu: {
      name: 'ReLU',
      f: x => (x > 0 ? x : 0),
      df: z => (z > 0 ? 1 : 0)
    },
    sigmoid: {
      name: 'Sigmoid',
      f: x => 1 / (1 + Math.exp(-x)),
      df: z => { const s = 1 / (1 + Math.exp(-z)); return s * (1 - s); }
    }
  };

  function softmaxInPlace(z, n) {
    let m = -Infinity;
    for (let i = 0; i < n; i++) if (z[i] > m) m = z[i];
    let s = 0;
    for (let i = 0; i < n; i++) { z[i] = Math.exp(z[i] - m); s += z[i]; }
    for (let i = 0; i < n; i++) z[i] /= s;
    return z;
  }

  /* ---------------- 优化器 ---------------- */
  function createOptimizer(name, lr, model, opts) {
    opts = opts || {};
    if (name === 'sgd') return new SGD(lr);
    if (name === 'adam') return new Adam(lr, model, opts);
    return new Momentum(lr, model, opts);
  }

  class SGD {
    constructor(lr) { this.lr = lr; this.name = 'sgd'; }
    step(m) {
      const lr = this.lr;
      for (let l = 0; l < m.L; l++) {
        const W = m.W[l], dW = m.dW[l], b = m.b[l], db = m.db[l];
        for (let k = 0; k < W.length; k++) W[k] -= lr * dW[k];
        for (let k = 0; k < b.length; k++) b[k] -= lr * db[k];
      }
    }
  }

  class Momentum {
    constructor(lr, m, opts) {
      this.lr = lr; this.name = 'momentum';
      this.beta = (opts && opts.beta != null) ? opts.beta : 0.9;
      this.vW = m.W.map(a => new Float64Array(a.length));
      this.vb = m.b.map(a => new Float64Array(a.length));
    }
    step(m) {
      const lr = this.lr, be = this.beta;
      for (let l = 0; l < m.L; l++) {
        const W = m.W[l], dW = m.dW[l], vW = this.vW[l];
        const b = m.b[l], db = m.db[l], vb = this.vb[l];
        for (let k = 0; k < W.length; k++) { vW[k] = be * vW[k] + dW[k]; W[k] -= lr * vW[k]; }
        for (let k = 0; k < b.length; k++) { vb[k] = be * vb[k] + db[k]; b[k] -= lr * vb[k]; }
      }
    }
  }

  class Adam {
    constructor(lr, m, opts) {
      this.lr = lr; this.name = 'adam';
      opts = opts || {};
      this.b1 = opts.beta1 != null ? opts.beta1 : 0.9;
      this.b2 = opts.beta2 != null ? opts.beta2 : 0.999;
      this.eps = 1e-8; this.t = 0;
      this.mW = m.W.map(a => new Float64Array(a.length));
      this.vW = m.W.map(a => new Float64Array(a.length));
      this.mb = m.b.map(a => new Float64Array(a.length));
      this.vb = m.b.map(a => new Float64Array(a.length));
    }
    step(m) {
      this.t++;
      const lr = this.lr * Math.sqrt(1 - Math.pow(this.b2, this.t)) / (1 - Math.pow(this.b1, this.t));
      for (let l = 0; l < m.L; l++) {
        const W = m.W[l], dW = m.dW[l], mW = this.mW[l], vW = this.vW[l];
        const b = m.b[l], db = m.db[l], mb = this.mb[l], vb = this.vb[l];
        for (let k = 0; k < W.length; k++) {
          const g = dW[k];
          mW[k] = this.b1 * mW[k] + (1 - this.b1) * g;
          vW[k] = this.b2 * vW[k] + (1 - this.b2) * g * g;
          W[k] -= lr * mW[k] / (Math.sqrt(vW[k]) + this.eps);
        }
        for (let k = 0; k < b.length; k++) {
          const g = db[k];
          mb[k] = this.b1 * mb[k] + (1 - this.b1) * g;
          vb[k] = this.b2 * vb[k] + (1 - this.b2) * g * g;
          b[k] -= lr * mb[k] / (Math.sqrt(vb[k]) + this.eps);
        }
      }
    }
  }

  /* ---------------- MLP ---------------- */
  class MLP {
    /**
     * @param {Object} cfg
     *   sizes      [2, 8, 8, K]  输入维 · 各隐藏层宽度 · 输出类别数
     *   activation 'tanh' | 'relu' | 'sigmoid'
     *   init       'xavier' | 'he' | 'zero'
     *   optimizer  'sgd' | 'momentum' | 'adam'
     *   lr         学习率
     *   l2         L2 正则系数
     *   seed       随机种子
     */
    constructor(cfg) {
      cfg = cfg || {};
      this.sizes = (cfg.sizes || [2, 8, 8, 2]).slice();
      this.act = ACTIVATIONS[cfg.activation || 'tanh'];
      this.L = this.sizes.length - 1;             // 权重层数
      this.K = this.sizes[this.L];                // 输出维度
      this.rng = makeRNG(cfg.seed == null ? 20261115 : cfg.seed);
      this.initMode = cfg.init || 'xavier';
      this.l2 = cfg.l2 || 0;
      this._initParams();
      this.opt = createOptimizer(cfg.optimizer || 'momentum', cfg.lr == null ? 0.1 : cfg.lr, this, cfg);
      this.gradNorm = 0;
      this.step = 0;
      // 前向推理的复用缓冲区，避免在热力图上疯狂分配内存
      this._buf = this.sizes.map(n => new Float64Array(n));
    }

    get nParams() {
      let s = 0;
      for (let l = 0; l < this.L; l++) s += this.sizes[l] * this.sizes[l + 1] + this.sizes[l + 1];
      return s;
    }

    _initParams() {
      this.W = []; this.b = []; this.dW = []; this.db = [];
      for (let l = 0; l < this.L; l++) {
        const nIn = this.sizes[l], nOut = this.sizes[l + 1];
        const W = new Float64Array(nIn * nOut), b = new Float64Array(nOut);
        if (this.initMode !== 'zero') {
          let std;
          if (this.initMode === 'he') std = Math.sqrt(2 / nIn);
          else if (this.initMode === 'xavier') std = Math.sqrt(2 / (nIn + nOut));
          else std = 0.3;
          for (let i = 0; i < W.length; i++) W[i] = gauss(this.rng) * std;
        }
        this.W.push(W); this.b.push(b);
        this.dW.push(new Float64Array(nIn * nOut)); this.db.push(new Float64Array(nOut));
      }
    }

    setOptimizer(name, lr, opts) {
      this.opt = createOptimizer(name, lr, this, opts);
      // 换优化器时 Adam 的动量统计清零，保证行为符合预期
      return this.opt;
    }

    setLR(lr) { this.opt.lr = lr; }

    /* --- 单样本前向：把每层的激活值写进复用缓冲区 --- */    _forward(x) {
      const buf = this._buf;
      const L = this.L;
      const A0 = buf[0];
      for (let i = 0; i < this.sizes[0]; i++) A0[i] = x[i];
      for (let l = 0; l < L; l++) {
        const nIn = this.sizes[l], nOut = this.sizes[l + 1];
        const W = this.W[l], b = this.b[l], prev = buf[l], cur = buf[l + 1];
        for (let j = 0; j < nOut; j++) {
          let s = b[j]; const off = j * nIn;
          for (let q = 0; q < nIn; q++) s += W[off + q] * prev[q];
          cur[j] = s;
        }
        if (l < L - 1) {
          const f = this.act.f;
          for (let j = 0; j < nOut; j++) cur[j] = f(cur[j]);
        } else {
          softmaxInPlace(cur, nOut);
        }
      }
      return buf[L];
    }

    /** 预测类别概率，结果写入 out（长度 K）并返回 */
    predict(x, out) {
      const p = this._forward(x);
      if (out) { for (let k = 0; k < this.K; k++) out[k] = p[k]; return out; }
      return Float64Array.from(p);
    }

    /** 取第 layer 层（0 基，权重层编号）之后那层的激活值副本 */
    activations(x, layer) {
      this._forward(x);
      return Float64Array.from(this._buf[layer + 1]);
    }

    /** 完整记录一次前向的净输入与激活，供页面展示计算过程 */
    trace(x) {
      const L = this.L;
      const A = [Float64Array.from(x)];
      const Z = [null];
      let prev = x;
      for (let l = 0; l < L; l++) {
        const nIn = this.sizes[l], nOut = this.sizes[l + 1];
        const W = this.W[l], b = this.b[l];
        const z = new Float64Array(nOut);
        for (let j = 0; j < nOut; j++) {
          let s = b[j]; const off = j * nIn;
          for (let q = 0; q < nIn; q++) s += W[off + q] * prev[q];
          z[j] = s;
        }
        Z.push(z);
        let a;
        if (l < L - 1) {
          a = new Float64Array(nOut);
          const f = this.act.f;
          for (let j = 0; j < nOut; j++) a[j] = f(z[j]);
        } else {
          a = softmaxInPlace(Float64Array.from(z), nOut);
        }
        A.push(a);
        prev = a;
      }
      return { A, Z };
    }

    /**
     * 单样本「前向 + 反向」，不修改参数。
     * 返回逐层激活 A、净输入 Z、误差项 delta、梯度 dW/dB，
     * 供页面把一次迭代的每个中间量都摊开画出来。
     */
    gradSample(x, t) {
      const L = this.L, K = this.K;
      const tr = this.trace(x);
      const A = tr.A, Z = tr.Z;
      const p = A[L];

      let loss = 0;
      const dOut = new Float64Array(K);
      for (let k = 0; k < K; k++) {
        const tgt = typeof t === 'number' ? (k === t ? 1 : 0) : t[k];
        loss += -tgt * Math.log(Math.max(p[k], 1e-12));
        dOut[k] = p[k] - tgt;
      }

      const delta = new Array(L + 1);
      delta[L] = dOut;
      for (let l = L - 1; l >= 1; l--) {
        const nIn = this.sizes[l], nOut = this.sizes[l + 1];
        const Wl = this.W[l], df = this.act.df, zl = Z[l];
        const prev = new Float64Array(nIn);
        for (let q = 0; q < nIn; q++) {
          let s = 0;
          for (let j = 0; j < nOut; j++) s += Wl[j * nIn + q] * delta[l + 1][j];
          prev[q] = s * df(zl[q]);
        }
        delta[l] = prev;
      }

      const dW = [], dB = [], deltaNorm = [0];
      for (let l = 1; l <= L; l++) {
        const d = delta[l];
        let s = 0;
        for (let k = 0; k < d.length; k++) s += d[k] * d[k];
        deltaNorm.push(Math.sqrt(s));
      }
      for (let l = 0; l < L; l++) {
        const nIn = this.sizes[l], nOut = this.sizes[l + 1];
        const gW = new Float64Array(nOut * nIn), gB = new Float64Array(nOut);
        for (let j = 0; j < nOut; j++) {
          const dj = delta[l + 1][j];
          gB[j] = dj;
          for (let q = 0; q < nIn; q++) gW[j * nIn + q] = dj * A[l][q];
        }
        dW.push(gW); dB.push(gB);
      }
      return { x, A, Z, p, loss, delta, dW, dB, deltaNorm };
    }

    /** 给定数据集的平均激活值，用于网络图着色 */
    meanActivations(X) {
      const L = this.L;
      const means = this.sizes.map(n => new Float64Array(n));
      const cnt = X.length;
      for (const x of X) {
        this._forward(x);
        for (let l = 0; l <= L; l++) {
          const b = this._buf[l], m = means[l];
          for (let i = 0; i < b.length; i++) m[i] += b[i] / cnt;
        }
      }
      return means;
    }

    /** 一个 mini-batch 的训练：前向 → 反向 → 更新参数，返回平均损失与准确率 */
    trainBatch(X, y, idx) {
      const L = this.L, K = this.K;
      for (let l = 0; l < L; l++) { this.dW[l].fill(0); this.db[l].fill(0); }

      const n = idx.length;
      let loss = 0, correct = 0;
      const A = new Array(L + 1);       // 各层激活
      const Zpre = new Array(L + 1);    // 各层净输入（反向传播需要）

      for (let s = 0; s < n; s++) {
        const i = idx[s];
        const xi = X[i], ti = y[i];

        /* ---------- 前向 ---------- */
        A[0] = xi;
        for (let l = 0; l < L; l++) {
          const nIn = this.sizes[l], nOut = this.sizes[l + 1];
          const W = this.W[l], b = this.b[l];
          const z = new Float64Array(nOut);
          for (let j = 0; j < nOut; j++) {
            let sum = b[j]; const off = j * nIn;
            for (let q = 0; q < nIn; q++) sum += W[off + q] * A[l][q];
            z[j] = sum;
          }
          Zpre[l + 1] = z;
          if (l < L - 1) {
            const a = new Float64Array(nOut), f = this.act.f;
            for (let j = 0; j < nOut; j++) a[j] = f(z[j]);
            A[l + 1] = a;
          } else {
            A[l + 1] = softmaxInPlace(z, nOut);
          }
        }

        const p = A[L];
        // 支持两种标签：整数（one-hot）与向量（软标签，比如某抽数上的经验出货频率）
        const soft = typeof ti !== 'number';
        let lossI = 0, ai = 0, best = -Infinity;
        for (let k = 0; k < K; k++) {
          const tk = soft ? ti[k] : (k === ti ? 1 : 0);
          lossI += -tk * Math.log(Math.max(p[k], 1e-12));
          if (p[k] > best) { best = p[k]; ai = k; }
        }
        loss += lossI;
        if (soft) {
          let bm = -Infinity, tiBest = 0;
          for (let k = 0; k < K; k++) if (ti[k] > bm) { bm = ti[k]; tiBest = k; }
          if (ai === tiBest) correct++;
        } else if (ai === ti) correct++;

        /* ---------- 反向 ---------- */
        // δ^(L) = p - t（t 既可以是 one-hot，也可以是软标签）
        let delta = new Float64Array(K);
        for (let k = 0; k < K; k++) delta[k] = p[k] - (soft ? ti[k] : (k === ti ? 1 : 0));

        for (let l = L - 1; l >= 0; l--) {
          const nIn = this.sizes[l], nOut = this.sizes[l + 1];
          const dWl = this.dW[l], dbl = this.db[l], aPrev = A[l];
          for (let j = 0; j < nOut; j++) {
            const dj = delta[j];
            dbl[j] += dj;
            const off = j * nIn;
            for (let q = 0; q < nIn; q++) dWl[off + q] += dj * aPrev[q];
          }
          if (l > 0) {
            const Wl = this.W[l], df = this.act.df;
            const prev = new Float64Array(nIn);
            for (let q = 0; q < nIn; q++) {
              let s2 = 0;
              for (let j = 0; j < nOut; j++) s2 += Wl[j * nIn + q] * delta[j];
              prev[q] = s2 * df(Zpre[l][q]);
            }
            delta = prev;
          }
        }
      }

      /* ---------- 平均梯度 + L2 + 参数更新 ---------- */
      const inv = 1 / n;
      let g2 = 0;
      for (let l = 0; l < L; l++) {
        const dW = this.dW[l], W = this.W[l], db = this.db[l];
        for (let k = 0; k < dW.length; k++) {
          let g = dW[k] * inv;
          if (this.l2) g += this.l2 * W[k];
          dW[k] = g; g2 += g * g;
        }
        for (let k = 0; k < db.length; k++) {
          const g = db[k] * inv;
          db[k] = g; g2 += g * g;
        }
      }
      this.gradNorm = Math.sqrt(g2);
      this.opt.step(this);
      this.step++;

      return { loss: loss / n, acc: correct / n };
    }

    /** 整个数据集上的损失与准确率（不做参数更新）；标签可为软标签 */
    evaluate(X, y) {
      const out = new Float64Array(this.K);
      let loss = 0, correct = 0;
      for (let i = 0; i < X.length; i++) {
        this.predict(X[i], out);
        const ti = y[i];
        if (typeof ti === 'number') {
          loss += -Math.log(Math.max(out[ti], 1e-12));
        } else {
          for (let k = 0; k < this.K; k++) loss += -ti[k] * Math.log(Math.max(out[k], 1e-12));
        }
        let am = -1, ai = 0;
        for (let k = 0; k < this.K; k++) if (out[k] > am) { am = out[k]; ai = k; }
        if (typeof ti === 'number') { if (ai === ti) correct++; }
        else {
          let bm = -Infinity, bi = 0;
          for (let k = 0; k < this.K; k++) if (ti[k] > bm) { bm = ti[k]; bi = k; }
          if (ai === bi) correct++;
        }
      }
      return { loss: loss / X.length, acc: correct / X.length };
    }
  }

  /* ---------------- 玩具数据集 ---------------- */
  const DATASETS = {
    xor: {
      name: '异或 XOR', K: 2, ext: 1.7,
      gen(n, noise, rng) {
        const X = [], y = [];
        const corners = [[-1, -1], [-1, 1], [1, -1], [1, 1]];
        for (let i = 0; i < n; i++) {
          const c = corners[i % 4];
          const cx = c[0] * 0.65, cy = c[1] * 0.65;
          const x = cx + gauss(rng) * (0.18 + noise * 0.5);
          const z = cy + gauss(rng) * (0.18 + noise * 0.5);
          X.push([x, z]);
          y.push((c[0] * c[1]) > 0 ? 0 : 1);
        }
        return { X, y };
      }
    },

    moons: {
      name: '双月 Moons', K: 2, ext: 1.6,
      gen(n, noise, rng) {
        const X = [], y = [];
        const half = Math.floor(n / 2);
        for (let i = 0; i < half; i++) {
          const t = Math.PI * (i / (half - 1));
          X.push([Math.cos(t) * 0.9 - 0.2 + gauss(rng) * (0.06 + noise * 0.35),
                  Math.sin(t) * 0.9 - 0.25 + gauss(rng) * (0.06 + noise * 0.35)]);
          y.push(0);
        }
        for (let i = 0; i < n - half; i++) {
          const t = Math.PI * (i / (n - half - 1));
          X.push([Math.cos(t) * 0.9 + 0.2 + gauss(rng) * (0.06 + noise * 0.35),
                  0.35 - Math.sin(t) * 0.9 + gauss(rng) * (0.06 + noise * 0.35)]);
          y.push(1);
        }
        return { X, y };
      }
    },

    circles: {
      name: '同心圆 Circles', K: 2, ext: 1.5,
      gen(n, noise, rng) {
        const X = [], y = [];
        for (let i = 0; i < n; i++) {
          const inner = i % 2 === 0;
          const r = inner ? rng() * 0.42 : 0.82 + rng() * 0.18;
          const th = rng() * Math.PI * 2;
          const nr = r + gauss(rng) * (0.03 + noise * 0.18);
          X.push([Math.cos(th) * nr, Math.sin(th) * nr]);
          y.push(inner ? 0 : 1);
        }
        return { X, y };
      }
    },

    spiral: {
      name: '螺旋 Spiral（3 类）', K: 3, ext: 1.5,
      gen(n, noise, rng) {
        const X = [], y = [];
        const K = 3;
        for (let i = 0; i < n; i++) {
          const c = i % K;
          const t = (Math.floor(i / K) / Math.max(1, Math.floor(n / K) - 1));
          const r = 0.12 + t * 1.05;
          const th = t * 2.9 * Math.PI + c * (2 * Math.PI / K);
          X.push([Math.cos(th) * r + gauss(rng) * (0.02 + noise * 0.1),
                  Math.sin(th) * r + gauss(rng) * (0.02 + noise * 0.1)]);
          y.push(c);
        }
        return { X, y };
      }
    },

    gauss: {
      name: '高斯团（线性可分）', K: 2, ext: 1.6,
      gen(n, noise, rng) {
        const X = [], y = [];
        for (let i = 0; i < n; i++) {
          const c = i % 2;
          X.push([(c === 0 ? -0.75 : 0.75) + gauss(rng) * (0.35 + noise * 0.4),
                  gauss(rng) * (0.35 + noise * 0.4)]);
          y.push(c);
        }
        return { X, y };
      }
    }
  };

  function makeDataset(key, n, noise, seed) {
    const d = DATASETS[key] || DATASETS.xor;
    const rng = makeRNG(seed == null ? 7 : seed);
    const { X, y } = d.gen(n, noise == null ? 0.1 : noise, rng);
    return { X, y, K: d.K, name: d.name, ext: d.ext };
  }

  return { MLP, ACTIVATIONS, DATASETS, makeDataset, makeRNG, gauss };
});
