/* =============================================================
 * gacha.js —— 原神（Genshin Impact）抽卡机制的数学模型
 * 1) 单抽概率曲线 p(n)：0.6% 基础概率 + 74 抽起的软保底 + 90 抽硬保底
 * 2) 出货抽数的理论分布、期望、分位数
 * 3) 蒙特卡洛模拟：抽卡流水（可关闭保底做对照）
 * 4) 生成"抽卡日志"给 MLP 学习：特征 = 当前垫抽数，标签 = 是否出金
 * 说明：软保底具体递增率官方未公布，此处采用社区拟合最广的
 *       "74 抽起每抽 +6%，第 90 抽必出"，其综合概率约 1.6%，与官方公示一致。
 * ============================================================= */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.GachaKit = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  function makeRNG(seed) {
    let s = (seed >>> 0) || 1;
    return function () {
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      return s / 4294967296;
    };
  }

  const GACHA = {
    P5_BASE: 0.006,        // 五星基础概率
    SOFT_START: 74,        // 第 74 抽开始进入软保底
    SOFT_STEP: 0.06,       // 软保底阶段每抽概率增量
    HARD: 90,              // 硬保底
    P4_BASE: 0.051,        // 四星基础概率
    P4_HARD: 10,           // 四星十抽保底
    UP_RATE: 0.5,          // 小保底：50% 为当期 UP
    COST_PER_PULL: 160,     // 每抽原石
    CNY_PER_CRYSTAL: 648 / 8080  // 非首充：648 元 = 8080 原石
  };

  /** 第 n 抽出五星的概率（n 从 1 开始，表示距上次出金后的第 n 抽） */
  function p5(n) {
    if (n >= GACHA.HARD) return 1;
    if (n < GACHA.SOFT_START) return GACHA.P5_BASE;
    return Math.min(1, GACHA.P5_BASE + GACHA.SOFT_STEP * (n - GACHA.SOFT_START + 1));
  }

  /** 出货抽数的理论分布：P(N = n) = p(n) · Π_{k<n}(1 - p(k)) */
  function pityDist(opts) {
    opts = opts || {};
    const soft = opts.softPity !== false;
    const hard = opts.hardPity !== false ? GACHA.HARD : Infinity;
    const limit = opts.maxN || 200;
    const dist = [];
    let survive = 1;
    for (let n = 1; n <= limit; n++) {
      const p = soft ? p5(n) : GACHA.P5_BASE;
      const q = Math.min(1, n >= hard ? 1 : p);
      const prob = survive * q;
      dist.push({ n, p: q, pmf: prob });
      survive *= (1 - q);
      if (survive < 1e-7) break;
    }
    return dist;
  }

  /** 由分布导出期望、分位数、综合概率 */
  function stats(opts) {
    opts = opts || {};
    // 纯几何分布（既无软保底也无硬保底）：尾巴很长，必须用解析式，否则截断会低估期望
    if (opts.softPity === false && opts.hardPity === false) {
      const p = GACHA.P5_BASE;
      const q = t => Math.ceil(Math.log(1 - t) / Math.log(1 - p));
      return {
        mean: 1 / p, integratedRate: p,
        q25: q(0.25), q50: q(0.5), q75: q(0.75), q90: q(0.9), q99: q(0.99),
        before74: 1 - Math.pow(1 - p, 73),
        dist: pityDist(opts)
      };
    }
    const d = pityDist(opts);
    let mean = 0, cdf = 0;
    const qs = {};
    const targets = [0.25, 0.5, 0.75, 0.9, 0.99];
    let ti = 0;
    for (const row of d) {
      mean += row.n * row.pmf;
      cdf += row.pmf;
      while (ti < targets.length && cdf >= targets[ti]) {
        qs[targets[ti]] = row.n;
        ti++;
      }
    }
    const before74 = d.filter(r => r.n < GACHA.SOFT_START).reduce((a, r) => a + r.pmf, 0);
    return {
      mean, integratedRate: 1 / mean,
      q25: qs[0.25], q50: qs[0.5], q75: qs[0.75], q90: qs[0.9], q99: qs[0.99],
      before74, dist: d
    };
  }

  /**
   * 模拟一条抽卡流水
   * @returns {pulls:[{n, hit}], hits:[n...], stars5, stars4, up, lost}
   */
  function simulate(opts) {
    opts = opts || {};
    const pulls = opts.pulls || 10000;
    const rng = makeRNG(opts.seed == null ? 20260101 : opts.seed);
    const soft = opts.softPity !== false;
    const hard = opts.hardPity !== false ? GACHA.HARD : Infinity;
    const upRate = opts.upRate == null ? GACHA.UP_RATE : opts.upRate;
    const record = opts.record !== false;

    const stream = [];
    const hits = [];
    let pity = 0, stars4 = 0, count4 = 0, up = 0, lost = 0, guarantee = false;

    for (let i = 0; i < pulls; i++) {
      pity++;
      const p = soft ? p5(pity) : GACHA.P5_BASE;
      const q = pity >= hard ? 1 : p;
      const hit = rng() < q;

      // 四星：5.1% 基础 + 10 抽保底（独立计数，出五星不重置四星保底）
      count4++;
      const hit4 = rng() < (count4 >= GACHA.P4_HARD ? 1 : GACHA.P4_BASE);
      if (hit4) { stars4++; count4 = 0; }

      if (record) stream.push({ n: pity, hit: hit ? 1 : 0 });

      if (hit) {
        hits.push(pity);
        if (guarantee) { up++; guarantee = false; }
        else if (rng() < upRate) { up++; }
        else { lost++; guarantee = true; }
        pity = 0;
      }
    }
    return { pulls: stream, hits, stars5: hits.length, stars4, up, lost };
  }

  /** 跑多次"抽到 1 个 UP 五星为止"，得到 UP 出货抽数样本 */
  function simulateUntilUp(runs, seed, opts) {
    opts = opts || {};
    const rng = makeRNG(seed == null ? 7 : seed);
    const upRate = opts.upRate == null ? GACHA.UP_RATE : opts.upRate;
    const soft = opts.softPity !== false;
    const hard = opts.hardPity !== false ? GACHA.HARD : Infinity;
    const cost = [];
    for (let r = 0; r < runs; r++) {
      let total = 0, guarantee = false;
      for (;;) {
        let pity = 0;
        for (;;) {
          pity++; total++;
          const p = soft ? p5(pity) : GACHA.P5_BASE;
          const q = pity >= hard ? 1 : p;
          if (rng() < q) break;
          if (total > 2000) break; // 安全阀
        }
        if (guarantee) { guarantee = false; break; }
        if (rng() < upRate) break;
        guarantee = true;
      }
      cost.push(total);
    }
    return cost;
  }

  /** 直方图分箱 */
  function histogram(values, binSize, max) {
    const bins = new Array(Math.ceil(max / binSize)).fill(0);
    for (const v of values) {
      const i = Math.min(bins.length - 1, Math.floor((v - 1) / binSize));
      bins[i]++;
    }
    return bins;
  }

  function mean(a) { return a.reduce((x, y) => x + y, 0) / a.length; }
  function quantile(sorted, q) {
    const i = Math.min(sorted.length - 1, Math.floor(q * sorted.length));
    return sorted[i];
  }

  /** 一次前向的中间量，用于页面上的"参数如何参与计算"表格 */
  function traceForward(model, n) {
    const x = model.sizes[0] === 1 ? [n / GACHA.HARD] : [n / GACHA.HARD, 0];
    const out = new Float64Array(model.K);
    model.predict(x, out);
    return { x, layers: model._buf.map(b => Float64Array.from(b)), probs: out };
  }

  return {
    GACHA, p5, pityDist, stats, simulate, simulateUntilUp,
    histogram, mean, quantile, makeRNG, traceForward
  };
});
