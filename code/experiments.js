/* 对照实验（抽卡概率拟合任务）：为报告生成真实数字
 * 运行： node code/experiments.js
 * 指标：交叉熵损失、平均绝对误差（对经验频率，单位百分点）、由学到的曲线反推的期望抽数
 */
const G = require('../js/gacha.js');
const { MLP } = require('../js/mlp.js');

/* 分层采样：对每个垫抽数 n 独立做 trials 次伯努利试验，得到经验频率 */
function buildFreq(trials, seed) {
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

/* 用模型给出的 p(n) 反推期望抽数：E = Σ n·p(n)·Π_{k<n}(1-p(k)) */
function expectFromCurve(model) {
  const out = new Float64Array(2);
  let survive = 1, E = 0;
  for (let n = 1; n <= 90; n++) {
    model.predict([n / 90], out);
    const p = Math.min(1, Math.max(0, out[1]));
    E += n * survive * p;
    survive *= (1 - p);
  }
  return E;
}

function trainOnce(cfg, trials, seed) {
  const ds = buildFreq(trials, 900 + seed * 37);
  const sizes = [1, ...cfg.hidden, 2];
  const m = new MLP({
    sizes, activation: cfg.act, optimizer: cfg.opt || 'adam',
    lr: cfg.lr, init: cfg.init || 'xavier', seed: seed * 11 + 3
  });
  const idx = [...Array(90).keys()];
  const rng = G.makeRNG(seed * 131 + 7);
  const t0 = Date.now();
  for (let e = 0; e < cfg.epochs; e++) {
    for (let i = 89; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    for (let s = 0; s < 90; s += 32) m.trainBatch(ds.X, ds.y, idx.slice(s, s + 32));
  }
  const ms = Date.now() - t0;
  let mae = 0;
  const out = new Float64Array(2);
  for (let n = 1; n <= 90; n++) {
    m.predict([n / 90], out);
    mae += Math.abs(out[1] - ds.freq[n - 1]);
  }
  const ev = m.evaluate(ds.X, ds.y);
  return { loss: ev.loss, mae: mae / 90 * 100, expect: expectFromCurve(m), ms };
}

const SEEDS = [1, 2, 3];
function bench(name, cfg, trials) {
  const rs = SEEDS.map(s => trainOnce(cfg, trials || 6000, s));
  const avg = k => rs.reduce((a, r) => a + r[k], 0) / rs.length;
  return {
    name, loss: avg('loss'), mae: avg('mae'), expect: avg('expect'),
    ms: rs.reduce((a, r) => a + r.ms, 0) / rs.length
  };
}
function table(title, rows) {
  console.log('\n### ' + title);
  console.log('| 配置 | 交叉熵 | 平均绝对误差 | 反推期望抽数 | 训练耗时 |');
  console.log('|---|---|---|---|---|');
  for (const r of rows) {
    console.log(`| ${r.name} | ${r.loss.toFixed(4)} | ${r.mae.toFixed(2)} pp | ${r.expect.toFixed(2)} 抽 | ${r.ms.toFixed(0)} ms |`);
  }
}

const base = { hidden: [16, 16], act: 'tanh', opt: 'adam', lr: 0.05, epochs: 600 };
const mk = o => Object.assign({}, base, o);

console.log('理论值：期望 62.30 抽，综合概率 1.605%');

table('A 激活函数（2-16-16，Adam，η=0.05，600 epoch）', [
  bench('tanh', mk({ act: 'tanh' })),
  bench('ReLU', mk({ act: 'relu' })),
  bench('Sigmoid', mk({ act: 'sigmoid' }))
]);

table('B 网络结构（tanh，Adam，η=0.05，600 epoch）', [
  bench('无隐藏层 1-2（线性）', mk({ hidden: [] })),
  bench('1-8-2', mk({ hidden: [8] })),
  bench('1-16-2', mk({ hidden: [16] })),
  bench('1-16-16-2', mk({ hidden: [16, 16] })),
  bench('1-32-32-2', mk({ hidden: [32, 32] }))
]);

table('C 学习率（1-16-16-2，tanh，Adam，600 epoch）', [
  bench('η = 0.005', mk({ lr: 0.005 })),
  bench('η = 0.02', mk({ lr: 0.02 })),
  bench('η = 0.05', mk({ lr: 0.05 })),
  bench('η = 0.10', mk({ lr: 0.10 })),
  bench('η = 0.30', mk({ lr: 0.30 }))
]);

table('D 优化器（1-16-16-2，tanh，600 epoch）', [
  bench('SGD, η=0.5', mk({ opt: 'sgd', lr: 0.5 })),
  bench('SGD, η=0.05', mk({ opt: 'sgd', lr: 0.05 })),
  bench('Momentum, η=0.05', mk({ opt: 'momentum', lr: 0.05 })),
  bench('Adam, η=0.05', mk({ opt: 'adam', lr: 0.05 }))
]);

table('E 初始化方式（1-16-16-2，tanh，Adam，η=0.05，600 epoch）', [
  bench('Xavier', mk({ init: 'xavier' })),
  bench('He', mk({ init: 'he' })),
  bench('全零', mk({ init: 'zero' }))
]);

table('F 每层试验次数（1-16-16-2，tanh，Adam，η=0.05，600 epoch）', [
  bench('200 次/抽数', mk({}), 200),
  bench('1000 次/抽数', mk({}), 1000),
  bench('6000 次/抽数', mk({}), 6000)
]);
