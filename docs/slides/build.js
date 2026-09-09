// 8/26 汇报 slides — 极简学术风：白底、全黑文字、Segoe UI、无装饰
const pptxgen = require("pptxgenjs");

const F = "Segoe UI";
const K = "000000";           // 全部文字统一黑色
const RULE = "BFBFBF";        // 表格细线
const HEAD = "F2F2F2";        // 表头浅灰底
const BAR1 = "1A1A1A";        // 图表主色（近黑）
const BAR2 = "B3B3B3";        // 图表次色（灰）

const W = 13.333, H = 7.5;
const ML = 0.62, CW = W - 2 * ML;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "RAP / DriveWeave";
pres.title = "RAP Alignment 受控实验与 DriveWeave 提案";

let page = 0;
function slide(opts = {}) {
  const s = pres.addSlide();
  s.background = { color: "FFFFFF" };
  page += 1;
  if (!opts.noNum) {
    s.addText(String(page), {
      x: W - ML - 1.0, y: H - 0.62, w: 1.0, h: 0.25,
      fontFace: F, fontSize: 9, color: K, align: "right", margin: 0,
    });
  }
  return s;
}

function title(s, text, sub) {
  s.addText(text, {
    x: ML, y: 0.44, w: CW, h: 0.46,
    fontFace: F, fontSize: 23, bold: true, color: K, margin: 0, valign: "top",
  });
  if (sub) {
    s.addText(sub, {
      x: ML, y: 0.92, w: CW, h: 0.26,
      fontFace: F, fontSize: 11, color: K, margin: 0,
    });
  }
}

// 结论先行块
function lead(s, text, y = 1.14, h = 0.72) {
  s.addText(text, {
    x: ML, y, w: CW, h,
    fontFace: F, fontSize: 14.5, bold: true, color: K, margin: 0,
    lineSpacingMultiple: 1.15, valign: "top",
  });
}

function body(s, items, o = {}) {
  const runs = items.map((t, i) => ({
    text: typeof t === "string" ? t : t.text,
    options: Object.assign(
      { bullet: typeof t === "string" ? true : (t.bullet !== false),
        breakLine: i !== items.length - 1,
        bold: typeof t === "object" && !!t.bold,
        indentLevel: (typeof t === "object" && t.lvl) || 0 },
      {}
    ),
  }));
  s.addText(runs, Object.assign({
    x: ML, y: 2.0, w: CW, h: 4.4,
    fontFace: F, fontSize: 13, color: K, margin: 0,
    paraSpaceAfter: 7, lineSpacingMultiple: 1.12, valign: "top",
  }, o));
}

function para(s, text, o = {}) {
  s.addText(text, Object.assign({
    x: ML, y: 2.0, w: CW, h: 1.0,
    fontFace: F, fontSize: 13, color: K, margin: 0,
    lineSpacingMultiple: 1.2, valign: "top",
  }, o));
}

function table(s, rows, o = {}) {
  const border = [
    { type: "solid", color: RULE, pt: 0.5 },
    { type: "solid", color: RULE, pt: 0.5 },
    { type: "solid", color: RULE, pt: 0.5 },
    { type: "solid", color: RULE, pt: 0.5 },
  ];
  const built = rows.map((r, ri) =>
    r.map((c) => {
      const cell = typeof c === "string" ? { text: c } : c;
      return {
        text: cell.text,
        options: Object.assign(
          {
            fontFace: F, fontSize: o.fontSize || 11.5, color: K,
            bold: ri === 0 ? true : !!cell.bold,
            align: cell.align || (ri === 0 ? "left" : "left"),
            fill: ri === 0 ? HEAD : (cell.fill || "FFFFFF"),
            valign: "middle", margin: [3, 6, 3, 6],
          },
          {}
        ),
      };
    })
  );
  s.addTable(built, Object.assign({
    x: ML, y: 2.1, w: CW, border,
    autoPage: false, rowH: 0.32,
  }, o.tbl || {}));
}

function chartBase(extra = {}) {
  return Object.assign({
    showLegend: false,
    showTitle: false,
    chartColors: [BAR1, BAR2],
    catAxisLabelFontFace: F, catAxisLabelFontSize: 11, catAxisLabelColor: K,
    valAxisLabelFontFace: F, valAxisLabelFontSize: 10, valAxisLabelColor: K,
    dataLabelFontFace: F, dataLabelFontSize: 10, dataLabelColor: K,
    catAxisLineShow: true, valAxisLineShow: false,
    valGridLine: { color: "E6E6E6", size: 0.5 },
    catGridLine: { style: "none" },
    border: { pt: 0, color: "FFFFFF" },
  }, extra);
}

function note(s, txt) { s.addNotes(txt); }

// 小字脚注
function foot(s, txt, y = H - 1.0, o = {}) {
  s.addText(txt, Object.assign({
    x: ML, y, w: CW, h: 0.44,
    fontFace: F, fontSize: 10, color: K, margin: 0,
    lineSpacingMultiple: 1.15, valign: "top",
  }, o));
}

function divider(num, t, sub) {
  const s = slide();
  s.addText(num, { x: ML, y: 2.5, w: 3.0, h: 1.2, fontFace: F, fontSize: 54, bold: true, color: K, margin: 0 });
  s.addText(t, { x: ML, y: 3.75, w: 9.0, h: 0.6, fontFace: F, fontSize: 26, bold: true, color: K, margin: 0 });
  if (sub) s.addText(sub, { x: ML, y: 4.42, w: 9.5, h: 0.5, fontFace: F, fontSize: 13, color: K, margin: 0 });
  return s;
}

/* ============================ 1. 封面 ============================ */
{
  const s = slide({ noNum: true });
  s.addText("Alignment 受控实验 与 DriveWeave 提案", {
    x: ML, y: 2.25, w: 11.6, h: 0.8, fontFace: F, fontSize: 34, bold: true, color: K, margin: 0,
  });
  s.addText("Source-Sensor-Free Cross-Dataset Recovery Scaling for End-to-End Driving", {
    x: ML, y: 3.12, w: 11.6, h: 0.45, fontFace: F, fontSize: 15, color: K, margin: 0,
  });
  s.addText([
    { text: "本轮没有关于 alignment 的方向性结论 —— 这句话在看到分数之前就写进了协议。", options: { breakLine: true, bold: true } },
    { text: "交付的是：一台受控测量装置、一把噪声尺子、一个保守 planner 的失败画像。", options: {} },
  ], { x: ML, y: 4.15, w: 11.6, h: 0.8, fontFace: F, fontSize: 13, color: K, margin: 0, lineSpacingMultiple: 1.2 });
  s.addText("2026 年 8 月 26 日", { x: ML, y: 6.3, w: 6.0, h: 0.3, fontFace: F, fontSize: 12, color: K, margin: 0 });
  note(s, "开场 15 秒：先声明状态，再讲内容。全程 30 分钟，Part 1 约 12 分钟，Part 2 约 14 分钟。");
}

/* ============================ 2. 总纲 ============================ */
{
  const s = slide();
  title(s, "总纲：这场汇报是一件事，不是两件");
  lead(s, "RAP 已经证明「用目标数据集自己的 structured logs 造 recovery 监督」有效。它没回答的是：这件事能不能跨数据集。");

  const cols = [
    ["已 证 明", "RAP（同源）", "用目标数据集自己的\nstructured logs 造 recovery 监督\n是有效的。\n\nv2 EPDMS 32.5 → 36.9"],
    ["本 轮", "校准与定位", "把传输机制和测量精度\n校准一遍，并定位\nplanner 稳定失败的形状。\n\n交付装置，不交付结论"],
    ["下 一 步", "DriveWeave", "recovery 监督能否\n跨 sensor-incompatible\n的真实数据集扩展？\n\nsource 端一张图都不读"],
  ];
  cols.forEach((c, i) => {
    const x = ML + i * (CW / 3);
    const w = CW / 3 - 0.45;
    s.addText(c[0], { x, y: 2.25, w, h: 0.3, fontFace: F, fontSize: 11, color: K, margin: 0, charSpacing: 1 });
    s.addText(c[1], { x, y: 2.6, w, h: 0.42, fontFace: F, fontSize: 19, bold: true, color: K, margin: 0 });
    s.addText(c[2], { x, y: 3.15, w, h: 2.1, fontFace: F, fontSize: 12.5, color: K, margin: 0, lineSpacingMultiple: 1.25, valign: "top" });
  });

  foot(s, "比喻——桥与货：RAP 的闭环增益来自 raster-only recovery 增强样本（货）；alignment 只是让这些没有真图的样本能喂给真实 planner 的机制（桥）。\n本轮的受控实验刻意卸掉了货，只称桥本身的重量。", 5.55, { h: 0.7 });
  note(s, "1 分钟。三个框从左到右念一遍，落到最后一句「桥与货」。这个比喻后面 Q&A 会反复用到。");
}

/* ============================ Part 1 divider ============================ */
divider("Part 1", "Alignment 实验", "受控测量装置 · 噪声尺度 · 铁核画像");

/* ---------- 1.1 状态 ---------- */
{
  const s = slide();
  title(s, "首页必须声明：本轮交付的不是结论，是装置");
  lead(s, "status = incomplete　|　designation = pipeline_rehearsal_v1_only");

  table(s, [
    ["", "本轮实际", "官方 Stage-A 要求"],
    ["候选池", "21%（本地可用性）", "100%"],
    ["优化步数", "400 optimizer steps", "≈ 100 × 本轮"],
    ["评测指标族", "仅 v1 PDMS；v2.2 evaluator 未接入", "co-primary family 完整"],
    ["EPDMS", "not_tested", "official final metric"],
  ], { tbl: { y: 2.05, colW: [2.3, 5.2, 4.593], rowH: 0.36 } });

  s.addText("协议 forbidden_use 明令禁止：", { x: ML, y: 4.35, w: CW, h: 0.3, fontFace: F, fontSize: 13, bold: true, color: K, margin: 0 });
  body(s, [
    "对 alignment 是否改善 planning performance 作任何方向性结论",
    "把本轮数字当作 Stage-B 的效应量先导",
    "对外称为官方 Stage-A 结果",
  ], { y: 4.7, h: 1.2 });
  foot(s, "PDMS 的 p 值一律标注 descriptive_non_confirmatory。", 6.05);
  note(s, "1 分钟。原话：「这一轮没有关于 alignment 的科学结论——这句话在我看到分数之前就写进协议了。我交付的是三样东西：一台受控测量装置、一把噪声尺子、一个保守 planner 的活标本。」");
}

/* ---------- 1.2 评测器 ---------- */
{
  const s = slide();
  title(s, "先证明尺子是准的");
  lead(s, "constant-velocity 打出 20.6517，论文 Table 1 是 20.6 —— 差 0.05 个百分点。");

  para(s, "这一个数字同时验证了 metric cache、地图版本、数据集版本、LQR 仿真、PDMScorer 五个环节。", { y: 1.95, h: 0.4 });

  table(s, [
    ["检查项", "结果"],
    ["G1–G7 关系式门禁", "全部通过"],
    ["RAP fork vs official v1.1（0811876c…）", "唯一实质差异 = batch_lqr_utils.py 伪逆"],
    ["伪逆差异波及范围", "12,146 个场景中 9 个非零"],
    ["最大 |ΔPDMS|", "3.35e-12"],
    ["constant-velocity 复现", "论文 20.6 → 本次 20.6517"],
  ], { tbl: { y: 2.6, colW: [5.0, 7.093], rowH: 0.38 } });

  foot(s, "产物：outputs/alignment_stage_a/evaluator_validation/{scorer_equivalence_record.json, v1_paper_reference.json}", 5.35);
  note(s, "1.5 分钟。这是本轮最硬的正面成果，被问「你们的 evaluator 是自己改的，凭什么信」时直接翻回本页。");
}

/* ---------- 1.3 配置 ---------- */
{
  const s = slide();
  title(s, "受控配置：6 个 run，两个条件，三个种子");
  lead(s, "收敛门禁两条判据全部 FAILED —— 后面所有数字都带 under-trained regime 标注。");

  s.addText("训练配置", { x: ML, y: 2.0, w: 5.6, h: 0.3, fontFace: F, fontSize: 13.5, bold: true, color: K, margin: 0 });
  body(s, [
    "2,498 train / 464 val",
    "20 epochs　|　400 optimizer steps",
    "global batch 128（4 devices × 32）",
    "3 seeds × 2 conditions = 6 runs",
    "AdamW lr 1e-4，cosine + 1 warmup epoch",
    "dropout 0，16-mixed，final-step checkpoint",
  ], { x: ML, y: 2.38, w: 5.6, h: 2.6, fontSize: 12.5 });

  s.addText("收敛门禁", { x: ML + 6.4, y: 2.0, w: 5.7, h: 0.3, fontFace: F, fontSize: 13.5, bold: true, color: K, margin: 0 });
  body(s, [
    "6 个 run 的 argmin(val_loss) 全部落在 epoch 19",
    "末段斜率 [-0.048 … -0.065]，超阈值 2–3 倍",
    { text: "判定：FAILED（两条判据）", bold: true },
  ], { x: ML + 6.4, y: 2.38, w: 5.7, h: 1.8, fontSize: 12.5 });

  foot(s, "必须主动说的第一句：本轮模型处在 under-trained regime（75.8 vs RAP 论文 93.8）。后面「铁核」里有一部分会在全量训练后消失。", 5.5);
  note(s, "1 分钟。不要跳过 gate FAILED，这是诚实性的第一个锚点。");
}

/* ---------- 1.4 主结果表 ---------- */
{
  const s = slide();
  title(s, "主结果：效应不可与 seed 噪声区分");
  lead(s, "paired delta = −0.000184，CI [−0.00321, +0.00280] 跨 0，p = 0.9096，Cohen's dz = −0.0015。");

  table(s, [
    ["条件", "seed 0", "seed 1", "seed 2", "mean", "sd"],
    ["NoAlign PDMS", "76.02", "74.11", "77.32", { text: "75.82", bold: true }, "1.61"],
    ["FullAlign PDMS", "76.90", "76.35", "74.15", { text: "75.80", bold: true }, "1.46"],
    ["per-seed delta", "+0.88", "+2.24", "−3.18", { text: "−0.02", bold: true }, "—"],
  ], { tbl: { y: 2.05, colW: [3.093, 1.8, 1.8, 1.8, 1.8, 1.8], rowH: 0.4 } });

  s.addText("逐场景转移（12,146 个场景）", { x: ML, y: 3.95, w: 5.6, h: 0.3, fontFace: F, fontSize: 13.5, bold: true, color: K, margin: 0 });
  body(s, [
    "13.6% 完全未变",
    "39.6% 变好　|　46.8% 变差",
    "净和 −2.24 分",
  ], { x: ML, y: 4.32, w: 5.6, h: 1.3, fontSize: 12.5 });

  s.addText("符号稳定性", { x: ML + 6.4, y: 3.95, w: 5.7, h: 0.3, fontFace: F, fontSize: 13.5, bold: true, color: K, margin: 0 });
  body(s, [
    "per-seed delta 三个种子符号翻转",
    "5 个离散 sub-score 全部在 seed 间翻方向",
  ], { x: ML + 6.4, y: 4.32, w: 5.7, h: 1.3, fontSize: 12.5 });

  foot(s, "不能上台的观察：FullAlign 在 7 个指标里 6 个 seed 方差更小。n = 3、未预注册 —— 只能作为 Stage-B / A6 的预注册假设提出。", 6.0);
  note(s, "2 分钟。原话：「协议禁止我从这里得出方向性结论，我也不打算得。」");
}

/* ---------- 1.4b 174× ---------- */
{
  const s = slide();
  title(s, "把效应放在噪声旁边看");
  lead(s, "换掉 alignment：−0.02 分。换一个 seed：最多 3.21 分。相差 174 倍。");

  s.addChart(pres.ChartType.bar, [{
    name: "PDMS 变动幅度（分）",
    labels: ["换掉 alignment\n(FullAlign − NoAlign)", "换一个 seed\n(NoAlign s1 → s2)"],
    values: [0.02, 3.21],
  }], chartBase({
    x: ML, y: 1.95, w: 6.9, h: 4.0,
    barDir: "col", barGapWidthPct: 160,
    chartColors: [BAR1],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00",
    valAxisMinVal: 0, valAxisMaxVal: 3.5,
    catAxisLabelFontSize: 11,
  }));

  s.addText("参照系", { x: ML + 7.4, y: 2.05, w: 4.7, h: 0.3, fontFace: F, fontSize: 13.5, bold: true, color: K, margin: 0 });
  table(s, [
    ["", "PDMS"],
    ["constant-velocity", "20.65"],
    ["本轮（under-trained）", { text: "75.8", bold: true }],
    ["RAP 论文", "≈ 93.8"],
    ["human", "94.8"],
  ], { tbl: { x: ML + 7.4, y: 2.45, w: 4.7, colW: [2.9, 1.8], rowH: 0.36 }, fontSize: 11.5 });

  foot(s, "接缝 A：−0.02 留下一个我可以回答的问题 —— 它到底是因为 alignment 没用、因为我测不出来、还是因为那个位置本来就没多少东西可动？", 5.9);
  note(s, "1 分钟。本页只讲一个数：174 倍。讲完立刻抛出接缝 A 的三选一问题，引到下一段。");
}

/* ---------- 1.4d 机制：它在运行，但够不到 DINO ---------- */
{
  const s = slide();
  title(s, "效应为什么这么小：它在运行，接线正确，但它够不到 DINO");
  lead(s, "「没效果是因为有 bug」已被排除。真正的限制是结构性的 —— 而这一条来自代码，不来自分数，不受协议约束。");

  s.addText("先排除三件事", { x: ML, y: 2.05, w: 5.6, h: 0.3, fontFace: F, fontSize: 13.5, bold: true, color: K, margin: 0 });
  body(s, [
    "三 seed 逐行验证全部 passed：NoAlign 权重逐行为零、FullAlign 逐行 0.002 / 0.1、GRL 逐行合公式、total loss 可逐行重构",
    "它改变了 86.4% 场景的输出（39.6% 变好 / 46.8% 变差 / 13.6% 未变）→ 不是数值惰性项",
    "逐场景 |Δ| q95 = 0.26，换 seed 是 0.74 – 0.80（约 1/3）",
    { text: "所以：它在跑、接线对、幅度可测，只是总量效应不可与 seed 噪声区分。", bold: true },
  ], { x: ML, y: 2.42, w: 5.6, h: 3.1, fontSize: 12 });

  s.addText("结构性限制（来自代码）", { x: ML + 6.4, y: 2.05, w: 5.7, h: 0.3, fontFace: F, fontSize: 13.5, bold: true, color: K, margin: 0 });
  body(s, [
    "img_backbone 全部参数 requires_grad = False；vit 参数组被注释掉",
    "alignment 与 domain classifier 作用在 image_feature 上；img_backbone = DINOv3（冻结）",
    { text: "alignment 梯度到达 BEV 聚合层与 DINO 之后的投影，永远到不了 DINOv3 本身。它不是在让图像编码器更好地看见几何，而是在重塑冻结 DINO 特征的 BEV 聚合方式 —— DINO 没编码的东西它加不进去。", bold: true },
    "而且只探了损失权重空间的一个点：λ_spatial = 0.002（论文默认），λ_global = 0.1",
  ], { x: ML + 6.4, y: 2.42, w: 5.7, h: 3.4, fontSize: 12 });

  foot(s, "代码位置：rap_agent.py:571-572, 576　|　rap_model.py:123, 150-161　|　bevformer/image_encoder.py:105", 6.05);
  note(s, "1.5 分钟。这是 §3.6 里最硬的一条：它来自代码而不是分数，所以不受 forbidden_use 约束，可以下结论。\n它同时给 DriveWeave 的 A5（planner / backbone 适配）提供了机制理由——不只是「换个预算重做一遍」。");
}

/* ---------- 1.4e 离散噪声地板 ---------- */
{
  const s = slide();
  title(s, "离散噪声地板：补上报告 §12.1 自己标出的洞");
  lead(s, "五个离散 sub-score 的跨条件方向性成分只有种子噪声的 0.35 – 0.77 倍，而且五个全部在 seed 之间翻符号。");

  s.addChart(pres.ChartType.bar, [
    { name: "cross |dominance|（同 seed，只差 alignment）", labels: ["NC", "DAC", "TTC", "C", "DDC"], values: [0.0114, 0.0312, 0.0138, 0.0203, 0.0118] },
    { name: "same-condition null |dominance|（换 seed）", labels: ["NC", "DAC", "TTC", "C", "DDC"], values: [0.0173, 0.0499, 0.0312, 0.0264, 0.0336] },
  ], chartBase({
    x: ML, y: 1.95, w: 6.6, h: 3.6,
    barDir: "col", barGapWidthPct: 60,
    showLegend: true, legendPos: "t", legendFontFace: F, legendFontSize: 10, legendColor: K,
    valAxisMinVal: 0, valAxisMaxVal: 0.055,
  }));

  table(s, [
    ["", "cross", "null", "比", "cross churn", "null churn", "比"],
    ["NC", ".0114", ".0173", "0.66", "2.88%", "3.70%", "0.78"],
    ["DAC", ".0312", ".0499", "0.63", { text: "7.04%", bold: true }, { text: "7.15%", bold: true }, { text: "0.98", bold: true }],
    ["TTC", ".0138", ".0312", "0.44", "6.57%", "7.88%", "0.83"],
    ["C", ".0203", ".0264", "0.77", "2.99%", "3.30%", "0.91"],
    ["DDC", ".0118", ".0336", "0.35", "3.90%", "4.21%", "0.93"],
  ], { tbl: { x: ML + 7.1, y: 2.15, w: 4.99, colW: [0.63, 0.66, 0.66, 0.5, 0.98, 0.86, 0.7], rowH: 0.32 }, fontSize: 9.5 });

  s.addText("overall_verdict = all_discrete_metrics_indistinguishable_from_seed_noise", {
    x: ML + 7.1, y: 4.25, w: 4.99, h: 0.5, fontFace: F, fontSize: 11, bold: true, color: K, margin: 0, lineSpacingMultiple: 1.15,
  });

  s.addText("必须同页声明的不对称性", { x: ML + 7.1, y: 4.85, w: 4.99, h: 0.28, fontFace: F, fontSize: 12, bold: true, color: K, margin: 0 });
  body(s, [
    "cross 对共享 init checkpoint、batch 顺序、augmentation RNG，只差 alignment 梯度；null 对三者全不同 → null 携带的变异源更多",
    "所以 cross 落在 null 区间内是有意义的；cross 低于 null 区间是共享 seed 的必然结果，不构成任何证据",
  ], { x: ML + 7.1, y: 5.15, w: 4.99, h: 1.2, fontSize: 9.5 });

  foot(s, "解释规则逐字继承 summary_protocol_v2.json 的 noise_floor.interpretation_rule：cross |dom| 不超过 null 最大 |dom| → 必须写「效应不可与 seed 噪声区分」，不得写「无效应」或「有微小效应」。\n红线：「alignment 搬动了 1,860 个 DAC 场景，是净破坏」不成立 —— null 搬动得一样多甚至更多。", 5.75, { w: 6.6, h: 0.9 });
  note(s, "1 分钟。可以说的原话：「报告 §12.1 记了一个洞：离散指标没有零分布可比。我补上了。」\nfig_e 读法：灰带 = seed 噪声区间，灰点 = 6 个 null 对，蓝菱 = 3 个 cross 对；蓝菱落在灰带内 = 不可区分。\n脚本 scripts/alignment/discrete_noise_floor.py → summary/discrete_noise_floor.json + figs/fig_e_discrete_noise_floor.png");
}

/* ---------- 1.4c sub-scores ---------- */
{
  const s = slide();
  title(s, "这个 planner 的画像：不撞、不越线、不违规，就是不走");
  lead(s, "75.8 与 93.8 之间那 18 分的缺口，主要在进展，不在安全。");

  s.addChart(pres.ChartType.bar, [{
    name: "NoAlign sub-score",
    labels: ["NC", "DAC", "TTC", "C", "DDC", "EP"],
    values: [97.56, 93.26, 91.35, 97.32, 97.85, 61.37],
  }], chartBase({
    x: ML, y: 2.05, w: 7.6, h: 3.9,
    barDir: "col", barGapWidthPct: 90,
    chartColors: [BAR2, BAR2, BAR2, BAR2, BAR2, BAR1],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00",
    valAxisMinVal: 0, valAxisMaxVal: 110,
  }));

  body(s, [
    { text: "五项安全类指标 91–98 分", bold: true },
    "NC 97.56　DAC 93.26　TTC 91.35",
    "C 97.32　DDC 97.85",
    { text: "进展指标 EP 只有 61.37", bold: true },
    "这是后面 Part 2「Stage 2 gain 必须同时查 EP」那条规则的活标本",
  ], { x: ML + 8.1, y: 2.35, w: 4.0, h: 3.4, fontSize: 12.5 });

  note(s, "0.5 分钟。一句话带过，但要留住这个印象——Part 2 讲指标选择时会回指本页。");
}

/* ---------- 1.5 场景 vs 运行 ---------- */
{
  const s = slide();
  title(s, "场景固有 vs 运行随机：把 6 个 run 当成 6 次重复观测");
  lead(s, "242 个场景六次运行全部硬失败。如果失败是随机的，期望 0.0045 个 —— 差 54,000 倍。");

  para(s, "设计要点：这个分析全程不比较两个条件，因此合法地绕开了 forbidden_use。", { y: 1.95, h: 0.35, fontSize: 12.5 });

  table(s, [
    ["现象（六次运行全部满足）", "实测", "随机预期", "倍数"],
    ["硬失败　NC × DAC == 0", { text: "242（1.99%）", bold: true }, "0.0045", { text: "≈ 54,000×", bold: true }],
    ["落在 EP 最低十分位", { text: "207", bold: true }, "0.0034", "≈ 60,000×"],
    ["安全项满分且 EP < 0.30", { text: "31", bold: true }, "3e-7", "—"],
  ], { tbl: { y: 2.5, colW: [5.293, 2.6, 2.1, 2.1], rowH: 0.4 } });

  body(s, [
    "逐次运行的硬失败率在 5.71% – 11.72% 之间摆动（2.05 倍），但那 242 个场景一次不落",
    "safe & EP<0.30 逐 run：413 / 142 / 86 / 365 / 178 / 224（4.8 倍摆动），重合的仍是那 31 个",
  ], { y: 4.5, h: 1.2, fontSize: 12.5 });

  foot(s, "脚本：scripts/alignment/scene_vs_run_decomposition.py　产物：outputs/alignment_stage_a/summary/scene_vs_run_decomposition.json", 6.0);
  note(s, "1 分钟。强调「全程不比较两个条件」——这是这段分析合法的唯一理由，必须先说出口。");
}

/* ---------- 1.5b 分布 ---------- */
{
  const s = slide();
  title(s, "两个尾巴都比随机厚");
  lead(s, "如果失败在 run 之间独立，中间应该是一座山。实测是两端高、中间塌。");

  s.addChart(pres.ChartType.bar, [
    { name: "实测", labels: ["0 次", "1 次", "2 次", "3 次", "4 次", "5 次", "6 次"], values: [9683, 947, 500, 314, 264, 196, 242] },
    { name: "独立性预期", labels: ["0 次", "1 次", "2 次", "3 次", "4 次", "5 次", "6 次"], values: [6999, 4055, 964, 120, 8.3, 0.30, 0.0045] },
  ], chartBase({
    x: ML, y: 2.0, w: 8.0, h: 4.0,
    barDir: "col", barGapWidthPct: 60,
    showLegend: true, legendPos: "t", legendFontFace: F, legendFontSize: 11, legendColor: K,
    valAxisLogScaleBase: 10, valAxisMinVal: 0.001, valAxisMaxVal: 10000,
  }));

  s.addText("横轴 = 一个场景在 6 次运行中硬失败的次数（对数纵轴）", {
    x: ML, y: 6.05, w: 8.0, h: 0.3, fontFace: F, fontSize: 10, color: K, margin: 0,
  });

  body(s, [
    { text: "6 次全败：242 vs 0.0045", bold: true },
    { text: "0 次失败：9,683 vs 6,999", bold: true },
    "中间档（1–3 次）全部低于独立预期",
    "→ 失败不是每次运行独立掷骰子的结果，而是场景自带的属性",
    { text: "顺带一个评测层面的推论", bold: true },
    "9,683 六次全对 + 242 六次全错 → 真正的争议区只有 2,221 个（18.3%）",
    "aggregate PDMS 把任何效应稀释约 5 倍",
  ], { x: ML + 8.5, y: 2.35, w: 3.6, h: 3.7, fontSize: 12 });

  foot(s, "⚠️ 若要按争议区分层报告，分层必须由独立的参照 run 集定义，否则就是循环论证。", 6.4, { w: 8.0, h: 0.32 });
  note(s, "1 分钟。对应 fig_b。若已从集群同步 figs/fig_b*.png，可直接替换本页图表。\n右下角那两句是 §3.6 启示②：评测目标本身选错了粒度。");
}

/* ---------- 1.5c 稳定性统计 ---------- */
{
  const s = slide();
  title(s, "不是「撞车 / 不撞车」的二值分裂造成的假象");
  lead(s, "排除全部硬失败之后，ICC 反而更高：0.601 → 0.761。");

  table(s, [
    ["统计量", "取值", "读法"],
    ["ICC(1)　全体", "0.601", "60% 的 PDMS 方差由场景决定"],
    ["ICC(1)　排除硬失败", { text: "0.761", bold: true }, "在 9,683 个从未硬失败的场景内，仍有 76%"],
    ["跨运行 Spearman", "0.776", "场景排序在 6 次运行间高度一致"],
    ["失败集 pairwise Jaccard", "0.374", "独立预期 0.046"],
  ], { tbl: { y: 2.05, colW: [3.2, 1.9, 6.993], rowH: 0.42 } });

  s.addText("constant-velocity 交叉验证", { x: ML, y: 4.35, w: CW, h: 0.3, fontFace: F, fontSize: 13.5, bold: true, color: K, margin: 0 });
  body(s, [
    "CV 全体通过率 33.1%；在 242 个铁核内只有 7.9%（19/242，独立预期 80）",
    "→ 铁核对不含学习的规则策略也难 4.2 倍",
    { text: "红线：「CV 能过而 planner 全败」的叙事不成立（19 低于预期 80），不要用。", bold: true },
  ], { y: 4.72, h: 1.4, fontSize: 12.5 });

  foot(s, "必须主动说：CV 的一致性有机械成分——直行策略必然在弯道出界。所以只用「4.2 倍」这个相对量，不做绝对解读。", 6.25);
  note(s, "1 分钟。ICC 排除硬失败后升高，是堵住「二值分裂」反驳的关键，对应 fig_c。");
}

/* ---------- 1.5d 图位 ---------- */
{
  const s = slide();
  title(s, "三张图的分工", "fig_a 提出 · fig_b 证明 · fig_c 加固");
  lead(s, "fig_c 的存在意义，是堵住「你那 60% ICC 只是撞车 / 不撞车的二值分裂」这个反驳。", 1.32);

  const items = [
    ["fig_a　场景 × 运行 热力图", "提出", "横向条纹：分数由行（场景）决定，不由列（运行）决定。定性证据。"],
    ["fig_b　重合度柱状图", "证明", "实测 vs 独立预期，54,000 倍。但只覆盖硬失败这一个维度。"],
    ["fig_c　EP 直方图", "加固 + 扩展", "换连续指标，且只取全无硬失败的场景，现象照旧；同时暴露 seed 方差在 EP 上最大。"],
  ];
  items.forEach((it, i) => {
    const y = 2.15 + i * 1.28;
    s.addText(it[0], { x: ML, y, w: 4.2, h: 0.32, fontFace: F, fontSize: 13.5, bold: true, color: K, margin: 0 });
    s.addText(it[1], { x: ML + 4.3, y, w: 1.6, h: 0.32, fontFace: F, fontSize: 12, color: K, margin: 0 });
    s.addText(it[2], { x: ML + 6.0, y, w: 6.09, h: 0.9, fontFace: F, fontSize: 12.5, color: K, margin: 0, lineSpacingMultiple: 1.15 });
  });

  foot(s, "本页为图位说明页。三张图的 png 在 outputs/alignment_stage_a/summary/figs/ 下，做 slides 时从集群同步后替换本页。", 6.15);
  note(s, "0.5 分钟。若三张图已同步，本页可以删掉，直接把 fig_a 放大成一整页。");
}

/* ---------- 1.6 画像资格 ---------- */
{
  const s = slide();
  title(s, "先解决资格问题：这 242 个是不是只是某几段路？");
  lead(s, "64 个 log，top-3 只占 17.4%，未触发混杂标记 —— 可以解释为场景类型效应。");

  table(s, [
    ["子集", "场景数", "logs", "top-3 占比", "判定"],
    ["hard_core", "242", "64", { text: "17.4%", bold: true }, { text: "可用：场景类型效应", bold: true }],
    ["ep_bottom", "207", "42", "44.9%", "log_confounded → 画像作废"],
    ["safe_stalled", "31", "16", "45.2%", "log_confounded → 画像作废"],
    ["failed_once", "947", "115", "13.5%", "可用"],
    ["never_failed", "9,683", "136", "12.2%", "可用（对照组）"],
  ], { tbl: { y: 2.05, colW: [2.4, 1.5, 1.3, 1.9, 4.993], rowH: 0.36 } });

  body(s, [
    { text: "画像作废 ≠ 重合度结论作废。", bold: true },
    "207 / 31 的「六次全部重合」讲的是跨运行可复现性，与 log 集中无关，继续用；只是不再给它们配元数据画像。",
  ], { y: 4.45, h: 1.0, fontSize: 12.5 });

  foot(s, "验收：endpoint 自校验 2,000 条 max |Δ| = 1.42e-14（阈值 1e-3）；图像覆盖 12,146 / 12,146 = 100%；每格含 10,000 次 log-cluster bootstrap + 2,000 次随机零分布。", 5.7);
  note(s, "1 分钟。主动把两个作废的子集画出来——当场演示自我审查标准，比等着被问强得多。");
}

/* ---------- 1.6b 三个不是 ---------- */
{
  const s = slide();
  title(s, "铁核画像（一）：先排除三个「不是」");
  lead(s, "不是拥挤交互问题，不是看不清的问题，也不是导航指令能预测的问题。");

  const rows = [
    ["失败方式", "197 个 DAC = 0（驶出可行驶区），仅 44 个 NC = 0（碰撞）", "→ 几何 / 路形，不是交互"],
    ["周围车辆", "0–5 辆档 .356 → .483（1.36×）；> 20 辆档 .118 → .050（0.42×）", "→ 车更少，不是更多"],
    ["图像统计", "illumination 0.79×　contrast 1.04×　blur_like 1.00×（全部落在零分布内）", "→ 与全体无差异"],
    ["导航指令", "GO_STRAIGHT 2.18%　TURN_LEFT 1.96%　TURN_RIGHT 1.08%（全局 1.99%）", "→ 几乎无区分度"],
  ];
  rows.forEach((r, i) => {
    const y = 2.05 + i * 1.05;
    s.addText(r[0], { x: ML, y, w: 1.7, h: 0.3, fontFace: F, fontSize: 13.5, bold: true, color: K, margin: 0 });
    s.addText(r[1], { x: ML + 1.85, y, w: 7.0, h: 0.72, fontFace: F, fontSize: 12.5, color: K, margin: 0, lineSpacingMultiple: 1.15 });
    s.addText(r[2], { x: ML + 9.0, y, w: 3.09, h: 0.6, fontFace: F, fontSize: 12.5, bold: true, color: K, margin: 0, lineSpacingMultiple: 1.15 });
  });

  foot(s, "红线：「图像三项无差异 = 视觉不是问题」是错的。只说明这三个全局统计量解释不了失败；遮挡 / 反光 / 物体类别它们测不到。", 6.2);
  note(s, "1 分钟。三个「不是」按顺序念，节奏要快，重点留给下一页的「是什么」。");
}

/* ---------- 1.6c 去混杂 ---------- */
{
  const s = slide();
  title(s, "去混杂：「车少更难」与「地理」是两个各自独立的效应");
  lead(s, "同城内部仍然单调（Boston 4.37% → 1.60%）；而同为 0–5 辆档，Boston 是 Vegas 的 7.8 倍。");

  table(s, [
    ["每格 hard-core 率（全局 1.99%）", "0–5 辆", "6–10 辆", "11–20 辆", "> 20 辆"],
    ["us-ma-boston", { text: "4.37%", bold: true }, "3.04%", "2.91%", "1.60%"],
    ["us-pa-pittsburgh-hazelwood", { text: "4.30%", bold: true }, "1.84%", "1.06%", "3.70% ¹"],
    ["sg-one-north", { text: "2.05%", bold: true }, "0.50%", "0.00%", "0.00%"],
    ["us-nv-las-vegas-strip", "0.56%", "0.93%", "0.53%", "0.34%"],
  ], { tbl: { y: 2.05, colW: [4.293, 1.95, 1.95, 1.95, 1.95], rowH: 0.38 } });

  body(s, [
    "「车少更难」成立：4 城中 3 城同城内单调下降；Vegas 平坦，但整城每一格都低于 1%",
    "「地理」也成立且更强：同为 0–5 辆档，Boston 4.37% 是 Vegas 0.56% 的 7.8 倍",
    { text: "新加坡基本不动（.158 → .140，0.89×）→ 左侧通行那条解释链不成立，不要用。", bold: true },
  ], { y: 4.2, h: 1.5, fontSize: 12.5 });

  foot(s, "¹ Pittsburgh「> 20 辆」格仅占该城 1.1% 场景，n 极小，不引用。　红线：map_location 只能说 geography / domain bundle，不得解释为道路风格或天气。", 6.1);
  note(s, "1 分钟。这页是防守页——先自己把混杂拆干净，后面「是什么」才立得住。");
}

/* ---------- 1.7 TVD ---------- */
{
  const s = slide();
  title(s, "铁核画像（二）：九个标签里只有两个真的偏离");
  lead(s, "endpoint_dy 是偏离最大的一个：TVD .3426，是随机零分布 p95 的 4.29 倍。");

  s.addChart(pres.ChartType.bar, [
    { name: "TVD（实测）", labels: ["endpoint_dy", "map_location", "actor_density", "endpoint_dyaw", "route_command", "contrast", "blur_like", "illumination", "endpoint_dx"], values: [0.3426, 0.2525, 0.1271, 0.0697, 0.0629, 0.0422, 0.0363, 0.0530, 0.0169] },
    { name: "随机零分布 p95", labels: ["endpoint_dy", "map_location", "actor_density", "endpoint_dyaw", "route_command", "contrast", "blur_like", "illumination", "endpoint_dx"], values: [0.0798, 0.0772, 0.0776, 0.0573, 0.0595, 0.0405, 0.0363, 0.0668, 0.0233] },
  ], chartBase({
    x: ML, y: 1.95, w: 8.1, h: 4.1,
    barDir: "bar", barGapWidthPct: 55,
    showLegend: true, legendPos: "t", legendFontFace: F, legendFontSize: 11, legendColor: K,
    valAxisMinVal: 0, valAxisMaxVal: 0.36,
    catAxisLabelFontSize: 10.5,
  }));

  body(s, [
    { text: "冲出零分布带", bold: true },
    "endpoint_dy　4.29×",
    "map_location　3.27×",
    "actor_density　1.64×",
    { text: "留在带内（= 噪声）", bold: true },
    "route_command 1.06×　contrast 1.04×",
    "blur_like 1.00×　illumination 0.79×",
    "endpoint_dx 0.73×",
  ], { x: ML + 8.6, y: 2.15, w: 3.5, h: 3.9, fontSize: 12 });

  note(s, "1 分钟。指出 route_command 落在带内——它被同一套流程判为噪声，正好说明这套流程会产生阴性结果，不是只挑赢的。");
}

/* ---------- 1.7b 单调性 ---------- */
{
  const s = slide();
  title(s, "单调性：失败频次越高，偏离越大");
  lead(s, "三个显著标签全部单调递增；route_command 不单调 —— 佐证它本就是噪声。");

  s.addChart(pres.ChartType.line, [
    { name: "endpoint_dy", labels: ["never_failed", "failed_once", "hard_core"], values: [0.0644, 0.2407, 0.3426] },
    { name: "map_location", labels: ["never_failed", "failed_once", "hard_core"], values: [0.0302, 0.1181, 0.2525] },
    { name: "actor_density", labels: ["never_failed", "failed_once", "hard_core"], values: [0.0266, 0.0850, 0.1271] },
    { name: "route_command（对照）", labels: ["never_failed", "failed_once", "hard_core"], values: [0.0320, 0.1322, 0.0629] },
  ], chartBase({
    x: ML, y: 2.0, w: 7.9, h: 4.0,
    chartColors: [BAR1, "666666", "999999", "CCCCCC"],
    lineSize: 2.25, lineDataSymbol: "circle", lineDataSymbolSize: 6,
    showLegend: true, legendPos: "b", legendFontFace: F, legendFontSize: 11, legendColor: K,
    valAxisMinVal: 0, valAxisMaxVal: 0.38, valAxisTitle: "TVD",
  }));

  table(s, [
    ["TVD", "never", "once", "core"],
    ["endpoint_dy", ".0644", ".2407", { text: ".3426", bold: true }],
    ["map_location", ".0302", ".1181", { text: ".2525", bold: true }],
    ["actor_density", ".0266", ".0850", { text: ".1271", bold: true }],
    ["route_command", ".0320", ".1322", ".0629"],
  ], { tbl: { x: ML + 8.4, y: 2.6, w: 3.69, colW: [1.59, 0.7, 0.7, 0.7], rowH: 0.34 }, fontSize: 10.5 });

  foot(s, "陷阱：never_failed 的 route TVD .0320 / null .0044 = 7.3 倍看似极显著，但绝对 TVD 只有 0.03 —— n = 9,683 让零分布极小。比值大 ≠ 重要，两个数必须一起看。", 6.15);
  note(s, "1 分钟。单调性是本轮画像里最强的论据，因为它不依赖任何单点显著性。");
}

/* ---------- 1.7c 决定性发现 ---------- */
{
  const s = slide();
  title(s, "决定性发现：导航指令看不见几何需求");
  lead(s, "在指令说「直行」的场景里，被稳定做坏的，是专家实际上要向左横移 2 米以上的那些。");

  s.addChart(pres.ChartType.bar, [
    { name: "全体 GO_STRAIGHT", labels: ["dy > 2\n（左）", "dy 0.5 ~ 2\n（左）", "dy −0.5 ~ 0.5\n（真直行）", "dy < −2\n（右）"], values: [0.147, 0.138, 0.529, 0.101] },
    { name: "hard_core GO_STRAIGHT", labels: ["dy > 2\n（左）", "dy 0.5 ~ 2\n（左）", "dy −0.5 ~ 0.5\n（真直行）", "dy < −2\n（右）"], values: [0.432, 0.295, 0.199, 0.045] },
  ], chartBase({
    x: ML, y: 1.95, w: 7.9, h: 3.85,
    barDir: "col", barGapWidthPct: 60,
    showLegend: true, legendPos: "t", legendFontFace: F, legendFontSize: 11, legendColor: K,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000",
    valAxisMinVal: 0, valAxisMaxVal: 0.62,
  }));

  body(s, [
    { text: "GO_STRAIGHT 占 hard_core 的 72.7%（n = 176）", bold: true },
    "实际向左横移：28.5% → 72.7%",
    "实际基本直行：52.9% → 19.9%",
    "dy > 2 一档：2.94 倍",
    { text: "route 级别几乎没变", bold: true },
    "hard_core {直 .727　左 .202　右 .070}",
    "全体　　 {直 .664　左 .206　右 .130}",
    "TURN_LEFT 占比完全没变",
  ], { x: ML + 8.4, y: 2.1, w: 3.69, h: 3.9, fontSize: 11.5 });

  foot(s, "+y = 左 已由数据确认（TURN_LEFT 88.6% 落 >2，TURN_RIGHT 83.2% 落 <−2）。TURN_RIGHT 行 n 仅 17，数字乱跳是噪声，不要引用。", 6.05);
  note(s, "1.5 分钟。这是 Part 1 的封面图。念完中间那句结论后停一拍再往下走。");
}

/* ---------- 1.7d 合起来的读法 ---------- */
{
  const s = slide();
  title(s, "合起来的读法");
  lead(s, "planner 稳定失败在「指令说直行、实际需要大幅横移」的空旷场景，失败方式是横着开出路面。\n不是交互问题，不是看不清的问题。缺的是几何 / 行为层面的监督。", 1.14, 1.05);

  body(s, [
    "197 / 242 是驶出可行驶区，仅 44 是碰撞",
    "周围车更少（0–5 档 1.36×，>20 档 0.42×），去混杂后同城内仍单调",
    "需要的横向位移显著更大，且集中在被标为直行的场景",
    "亮度 / 对比度 / 清晰度与全体无差异",
    "转向指令无区分度",
  ], { y: 2.55, h: 2.1, fontSize: 13 });

  s.addText("必须同页说的三句诚实话", { x: ML, y: 4.7, w: CW, h: 0.3, fontFace: F, fontSize: 13.5, bold: true, color: K, margin: 0 });
  body(s, [
    "画像标签是在看到分数之后生成的 → descriptive / exploratory，非 confirmatory",
    "图像阈值取自 navtest 自身分位数，是对 spec §5「只从 train split 计算」的有意偏离（train 侧真图仅 21% 可用）",
    "左右不对称照实报告，不做因果归因；新加坡左侧通行混在池中（占 15.8%），对应 DriveWeave §6.1 已列的 traffic-rule conflict",
  ], { y: 5.05, h: 1.5, fontSize: 12 });

  note(s, "1 分钟。三句诚实话不要被问了才说——主动说完，这段的可信度就锁死了。");
}

/* ---------- 1.8 MDE ---------- */
{
  const s = slide();
  title(s, "附带产出：噪声地板与指标选择");
  lead(s, "3 个 seed 的检出下限是 4.56 分，而 RAP 自己报告的 recovery 效应是 4.4 分 —— 用现在这个配置，连已发表的效应都测不出来。");

  s.addText("最小可检出效应（MDE）", { x: ML, y: 2.15, w: 5.6, h: 0.3, fontFace: F, fontSize: 13.5, bold: true, color: K, margin: 0 });
  table(s, [
    ["目标效应量", "所需 seed 数"],
    ["4.4 分（RAP recovery）", { text: "4", bold: true }],
    ["3.0 分", "7"],
    ["2.0 分", "16"],
  ], { tbl: { x: ML, y: 2.55, w: 5.6, colW: [3.4, 2.2], rowH: 0.36 }, fontSize: 12 });
  body(s, [
    "seed sd = 2.82 分 → n=3 的检出下限 4.56 分",
    "提案里 ε 定义为「多随机种子的实验波动」，是个待填的空 —— 现在有数了",
  ], { x: ML, y: 4.1, w: 5.6, h: 1.2, fontSize: 12 });

  s.addText("指标选择", { x: ML + 6.4, y: 2.15, w: 5.7, h: 0.3, fontFace: F, fontSize: 13.5, bold: true, color: K, margin: 0 });
  table(s, [
    ["RAP Table 6", "无 recovery", "有 recovery"],
    ["v1 PDMS", "92.5", "92.5"],
    ["v2 EPDMS", "32.5", { text: "36.9", bold: true }],
  ], { tbl: { x: ML + 6.4, y: 2.55, w: 5.7, colW: [2.3, 1.7, 1.7], rowH: 0.36 }, fontSize: 12 });
  body(s, [
    "v1 对 recovery 完全不敏感，v2 才动",
    "加上我训出的「安全 93–98、EP 只有 61」这个模型",
    { text: "→ EPDMS 必须是 official final metric，且 Stage 2 gain 必须同时查 EP", bold: true },
  ], { x: ML + 6.4, y: 3.9, w: 5.7, h: 1.6, fontSize: 12 });

  foot(s, "必须主动说：MDE 是在 v1 PDMS 上测的，EPDMS 的方差方向未知。　离散指标那一侧的噪声地板已补齐（见前页），报告 §12.1 的洞已经合上。", 6.1);
  note(s, "0.5 分钟。两句压缩带过，不占主链。这是四条连接链里的第 2、第 4 条。");
}

/* ---------- 过渡 ---------- */
{
  const s = slide();
  title(s, "三个问题，各答了一半");
  lead(s, "我用实证消除法划掉了「优化侧」这个分支，并且定位到了失败的自由度——横向。\n而横向恰好是 recovery 扰动 δ = (Δx, Δy, Δθ, Δv) 直接作用的那个自由度。", 1.14, 1.05);

  table(s, [
    ["我问的", "本轮答到哪", "交给 DriveWeave"],
    ["该投在哪？", "优化侧已被实证消除（ICC 0.601 / 0.761）；缺口定位在横向几何监督", "S0 / S1 / S2 / S3 分开数据侧三个分支"],
    ["多大才算数？", "ε 有数了：≥ 3.2 分波动，需 ≥ 4 seed", "填上成功判据里那个待填的空"],
    ["用哪个指标？", "v1 对 recovery 不敏感 + 「高 safety / 低 EP」活标本", "EPDMS 为 final metric；Stage 2 gain 必须同时查 EP"],
  ], { tbl: { y: 2.75, colW: [2.2, 5.4, 4.493], rowH: 0.5 }, fontSize: 11.5 });

  body(s, [
    { text: "划清边界：navtest 是 nominal 起点，所以横移失败不等于恢复失败。我的分析排除的是优化侧，不是替数据侧选定了答案。", bold: true },
    "数据侧还剩三个分支：更多正常驾驶数据？目标数据集自己的 recovery 数据？还是跨数据集的 recovery 数据？",
    "而且这一轮不是白测：我这个 loss 项就是 DriveWeave Phase 2 的 L_struct（→ A6），而「backbone 冻结是硬天花板」这条给 A5 提供了机制理由 —— 两条消融轴现在都有了具体的、来自实测的动机。",
  ], { y: 5.05, h: 1.6, fontSize: 12 });

  note(s, "2 分钟。全场转折点。最后一句收在：「我划掉了一个分支，那个矩阵因此从一个设计选择，变成了必要步骤。」");
}

/* ============================ Part 2 divider ============================ */
divider("Part 2", "DriveWeave Proposal", "Source-Sensor-Free Cross-Dataset Recovery Scaling");

/* ---------- 2.1 大问题 ---------- */
{
  const s = slide();
  title(s, "大问题：recovery gap");
  lead(s, "Camera-based E2E planner 训练在 nominal manifold 上；部署后，小的规划误差把车推到日志里罕见的 off-nominal 状态。");

  s.addText("P_nominal ≠ P_recovery", { x: ML, y: 2.1, w: 5.8, h: 0.45, fontFace: F, fontSize: 21, bold: true, color: K, margin: 0 });
  body(s, [
    "人类驾驶日志集中在 P_nominal",
    "NAVSIM-v2 Stage 2 专门测 P_recovery：车辆已经产生横向偏移、heading mismatch 或异常速度之后，planner 能否安全且高效地回到合理驾驶状态",
    { text: "所以泛化的「更多 nominal samples」并不必然改善 Stage 2 —— 训练数据必须显式扩大 recovery-state support（O0）", bold: true },
  ], { x: ML, y: 2.7, w: 5.8, h: 3.0, fontSize: 12.5 });

  s.addText("Part 1 已经交出一个活标本", { x: ML + 6.4, y: 2.1, w: 5.7, h: 0.32, fontFace: F, fontSize: 13.5, bold: true, color: K, margin: 0 });
  body(s, [
    "安全类 sub-score 91–98，EP 只有 61.37",
    "197 / 242 个稳定失败是横着开出可行驶区",
    "而 NAVSIM-v2 的 Stage 2 扰动 δ = (Δx, Δy, Δθ, Δv) 施加的正是横向这个自由度",
    { text: "planner 最弱的自由度，恰好是 recovery 监督直接作用的那个", bold: true },
    "但要划清：navtest 是 nominal 起点，这不等于「恢复失败」",
  ], { x: ML + 6.4, y: 2.7, w: 5.7, h: 3.2, fontSize: 12.5 });

  note(s, "1.5 分钟。左边讲 recovery gap 的定义，右边把 Part 1 接上来。");
}

/* ---------- 2.2 先行研究 ---------- */
{
  const s = slide();
  title(s, "先行研究：四条已经成熟的路线");
  lead(s, "structured-to-vision transfer 不是我的 novelty，multi-dataset planning 不是，recovery augmentation 也不是。");
  s.addImage({ path: "assets/f1_abcd_top.png", x: (W - 11.0) / 2, y: 2.0, w: 11.0, h: 11.0 / 2.721 });
  foot(s, "A：同源 recovery 增强（RAP / SimScale / Gigapixel）　B：structured policy → vision（TerraTransfer）　C：多数据集 RGB 联合训练（HEAT / PixelPilot）　D：多数据集轨迹先验迁移（UniPlanner）\nRAP 同时是本提案的实现基座与主要受控 baseline。", 6.15, { h: 0.7 });
  note(s, "2 分钟。逐格 30 秒。重点让听众看到每格顶上那两个 YES/NO 芯片：Recovery-state coverage / Recovery across datasets。");
}

/* ---------- 2.3 小问题（一）：四个 Bottleneck ---------- */
{
  const s = slide();
  title(s, "先行研究未解决的问题（一）：四个 Bottleneck");
  lead(s, "四条路线里，没有任何一条同时拿到「Recovery-state coverage: YES」和「Recovery across datasets: YES」。");
  s.addImage({ path: "assets/f1_abcd_btm.png", x: (W - 11.6) / 2, y: 2.15, w: 11.6, h: 11.6 / 3.384 });
  foot(s, "A 的 recovery 多样性被单一数据源封顶；B 的 bounding-box 结构状态漏掉细粒度视觉线索；C 消费全部 source RGB 且异构联合训练可能负迁移；D 的 source 只贡献轨迹先验，本身不是可部署的相机 planner。", 5.95, { h: 0.6 });
  note(s, "1 分钟。四个红框逐格 15 秒。重点：没有任何一格同时拿到两个 YES。");
}

/* ---------- 2.3b 小问题（二）：四个 Observation ---------- */
{
  const s = slide();
  title(s, "先行研究未解决的问题（二）：四个 Observation");
  lead(s, "逻辑链不是并列罗列，而是 O1 →（O2, O3）→ O4。");

  table(s, [
    ["Obs", "性质", "缺口", "证据"],
    ["O1", "Opportunity", "planning-rich corpora 值得 scale", "1,200h 标注 vs 120h 图像。定性为 opportunity，不是 gap"],
    ["O2", "Engineering barrier", "用 source RGB 就要 sensor reconciliation", "HEAT 给 6-camera nuScenes 补空白视图；RAP 需 camera reorder + resize + 标定变换"],
    ["O3", "Learning barrier", "接口统一了，异构联合学习仍可能冲突", "HEAT 表：LTF 83.8 → 55.6，LAW 84.6 → 72.3，HEAT(full) 83.2 → 85.2"],
    ["O4", "Solution constraint", "source 去 image ≠ deployment 去 image", "structured teacher 缺 brake lights / hand signals"],
  ], { tbl: { y: 2.35, colW: [0.75, 2.3, 4.2, 4.843], rowH: 0.5 }, fontSize: 11 });

  s.addText("O0 定义任务本身：nominal logs 缺 recovery 覆盖 → structured counterfactual recovery → 多个数据集扩大 recovery 多样性。", {
    x: ML, y: 5.0, w: CW, h: 0.35, fontFace: F, fontSize: 12.5, bold: true, color: K, margin: 0,
  });
  foot(s, "红线：不能把 10:1 的 log-hour 比例写成「丢弃 90% 的独立驾驶知识」——data hours 不等于 unique planning knowledge。\nO3 只能推出「Direct multi-domain training does not automatically convert more datasets into better target planning」，不能推出「删掉 images 后 negative transfer 就自动消失」。", 5.6, { h: 0.7 });
  note(s, "1.5 分钟。讲完 O3 的 HEAT 表要停一拍——这是后面「不假设不掉点、胜利条件是持平」的前提。");
}

/* ---------- 2.4 最终目的 ---------- */
{
  const s = slide();
  title(s, "1. 最终目的");
  lead(s, "目标不是孤立最大化 Stage 2，是最大化官方 Final EPDMS，且 nominal 不许退。");

  s.addText("max  EPDMS_final(θ)　　s.t.　S₁(θ) ≥ S₁(θ₀) − ε　,　S₂(θ) ≥ S₂(θ₀) + δ", {
    x: ML, y: 2.2, w: CW, h: 0.5, fontFace: F, fontSize: 17, bold: true, color: K, margin: 0,
  });
  body(s, [
    "θ₀ = 完全相同 codebase、target visual budget 与训练预算下的 target-only baseline",
    { text: "ε 由多随机种子的实验波动确定 —— 这正是 Part 1 填上的空：≥ 3.2 分，需 ≥ 4 seed", bold: true },
    "Stage 1 / Stage 2 是诊断指标，Final EPDMS 才是官方最终指标",
  ], { y: 2.85, h: 1.4, fontSize: 12.5 });

  table(s, [
    ["成功分层", "判据", "解读"],
    ["Failure", "Stage 2 ↑，Stage 1 明显 ↓", "nominal-recovery trade-off"],
    ["Weak", "Stage 2 ↑，Stage 1 小幅 ↓ 且 Final EPDMS ↑", "刷榜可能有效，科学结论不干净"],
    ["Strong", "Stage 1 在波动范围内不退，Stage 2 显著 ↑，Final ↑", { text: "recovery without nominal regression", bold: true }],
    ["Ideal", "Stage 1 ↑，Stage 2 显著 ↑，Final ↑", "planning prior 同时改善两端"],
  ], { tbl: { y: 4.35, colW: [1.7, 5.6, 4.793], rowH: 0.38 }, fontSize: 11 });

  foot(s, "胜利条件是 Stage 1 持平 + Stage 2 显著涨，不是全面超越。", 6.35);
  note(s, "1 分钟。ε 那一行是全场最强的一次闭合——念到这里明确回指 Part 1。");
}

/* ---------- 2.5 Task 设定 ---------- */
{
  const s = slide();
  title(s, "2. Task 设定");
  lead(s, "Task、Method、Deployment 三者必须分开说，混在一起就会被误解成 sensor-agnostic。");

  s.addImage({ path: "assets/f1_ours.png", x: ML, y: 1.95, w: 4.2, h: 4.2 / 0.963 });
  const rows = [
    ["Task", "Recovery robustness without nominal regression"],
    ["Method", "Source-sensor-free cross-dataset structured recovery scaling"],
    ["Deployment", "target camera + 标准 ego status / navigation ONLY；不用 structured annotations，不用任何 source data"],
  ];
  rows.forEach((r, i) => {
    const y = 2.05 + i * 0.95;
    s.addText(r[0], { x: ML + 5.3, y, w: 1.5, h: 0.3, fontFace: F, fontSize: 13.5, bold: true, color: K, margin: 0 });
    s.addText(r[1], { x: ML + 6.8, y, w: 5.29, h: 0.85, fontFace: F, fontSize: 12.5, color: K, margin: 0, lineSpacingMultiple: 1.15 });
  });
  s.addText("「不使用 source images / calibration」的两层含义", { x: ML + 5.3, y: 4.95, w: 6.79, h: 0.3, fontFace: F, fontSize: 13, bold: true, color: K, margin: 0 });
  body(s, [
    "Capability：planning-only / motion-heavy 语料也能服务 camera E2E planning",
    "Controlled protocol：固定 target visual budget，排除额外 source RGB 预训练对结果的解释",
  ], { x: ML + 5.3, y: 5.3, w: 6.79, h: 1.0, fontSize: 11.5 });

  foot(s, "准确表述：不是「structured annotations 与传感器无关」，而是 DriveWeave training does not consume raw source sensor observations or source camera geometry.", 6.45);
  note(s, "1.5 分钟。图里 Phase 1 = Learning to Drive（NO source images / calibrations），Phase 2 = Learning to See（target real images only）。");
}

/* ---------- 2.5b Table 1 ---------- */
{
  const s = slide();
  title(s, "范式定位");
  lead(s, "DriveWeave 不是 A+B+C+D 的组合，是把 knowledge scaling 与 sensor coupling 这两件被前人绑在一起的事拆开。");
  s.addImage({ path: "assets/f1_table.png", x: ML, y: 2.15, w: CW, h: CW / 5.695 });
  body(s, [
    "左三格 / 第四格的 source 侧要么带 sensor，要么完全没有视觉分支",
    "O 的 source 侧只有 metadata，target 侧只有真实图像，两条路在同一个 Shared Planner 汇合",
    { text: "† 表中带标记的两列是本提案待验证的假设，不是已报告的结果。", bold: true },
  ], { y: 4.6, h: 1.6, fontSize: 12.5 });
  note(s, "1 分钟。只读最后一行：Recovery-state coverage YES(multi-source)† / Cross-dataset scaling Yes† / Needs source RGB No / Target visual cues Strong / Deployment Camera + ego status。");
}

/* ---------- 2.6 Motivation ---------- */
{
  const s = slide();
  title(s, "3. Motivation");
  lead(s, "两条独立证据把设计空间夹到只剩一个出口：source 端必须无相机，target 端必须有相机。");

  s.addText("上夹　扩展需求", { x: ML, y: 2.15, w: 5.7, h: 0.32, fontFace: F, fontSize: 13.5, bold: true, color: K, margin: 0 });
  body(s, [
    "假设 recovery 监督确实有用 —— 那是 Week 2 的 S1 − S0 要先过的门",
    "接下来的问题就变成：recovery 的场景多样性从哪来？",
    "目标数据集自己的日志能造出的 recovery 状态，受限于它自己跑过的那些路",
    { text: "要扩，只能上更多真实数据集", bold: true },
  ], { x: ML, y: 2.55, w: 5.7, h: 2.5, fontSize: 12.5 });

  s.addText("下夹　带上 source RGB 的代价", { x: ML + 6.4, y: 2.15, w: 5.7, h: 0.32, fontFace: F, fontSize: 13.5, bold: true, color: K, margin: 0 });
  body(s, [
    "HEAT 要给 nuScenes 补空白视图凑齐输入接口",
    "RAP 混训要统一相机顺序、resize 到 576×1024、旋转缩放标定矩阵",
    { text: "这些是妥协，不是方法", bold: true },
    "而且 naive 混合会掉点：HEAT 表里 LTF 从 83.8 掉到 55.6",
  ], { x: ML + 6.4, y: 2.55, w: 5.7, h: 2.5, fontSize: 12.5 });

  s.addText("唯一干净的出口：让 source 端的扩展完全发生在 canonical structured space 里。", {
    x: ML, y: 5.15, w: CW, h: 0.35, fontFace: F, fontSize: 14, bold: true, color: K, margin: 0,
  });
  foot(s, "分工澄清：去相机是对 source 的要求，不是对整个 pipeline 的洁癖。部署目标的相机几何本来就是已知的，所以 Phase 2 照样用 RAP 的 perspective raster。\n负控制：本轮跑的是纯 regularization、零新信息的测量 —— 它预先削减了 R7 的捷径解释。", 5.6, { h: 0.72 });
  note(s, "1.5 分钟。接缝 D。不要从「让 alignment 从 regularizer 变成 knowledge carrier」起手，那是空想推理。");
}

/* ---------- 2.7 方法总览 ---------- */
{
  const s = slide();
  title(s, "4. 提案方法详情：两个训练阶段 + 相机-only 部署");
  lead(s, "Phase 1 学会开（不看图），Phase 2 学会看（只看 target 图），部署时只剩相机。");
  s.addImage({ path: "assets/f2_all.png", x: (W - 10.7) / 2, y: 1.85, w: 10.7, h: 10.7 / 2.288 });
  foot(s, "图例：蓝框 = structured branch，橙框 = visual branch，绿框 = shared / deployed；火焰 = 可训练，雪花 = 冻结；灰虚线 = train-only（部署时丢弃）；红虚线 = 监督 / 梯度流；sg[·] = stop-gradient；×2 = 同一份权重前向两次。", 6.58, { w: 10.9, h: 0.5 });
  note(s, "0.5 分钟。先给全景，再逐相拆。讲图例只花 20 秒，但必须讲，否则后面三页听众读不懂。");
}

/* ---------- 2.7a Phase 1 ---------- */
{
  const s = slide();
  title(s, "Phase 1：Multi-Dataset Structured Recovery Pretraining");
  lead(s, "整个 Phase 1 没有任何图像输入，也不训练任何图像编码器。");
  s.addImage({ path: "assets/f2_p1.png", x: ML, y: 1.85, w: 3.65, h: 3.65 / 0.807 });

  body(s, [
    { text: "Canonicalizer C_i", bold: true },
    "deterministic · no parameters · train-only",
    "只负责 common interface，不负责消除 behavior / policy / city 差异",
    { text: "Canonical nominal state S_nominal", bold: true },
    "ego-centric vectorized BEV，ego 在原点朝 +x，范围 [−32, 32] × [−32, 96] m",
    "map · agents · route/navigation · ego state (v, a, ω) · traffic light · per-field availability mask",
    "UNKNOWN 与 NONE 严格区分；ego state 内嵌在 S_nominal 中，Phase 1 没有独立 ego encoder",
    { text: "structured recovery construction（train-only）", bold: true },
    "Recovery intervention T_δ，δ = (dx, dy, dθ, dv) → Recovery target generator → validity filter",
    "标签用 trajectory retrieval + structured scorer：kinematics · drivable / lane direction · collision / TTC · route progress · comfort",
    { text: "L_P1 = L_nom + λ_r·L_rec + λ_a·L_anchor", bold: true },
    "L_anchor 是可消融的 nominal-preservation 项，不是假设一定有益",
  ], { x: ML + 4.05, y: 1.9, w: 8.04, h: 4.5, fontSize: 11 });

  foot(s, "第二次点名横向：Part 1 定位的那个弱自由度，就是这里的 dy。scorer 的两条准则 drivable / lane direction 与 route progress，正对应 197 个 DAC 失败与 EP 61。", 6.5, { h: 0.32 });
  note(s, "1.5 分钟。合规声明要主动说：训练 perturbation 与官方 Stage 2 grid 相关但不相同。");
}

/* ---------- 2.7b Phase 2 ---------- */
{
  const s = slide();
  title(s, "Phase 2：Target-Only Visual Recovery Grounding");
  lead(s, "结构化支整体冻结并只在训练期存在；这一阶段真正训练的只有 visual adapter。");
  s.addImage({ path: "assets/f2_p2.png", x: ML, y: 1.85, w: 4.2, h: 4.2 / 0.928 });

  body(s, [
    { text: "两条支路，同一个 planner 前向两次（×2）", bold: true },
    "Nominal：target metadata → C_target → E_meta ❄ → G_meta ❄ → F_meta_nom；target real RGB → B_vis ❄（frozen RAP backbone）→ G_img 🔥 → F_img_nom",
    "Recovery：target metadata → C_target → T_delta → F_meta_rec；RAP-style target-only perspective rasterization → F_img_rec",
    { text: "L_P2 = L_GT_nom + λ_d·L_dec + λ_s·L_struct + λ_v·L_visual_rec", bold: true },
    "L_GT_nom：target real-image trajectory supervision，为 image-private cues 留学习通路",
    "L_dec：structured / visual decision consistency（注意图上两处 sg[·]，结构支是 teacher，梯度不回流）",
    "L_struct：batch-relational low-rank structural alignment —— 本轮实验测的就是这一项，A6 就是它的消融",
    "L_visual_rec：target recovery observation 上的 planning / alignment 监督",
    { text: "Planner 三个变体", bold: true },
    "Controlled 冻结（测 transfer retention）/ Performance small-lr unfreeze / Residual head τ = τ_base + g(O,H)·Δτ_rec",
  ], { x: ML + 4.6, y: 1.9, w: 7.49, h: 4.5, fontSize: 10.5 });

  foot(s, "split integrity（主动说）：所有 recovery observation 只由 navtrain 生成；navhard_two_stage / warmup_two_stage 与 private test follow-up observations 从不用于训练。", 6.5, { h: 0.32 });
  note(s, "1.5 分钟。讲到 L_struct 时手指图里那个菱形——这是四条连接链里的第 1 条（模块同一）。\n若被追问「为什么换成 batch-relational low-rank」：本轮的 alignment 是拍平 MSE(λ=0.002) + 全局域对抗(λ=0.1)，两者都不保留空间结构；而 81% 的残余失败是 drivable-area 几何。旁证是 TerraTransfer Table 6（0.319 / 0.307 / 0.490）。所以 L_struct 不是换个写法，是针对失败模式改损失形式。");
}

/* ---------- 2.7c 部署 ---------- */
{
  const s = slide();
  title(s, "Deployment：Camera-Only Inference");
  lead(s, "部署时只剩相机观测 + 标准 ego status / navigation。所有 source data、canonicalizer、structured encoder、recovery generator 与全部标注都被丢弃。");
  s.addImage({ path: "assets/f2_p3.png", x: ML, y: 1.95, w: 2.85, h: 2.85 / 0.641 });

  s.addText("跨阶段共享权重 —— 这就是知识从 structured 流到 camera 的物理通道", {
    x: ML + 3.3, y: 2.0, w: 8.79, h: 0.32, fontFace: F, fontSize: 13, bold: true, color: K, margin: 0,
  });
  table(s, [
    ["模块", "Phase 1", "Phase 2", "Deploy"],
    ["Structured Encoder E_meta", "训练", "冻结", "丢弃"],
    ["Structured Adapter G_meta", "训练", "冻结", "丢弃"],
    ["Visual Encoder B_vis", "—", "冻结（RAP backbone）", "参与"],
    ["Visual Adapter G_img", "—", "训练", "参与"],
    ["Shared Planner P_shared", { text: "训练", bold: true }, { text: "冻结 / small-lr", bold: true }, { text: "参与", bold: true }],
    ["optional residual head", "—", "可选", "可选"],
  ], { tbl: { x: ML + 3.3, y: 2.42, w: 8.79, colW: [3.19, 1.5, 2.6, 1.5], rowH: 0.34 }, fontSize: 11 });

  foot(s, "P_shared 贯穿三个阶段：Phase 1 在结构化空间里学到的 recovery 知识，通过它被 Phase 2 的视觉支继承，最后进入相机-only 的部署模型。", 5.0, { x: ML + 3.3, w: 8.79, h: 0.6 });
  note(s, "1 分钟。图右下角那张 Shared weights across phases 表是整张图的证明，本页把它转成中文表格。");
}

/* ---------- 2.8 矩阵 ---------- */
{
  const s = slide();
  title(s, "受控实验矩阵与识别性");
  lead(s, "矩阵存在的理由不是穷举，是把「多数据集」和「recovery-specific」这两个变量分开，否则任何 gain 都无法归因。");

  table(s, [
    ["ID", "输入", "多数据集", "Recovery 数据", "问题"],
    ["S0 / S1 / S2 / S3", "Structured", "No / No / Yes / Yes", "No / Yes / No / Yes", "privileged 空间里的诊断四格"],
    ["C0 / C1 / C2 / C3", "Camera", "No / No / Yes / Yes", "No / Yes / No / Yes", "可部署四格，C3 = DriveWeave full"],
  ], { tbl: { y: 2.35, colW: [2.5, 1.8, 2.4, 2.4, 2.993], rowH: 0.42 }, fontSize: 11.5 });

  s.addText("四个关键比较", { x: ML, y: 3.7, w: 5.7, h: 0.3, fontFace: F, fontSize: 13.5, bold: true, color: K, margin: 0 });
  body(s, [
    "S1 − S0：target-only structured recovery 增益",
    "S3 − S1：external 数据集的结构化增量价值",
    "C3 − C1：最终可部署的增量价值",
    "C3 − S3：迁移保持率，不是 external 的因果效应",
  ], { x: ML, y: 4.05, w: 5.7, h: 1.8, fontSize: 12 });

  s.addText("五个识别性负控制", { x: ML + 6.4, y: 3.7, w: 5.7, h: 0.3, fontFace: F, fontSize: 13.5, bold: true, color: K, margin: 0 });
  body(s, [
    "source scene-trajectory pairing shuffle",
    "source recovery label shuffle",
    "matched-volume duplicate-target control",
    "ego-status masked evaluation",
    "leave-overlap-city-out",
  ], { x: ML + 6.4, y: 4.05, w: 5.7, h: 1.8, fontSize: 12 });

  foot(s, "R7 identifiability failure（主动说）：若 source shuffle 仍产生同等 gain，则不能解释为 external recovery knowledge，必须改口报 regularization effect。", 6.15);
  note(s, "1 分钟。这一页是回答「怎么保证 gain 来自 external knowledge 而非正则化」的标准答案页。");
}

/* ---------- 2.9 下一步 ---------- */
{
  const s = slide();
  title(s, "下一步：Week 1 今天就能开始");
  lead(s, "canonical 无损性门禁不需要一张图像，所以图像侧的数据物流阻塞不阻塞方法验证。");

  table(s, [
    ["任务", "需要", "现在能做"],
    ["canonicalize_navsim.py + canonical planner 无损性门禁（PDMS > 90）", "零张图像", { text: "可以，今天", bold: true }],
    ["补 sensor blobs", "纯 IO / 带宽", "可以，后台挂着"],
    ["RAP 93.8 复现", "完整 sensor blobs", "阻塞"],
    ["Stage 1 / 2 / Final EPDMS", "v2.2 evaluator + navhard_two_stage", "阻塞"],
  ], { tbl: { y: 2.3, colW: [6.4, 3.3, 2.393], rowH: 0.44 }, fontSize: 11.5 });

  body(s, [
    "Week 2 的 S1 − S0 是整个提案的第一道生死门 —— 如果 target-only recovery 本身没有增益，后面三格矩阵都不用跑",
    { text: "红线：不能说「Week 1 的 RAP 复现已经在手」。手上是评测链路 + 受控装置 + CV 锚点复现；93.8 没做到（当前 75.8，残缺子集）。", bold: true },
  ], { y: 4.6, h: 1.4, fontSize: 12.5 });

  note(s, "1 分钟。收尾。念完最后一句就停，把时间交给讨论。");
}

/* ---------- 附录 1：红线 ---------- */
{
  const s = slide();
  title(s, "附录：表述红线", "备查，不讲");
  table(s, [
    ["不能说", "改成"],
    ["实验表明 alignment 是纯正则化 / 没有增益", "效应不可与 seed 噪声区分；协议禁止方向性结论"],
    ["FullAlign 方差更小说明它在正则化", "n = 3、未预注册 → 只作为下一轮的预注册假设"],
    ["Week 1 的 RAP 复现已经在手", "评测链路 + 受控装置 + CV 锚点；93.8 没做到"],
    ["21% 证明数据集图像稀缺", "本地可用性（data_bug），只作排期事实"],
    ["197 个 DAC 失败 = recovery 失败", "navtest 是 nominal 起点，只是同一类几何约束"],
    ["CV 能过而 planner 全败", "19 < 独立预期 80，叙事不成立；改用「铁核对 CV 也难 4.2 倍」"],
    ["铁核偏向左转", "TURN_LEFT 占比与失败率均为平均水平；左倾来自被标成直行、却要向左大幅横移的场景"],
    ["波士顿街道窄所以难", "map_location 只能说 geography / domain bundle"],
    ["图像三项无差异 = 视觉不是问题", "只说明这三个全局统计量解释不了失败"],
    ["新加坡左侧通行导致左倾", "sg 占比 .158 → .140（0.89×），该链不成立"],
    ["首个多数据集 planning / structured-to-vision transfer", "已被 UniPlanner / TerraTransfer 占用"],
    ["sensor-agnostic", "用 source-sensor-free；canonicalization 只负责 common interface"],
    ["DINO backbone lr = 2e-5", "backbone 是冻结的（img_backbone 全部 requires_grad = False）"],
    ["alignment 搬动了 1,860 个 DAC 场景，是净破坏", "离散零分布显示 null 搬动得一样多甚至更多；cross |dom| 只有 null 的 0.35–0.77 倍 → 仍是「不可与 seed 噪声区分」"],
    ["alignment 没触及那 242 个铁核，说明它没用", "400 steps 下什么都没触及它们；且该集合按观测构造（含全部 3 个 FullAlign run），不能外推到全预算"],
  ], { tbl: { y: 1.5, colW: [5.0, 7.093], rowH: 0.26 }, fontSize: 9.5 });
  note(s, "不讲。被问到时翻本页。");
}

/* ---------- 附录 2：必须主动说 + 预判问答 ---------- */
{
  const s = slide();
  title(s, "附录：必须主动说的五句 · 高频问答", "备查，不讲");

  s.addText("必须主动说的五句", { x: ML, y: 1.5, w: 5.7, h: 0.3, fontFace: F, fontSize: 13, bold: true, color: K, margin: 0 });
  body(s, [
    "under-trained（75.8 vs 93.8），铁核有一部分会在全量训练后消失",
    "navtest 是 nominal 起点，横移失败 ≠ recovery 失败",
    "CV 的一致性有机械成分（直行策略必然在弯道出界）",
    "MDE 在 v1 PDMS 上测，EPDMS 方差方向未知",
    "画像是 descriptive / exploratory；图像阈值取自 navtest 自身，是对 spec §5 的有意偏离",
  ], { x: ML, y: 1.85, w: 5.7, h: 2.6, fontSize: 11 });

  s.addText("三个不能碰的动作", { x: ML, y: 4.5, w: 5.7, h: 0.3, fontFace: F, fontSize: 13, bold: true, color: K, margin: 0 });
  body(s, [
    "把 FullAlign 赢 / NoAlign 赢的场景集合拿去配元数据画像（触犯 forbidden_use 第 1 条）",
    "引用 TURN_RIGHT 那一行（n 仅 17）",
    "引用 Pittsburgh「> 20 辆」那一格（占该城 1.1%）",
  ], { x: ML, y: 4.85, w: 5.7, h: 1.6, fontSize: 11 });

  s.addText("高频问答", { x: ML + 6.4, y: 1.5, w: 5.7, h: 0.3, fontFace: F, fontSize: 13, bold: true, color: K, margin: 0 });
  body(s, [
    { text: "你的 null 否定了 DriveWeave 吗？", bold: true },
    "那是 no-result 不是 null（gate 两条全 FAILED）；即便当真，它测的是空载的桥，理论本就预测没增益；正因如此放进 A6 而非主线",
    { text: "你有没有看过 alignment 在哪类场景上起作用？", bold: true },
    "看了，而且这正是不能回答的那个问题。seed 0 的 DAC 是 rescue 254 / regression 107，seed 2 是 238 / 617 —— 同一份数据、同一初始化，方向反了",
    { text: "242 个场景会不会只是某几段路？", bold: true },
    "64 个 log，top-3 仅 17.4%，未触发混杂标记；另两个子集 top-3 达 45%，它们的画像已作废",
    { text: "为什么不直接把 nuScenes 标注渲成 perspective raster？", bold: true },
    "raster 无法表达「不可观测」——「没标注」和「没红灯」渲成同一张图，planner 会把缺失当 negative evidence 学。这是 A8 的 UNKNOWN ≠ NONE",
    { text: "效应这么小，是不是根本没接上？", bold: true },
    "三 seed 逐行验证全部 passed，且它改变了 86.4% 场景的输出。bug 已排除；限制是 backbone 冻结 —— 梯度到不了 DINOv3（见第 9 页）",
    { text: "alignment 搬动了 1,860 个 DAC 场景，这不是净破坏吗？", bold: true },
    "补上同条件离散零分布后，null 搬动得一样多甚至更多；cross |dom| 只有 null 的 0.35–0.77 倍（见第 10 页）",
    { text: "HEAT 已证明多数据集会掉点，你凭什么不掉？", bold: true },
    "不假设不掉。O3 写在提案里，成功判据分四级，胜利条件是持平不是超越",
  ], { x: ML + 6.4, y: 1.85, w: 5.7, h: 4.9, fontSize: 8.5 });

  note(s, "不讲。Q&A 时按题翻。");
}

/* ---------- 附录 3：六条改进启示 ---------- */
{
  const s = slide();
  title(s, "附录：对改进 Alignment 的六条启示", "按杠杆排序 · 备查，不讲");

  table(s, [
    ["#", "启示", "具体动作"],
    ["①", "先修测量", "MDE 4.56 分 > RAP 自己的 4.4 分效应 → seed 数 ≥ 4；离散噪声地板已补（1 小时、零 GPU）"],
    ["②", "评测目标错了", "9,683 六次全对、242 六次全错 → 争议区仅 2,221（18.3%），aggregate PDMS 稀释约 5 倍。分层须由独立参照 run 集定义，否则循环"],
    ["③", "冻结 backbone 是硬天花板", "二选一并说清：(a) 部分解冻（末 N block / LoRA），backbone lr 当一等超参；(b) 承认它是 BEV 聚合正则项，停止称其为 representation transfer。← DriveWeave A5 已列为消融轴"],
    ["④", "单点 λ 不构成检验", "次噪声效应正是极小权重应有的表现 → 最小方案 {0.002, 0.02, 0.2} × ≥ 4 seeds；跨两个数量级仍不动才是发现"],
    ["⑤", "损失形式与失败模式不匹配", "现状 = 拍平 MSE(λ=0.002) + 全局域对抗(λ=0.1)，都不保留空间结构；而 81% 残余失败是 drivable-area 几何。→ 在保留空间结构的 BEV 特征图上对齐（或按可行驶区加权），与朴素 MSE 同协议对照。旁证 TerraTransfer Table 6（0.319 / 0.307 / 0.490）"],
    ["⑥", "paired-only 是零载配置", "重设计要么加回 raster-only 增强（载荷），要么换 source（DriveWeave）。再跑一次 paired-only 只配当 λ 扫描的对照组"],
  ], { tbl: { y: 1.5, colW: [0.5, 2.6, 8.993], rowH: 0.55 }, fontSize: 9.5 });

  s.addText("一句话：当前配置下的 alignment，是一个作用在「冻结 DINO 特征的 BEV 聚合」上的、权重 0.002 的弱正则项，被放在一个 80% 饱和的指标上评测，而评测的检出下限高于它要追的效应量。这次没测到它的「价值」；测到的是它的「作用范围被结构性限制」和「评测被系统性欠功率」—— 两件都能直接动手修。", {
    x: ML, y: 5.5, w: CW, h: 1.0, fontFace: F, fontSize: 11.5, bold: true, color: K, margin: 0, lineSpacingMultiple: 1.2,
  });
  note(s, "不讲。被问「那你下一步打算怎么改 alignment」时翻本页，从 ① 往下念。");
}

/* ---------- 附录 4：§3.6 其余备料 ---------- */
{
  const s = slide();
  title(s, "附录：它没解决什么 · churn 的两种解释", "备查，不讲");

  s.addText("它没解决什么 / 带来了什么", { x: ML, y: 1.5, w: 5.7, h: 0.3, fontFace: F, fontSize: 13, bold: true, color: K, margin: 0 });
  body(s, [
    "没触及铁核：242 个场景的定义就是六次全败，含全部 3 个 FullAlign run → 该集合零 rescue（按观测构造）。但 400 steps 下什么都没触及它们，不能外推到全预算",
    "它留下的失败模式已刻画：197/242 驶出可行驶区、车更少、GO_STRAIGHT 内向左横移 28.5% → 72.7%、图像三项与全体无差异 → 几何 / 行为类失败，发生在无可测量感知缺陷的场景里",
    "代价侧已量化，结论是「也在噪声内」：DAC 合并 930 regression / 932 rescue 曾看似是代价，补上同条件离散零分布后，五个离散指标的 cross |dominance| 与 churn 全部不超过 null",
    "不能用但该预注册：FullAlign 在 7 个指标里 6 个 seed 方差更小（n = 3、未预注册）",
  ], { x: ML, y: 1.85, w: 5.7, h: 4.4, fontSize: 10 });

  s.addText("churn 几乎持平（DAC 98%）的两种解释", { x: ML + 6.4, y: 1.5, w: 5.7, h: 0.3, fontFace: F, fontSize: 13, bold: true, color: K, margin: 0 });
  body(s, [
    "cross 对共享 init / batch 顺序 / augmentation RNG，只差 alignment 梯度；null 对三者全不同。照理 cross churn 应远低于 null，实际几乎持平",
    { text: "(a) 边缘场景饱和", bold: true },
    "约 7% 的场景本就卡在 DAC 判定边界，任何扰动都翻它们。与「争议区仅 18.3%」的结构一致，更可能",
    { text: "(b) alignment 扰动强度确实接近全量重随机", bold: true },
    "数据分不开这两种解释",
    { text: "分开它们的办法（约 20 行代码，明天用不上）", bold: true },
    "比较「alignment 翻的场景集」与「换 seed 翻的场景集」的重合度：高度重合 → (a)；基本不相交 → (b)",
  ], { x: ML + 6.4, y: 1.85, w: 5.7, h: 4.4, fontSize: 10 });

  foot(s, "产物：summary/discrete_noise_floor.json　|　figs/fig_e_discrete_noise_floor.png　|　脚本 scripts/alignment/discrete_noise_floor.py", 6.4, { h: 0.32 });
  note(s, "不讲。被问「你把 1,860 个场景搬动了，这不是破坏吗」或「churn 为什么没降下来」时翻本页。");
}

pres.writeFile({ fileName: "RAP_DriveWeave_0826.pptx" }).then((f) => console.log("written:", f));
