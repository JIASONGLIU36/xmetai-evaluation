# 项目技能（Skills）

本目录包含项目中可复用的自动化工作流和评估工具。

---

## 可用 Skills

### 📊 XMetAI 模型评估

**Skill**: `xmetai-evaluation`

**描述**: 气象模型评测结果的分析与出图报告

**覆盖的评测能力**:

| 能力（流程名） | 覆盖 | 状态 |
|---|---|---|
| `weather_field_scores` | 确定性连续场：RMSE / ACC / 预报活跃度 / 纬向谱 | ⚠️ 能力与渲染器在，格式契约骨架已删（单模型报告走 `generate_report.py`） |
| TS 系列（`weather_ts_det` / `weather_ts_ens` / `fdp_precip_ts` …） | 站点降水分类检验：TS / POD / FAR / 频率偏差 | ✅ 骨架分确定性 / 集合两份 |
| RMSE 批次家族（`weather_rmse_single_<模型>` / `weather_rmse_ens_<模型>`，外加谱检验补充报告） | 格点 RMSE / ACC / FA / 纬向谱，**多模型横向对比** | ✅ 三份骨架 + 三个渲染器都做了（`generate_det_report.py` / `_ens_` / `_wave_`） |
| 暴雨过程个例（`rainstorm_ts_single_<模型>`） | 2025 年 43 个暴雨过程 × **双口径**（逐日 24h / 过程累积）的站点 TS | ✅ 骨架 + 渲染器都做了（`generate_rainstorm_report.py`）。**流水线同为 `weather_ts_det`，但产物形状不同，走独立入口** |
| 其余流程（CRPS、FSS、概率评分、台风…） | — | ⛔ 报告模板还没做，调用时会明确报错 |

**能力身份自动识别**：给一个评测产物目录，skill 从 `manifest.json` 的
`resolved_config.pipeline` 自己认这是哪条链路再派发，**用户不用记能力名，
也不要靠目录名判断**（配置名与流程名本来就不绑定）。

**使用方法**——说一句话就行，**命令由智能体执行，你不用自己跑 python**：

> "帮我对 `evaluation_results/weather_ts_single_fgvp` 这个评测结果生成报告"

智能体跑的就是这一条：

```bash
python skills/xmetai-evaluation/scripts/generate_report.py <产物目录> --result reports/<名字>
```

它从 `manifest.json` 自己认出能力（TS 系列 / 连续场），派发到对应模板，出图并写 `REPORT.md`。
认不出、或认出来但模板还没做时**明确报错**，不会静默套用另一个模板。
更多选项见 `xmetai-evaluation/SKILL.md`。

**文档**: [`xmetai-evaluation/SKILL.md`](./xmetai-evaluation/SKILL.md)

---

## 快速开始

### 1. 注册 Skill

Claude Code 认的是**目录形式**：`.claude/skills/<name>/SKILL.md`。
本仓库用**复制**（不是符号链接——Windows 建符号链接要管理员或开发者模式）：

```bash
rm -rf .claude/skills/xmetai-evaluation
mkdir -p .claude/skills/xmetai-evaluation
cp -r skills/xmetai-evaluation/. .claude/skills/xmetai-evaluation/
diff -rq skills/xmetai-evaluation/ .claude/skills/xmetai-evaluation/   # 应当无输出
```

> **改完源文件要重新复制**。`skills/` 下的是源，`.claude/skills/` 下的是副本，
> 两边不会自动同步。
>
> **先 `rm -rf` 再复制，不要只 `cp -r`**：`cp -r` 只增不减，**源里删掉的文件
> 在副本里会一直留着**。这不是假设——2026-09-21 同步时实测副本的
> `assets/templates/` 里还躺着两份早已作废的 `README.md` 与 `ts.md`，
> 而源目录里那 7 份 `weather_*.md` 一份都没有。没装 `rsync` 就照上面这三条来，
> 最后那条 `diff -rq` 是验收动作。

### 2. 使用 Skill

在 Claude Code 里直接说需求即可（例如"这是连续场评测的结果目录，出个报告"），
skill 会按 `description` 里的触发词被自动选中；也可以用 `/xmetai-evaluation` 显式调用。

---

## 目录结构

```
skills/
├── README.md                    # 本文件（Skills 索引）
└── xmetai-evaluation/           # 模型评估 skill
    ├── SKILL.md                 # Skill 主定义：什么时候用、怎么用、硬性规则
    ├── scripts/
    │   ├── generate_report.py   # 产物目录 → 图表 + REPORT.md（认能力 + 派发）
    │   ├── generate_det_report.py    # 批次归档（单成员）→ 多模型对比图 + REPORT.md
    │   ├── generate_ens_report.py    # 批次归档（集合）→ 同上，另加 CRPS/Spread
    │   ├── generate_wave_report.py   # 批次归档（谱检验补充）→ 同上，另加球谐带
    │   ├── rainstorm_summary.py      # 43 个过程 × 2 口径 × N 模型 → 挑过程总表（CSV + 宽表）
    │   └── generate_rainstorm_report.py  # 暴雨过程个例产物根 → 多模型对比报告（按过程分节）
    ├── assets/
    │   └── templates/           # 报告格式契约（按能力家族一份，名字按 config 命名）
    │       ├── weather_ts_single.md    # 确定性降水检验：三份填空骨架（单模型 / 单基准 / 多模型）
    │       ├── weather_ts_ens.md       # 集合降水检验：同上三份，另加一节概率评分（AROC/BSS）
    │       ├── weather_rmse_single.md  # RMSE 批次家族（单成员）：一份骨架
    │       ├── weather_rmse_ens.md     # RMSE 批次家族（集合）：CRPS / Spread / Spread÷RMSE
    │       ├── weather_rmse_wave.md    # 纬向 FFT + 球谐带功率谱补充分析：一份骨架
    │       ├── weather_typhoon_single.md  # 台风路径/强度（确定性）
    │       ├── weather_typhoon_ens.md     # 台风路径/强度（集合口径 + 离散度）
    │       └── rainstorm_case.md          # 暴雨过程个例：双口径 × 多模型，按过程分节
    ├── references/              # 参考文档（口径与判读规则）
    │   ├── evaluation-metrics.md        # 指标词典：是什么（不判好坏）
    │   ├── precipitation-evaluation.md  # TS 系列：口径与产出契约
    │   ├── diagnostic-rules.md          # TS 系列判读规则
    │   ├── field-evaluation.md          # 连续场判读规则
    │   ├── rmse-batch-evaluation.md     # RMSE 批次家族（single/ens/wave）
    │   └── model-diff-analysis.md       # 模型差异归因：改动起没起作用
    └── agents/                  # 空。暂无子 agent
```

---

## 添加新 Skill

1. **创建 Skill 目录**：`mkdir -p skills/my-new-skill/references`
2. **创建 `SKILL.md`**（带 `name` / `description` frontmatter；`description` 写清**触发场景**，
   它是被自动选中的唯一依据）
3. **放参考文档**（`references/*.md`）
4. **更新本 README**（添加索引）
5. **注册到 `.claude/skills/`**（复制，见上）

---

## Skill 设计原则

参考 `xmetai-weather-modeling` 的标准结构：

### 1. SKILL.md 保持简洁
- 元数据和描述（`description` 决定什么时候被触发，要写触发词而不是功能清单）
- 快速入口
- 执行步骤
- Reference Routing（指向 `references/` 里的详细文档）

### 2. `scripts/` 只放入口，渲染逻辑放主仓库
`scripts/generate_report.py` 是 skill 的入口——解析参数、插 `sys.path`、
调 `xmetai_evaluation.visualization.report.build_report`、打印产物清单，**仅此而已**。

- **认能力、选模板、出图、写报告的逻辑全在主仓库**（`xmetai_evaluation/visualization/`）：
  那里能被 `pytest` 覆盖、能 import、能复用。入口脚本一旦开始有判断逻辑，
  就又变成一份会跟主仓库漂移的副本；
- 入口脚本存在的理由是**免安装**：`python -m xmetai_evaluation.visualization.report`
  要求包已经 `pip install -e .`，而它把仓库根插进 `sys.path`，直接就能跑；
- `references/*.md` 是代码的**散文镜像**，每篇都声明"以代码为准"。

### 3. `assets/templates/` 是报告格式契约
放的是**一份完整的报告骨架**——章节、表头、分隔行、口径说明全是真的，
只有数值位置用 `{…}` 空着。**不是说明文档，也不参与渲染**：
报告由主仓库的渲染器逐行拼出，骨架是给人看、给测试对的真值来源。
按**能力家族**分文件、不按流程分——确定性两条 TS 流程共用一份 `weather_ts_single.md`。
**文件名跟 config 名走**（`weather_rmse_single_<模型>` 对 `weather_rmse_single.md`、
`weather_rmse_ens_<模型>` 对 `weather_rmse_ens.md`、
`weather_ts_det_*` / `fdp_precip_ts_*` 对 `weather_ts_single.md`、
`weather_ts_ens_*` 对 `weather_ts_ens.md`），
两字母缩写（`det` / `ts` / `field`）和家族简称（`precip_ts` / `rmse_single`）都不再用。
**例外是 `weather_rmse_wave.md`**：它不由 config 名而来——**没有
`weather_rmse_wave_*` 这个 config**，那份报告吃的是 `weather_rmse_single_<模型>` /
`weather_rmse_ens_<模型>` 的同一批产物，只是换个口径重讲一遍谱。
**`weather_field_scores` 这一族现在没有骨架了**（`field_scores.md` 已删）；
注意它与 RMSE 批次家族**是同一个 pipeline**——前者出单模型报告，后者出多模型对比。
`tests/unit/visualization/test_ts_report.py` 会实跑一遍逐项比对：
改了渲染器的章节结构就必须同步改骨架，否则测试红。
逐字扫描只能钉住**不含占位符**的行，所以另有一组「编号与排版」用例
补上含 `{…}` 的那部分（图注/表题编号连续、每节小结带数字、每张图都被安排章节）。

### 4. 文档放在 references/
详细的使用说明、口径、诊断规则、与代码函数的对应关系。

### 5. 一个 Skill = 一个领域，能力靠路由区分
- `xmetai-evaluation`：**所有**评测相关能力，不要为每条流程建一个 skill；
- 新增能力时扩 `description` 的触发词 + 加一篇 `references/*.md`，
  再加一条路由判据（见 `xmetai_evaluation/visualization/report.py` 的
  `IMPLEMENTED` / `_FALLBACK_RULES`）。

---

## 维护记录

- **2026-09-09**: 初始版本
  - 创建 xmetai-evaluation skill
  - 实现降水评估功能
  - 完善文档体系
- **2026-09-11**: 加入连续场能力与自动路由
  - 新增 `references/field-evaluation.md`（`weather_field_scores` 判读规则）
  - SKILL.md 增加"能力路由"一节与两条硬性规则（`n_valid` 语义、`unit` 随数据源走）
  - 本 README 修正：删掉失效的 `xmetai-evaluation/README.md` 链接与旧的斜杠命令写法，
    目录结构改成实况，设计原则 2 改为"代码在主仓库"的实况描述
- **2026-09-11（下午）**: 接通 TS 路由 + 标准化 skill 库
  - `visualization/report.py` 按**报告家族**分派（`field` / `ts`）；
    `weather_ts_det` / `weather_ts_ens` / `fdp_precip_ts` 读
    `diagnostics/categorical_wide.csv`；报告标题默认取 `run_id`
    ——原先取 `forecast_source`，那是 reader 类型（`"fuxi"`），会张冠李戴
  - 新增 `scripts/generate_report.py`（skill 唯一入口，免 `pip install`）
    与 `assets/templates/`（`README.md` + `ts.md` 格式契约）
  - 新增 `tests/unit/visualization/test_ts_report.py`，其中一组用例实跑比对章节序列
  - 本 README：设计原则 2 从"`scripts/` 留空"改成"只放入口"，目录结构同步
- **2026-09-11（晚）**: TS 报告加「跨时效分布（箱线图）」+ 接通多模型对比
  - 渲染器新增第四节（箱线图 + 分位数表，同一个 `box_stats()` 出数）；
    `plot_metric_boxplot` 按 Tukey 几何画，口径由 `matplotlib.cbook.boxplot_stats` 当神谕钉住
  - 入口脚本新增 `--compare "名字=CSV路径"`（可重复）：对比模型是**裸结果表**，
    不是产物目录——`ts_multi_{fgvp,fuxi,aifs}` 那几个目录没有 `manifest.json`
  - `assets/templates/ts.md` 新增真正的骨架 C（多模型）；骨架 A/B 补第四节并重排章节号，
    至此三个分支各有端到端用例
  - 本 README 与 SKILL.md / references 同步：删掉"骨架 C 没有测试覆盖""多模型要手工调
    `create_report`""只有长表就合成宽表"三处已失效的说法
- **2026-09-11（深夜）**: 「对比其他模型」变成一句话
  - `evaluation_results/` 规整成**一个模型一个目录**（目录名 = config `name` = `run_id`，
    形如 `weather_ts_single_fgvp`），宽表统一落在 `diagnostics/categorical_wide.csv`；
    老式的 `ts_multi_*` 目录已迁移，被取代的挪进 `_archive/`（归档不删）
  - SKILL.md 工作流新增"第 4 步 · 用户要对比其他模型时"：**同级目录里有宽表的都是候选**，
    自己找齐、核对口径、**把清单报给用户确认**再跑——不加 `--compare-all` 之类的新开关，
    `--compare` 已经能表达一切，缺的只是"去哪儿找"

- **2026-09-11（更深）**: TS 报告改写成"论文实验章节"
  - 报告从"表格 + 末尾一堆裸 PNG"改成**每节自成一段实验叙述**：
    引导句 → 图表 → `图 N：图注`（图下方）/ `表 N：表题`（表上方）→ `**小结**：…`；
    **末尾的「图表」索引节整个取消**（图都就地带着图注出现，再列一遍是重复）
  - 新增第四节「误差形态与偏差结构」，POD / FAR / BIAS / 性能图四张图终于有了自己的章节
    （原先只在摘要里提一句）；章节链变成 二/三/四/五 +（可选）六，
    二～五在单模型/单基准/多模型三个分支里**逐字相同**
  - `ts_report.py` 新增五个 `analysis_*` 函数（分级/时效/误差形态/稳定性/对比），
    **只吃已算好的统计量，不重新读表、不重新实现指标**；无对比对象时不出对比段。
    顺带修掉两个自相矛盾：并列（差距 < 0.005）不再报领先者、
    名次改用**竞赛排名**（原先 `sorted().index()` 让并列的两家按字典顺序排，
    表里写 `3/3` 而正文算出第 2）
  - 契约补强：骨架的逐字扫描**抓不到含 `{…}` 的行**，而图注/表题/小结全在那种行里，
    所以新增一组「编号与排版」用例——`图 N：`/`表 N：` 连续不跳号、
    每节小结得有数字（防套话）、`create_report` 的每张图都必须被安排章节
  - 新增 `references/precipitation-evaluation.md` **§8 待办：空间分布图**——
    用户提的"卫星图"本轮不做，并查清了**不是缺绘图代码、是逐点场在
    `metrics/categorical.py:103-118` 就被展平求和丢掉了**，长表也没有 lat/lon 列；
    要画地图得先加"保留逐点场的 metric + 写场的 writer"再重跑，写进文档留作待办

- **2026-09-15**: 模板文件按 config 命名 + 新增集合预报骨架
  - `assets/templates/` 改名：`det.md`→`rmse_single.md`、`ts.md`→`precip_ts.md`、
    `field.md`→`field_scores.md`，新增 `rmse_ens.md`（集合口径：CRPS / Spread /
    Spread÷RMSE，对应 `weather_rmse_<模型>_ens`）。全仓库引用同步
    （本 README 目录结构、SKILL.md 路由表与参考文档、`ts_report.py` 的字节契约注释）
  - 顺手修掉 `rmse_single.md` 里两处与事实不符的旧说法：章节号写的是
    `一、二、…十一`（实际骨架是 `# 1.`…`# 11.`）、以及声称有
    `test_det.py` 守着（`tests/` 是空的，README 的「怎么被保证」才是实况）
  - `rmse_ens.md` **不含球谐带功率**：`vfc/metrics/spectrum.py` 只有纬向 FFT 谱，
    球谐停在「升级备忘」，收进来就是一条永远渲染不出来的章节；谱只保留纬向谱
  - 该骨架的渲染器（`visualization/ens_report.py` + `scripts/generate_ens_report.py`）
    **还没做**，模板索引里如实标为「还没做，调用时明确报错」

- **2026-09-15（下午）**: 模板全面改成 config 全名 + 新增谱检验骨架
  - `assets/templates/` 弃用家族简称：`precip_ts.md`→`weather_ts_single.md`、
    `rmse_single.md`→`weather_rmse_single.md`、`rmse_ens.md`→`weather_rmse_ens.md`；
    新建 `weather_ts_ens.md`（当时是空文件，当天下午写完，见下一条）
  - 新增 `weather_rmse_wave.md`——`weather_rmse_wave_<模型>` 批量归档的
    **纬向 FFT + 球谐带功率谱补充分析**报告骨架。章节 `# 1.`…`# 7.` + 附录、
    表不编号、图 `图 1：`…`图 7：` 连续；原件是语雀富文本导出，
    落地时去掉 `<font>` 彩色标签、语雀 CDN 图链换成 `![名字](名字.png)`，
    并按编号契约给每张图补了图注（原件一张图注都没有）。
    **渲染器还没做**
  - `field_scores.md` 与 `assets/templates/README.md` 同日删除；模板索引的职责
    收归 `xmetai-evaluation/SKILL.md` 的「参考文档」一节
  - 本文件的目录结构与「模板文件按 config 命名」一节、`SKILL.md` 的路由表与
    参考文档、三份 RMSE 骨架头部的互相引用，全部按新名字同步

- **2026-09-15（傍晚）**: `weather_ts_ens.md` 写完，TS 骨架正式分成两份
  - 集合那份＝确定性那份的二～五节**逐字相同**，加两处：头部多一行「集合成员数」
    与**集合口径声明**（TS/POD/FAR/漏报率/BIAS 都算在成员平均场上，不是逐成员
    评分再平均）；并**新增第六节「概率评分（AROC / BSS）」**，后面章节号整体后移
    一位（A 到八、B/C 到九）
  - AROC/BSS 一节的口径按 `core/categorical_ref.py` 的实际实现写，不是照抄指标名：
    `BS_ref` **只有给了外部气候概率参考的 6h 窗才有值**、BSS=1−BS/BS_ref、
    AROC 是**逐成员插值后**按超越式阈值算、每个时效末尾还有一行 `grade="AVG"`
    的**可算阈值等权平均**（不是按样本数加权）
  - **这一节目前没有代码支撑，骨架头部如实标了两处**：`run_ts_ens` 把 `window_aroc`
    置空（该能力不产 AROC/BSS 表）、`weather_ts_ens_prob` 不在 `report.py` 的
    `IMPLEMENTED` 里且 `README.md` 标着 `[BLOCKED]`。数据表本身是有的
    （`aroc_bss_<name>.csv`，11 列）
  - `weather_ts_single.md` 头部从"覆盖三条流程、三条共用这一份"改成
    **只管确定性两条**，并指向集合那份；`SKILL.md` 路由表拆成两行、
    "三条 TS 流程共用一套模板"那句改写成"同一个渲染器、两份骨架"

- **2026-09-15（晚）**: 清掉删除文件留下的失效引用
  - `field_scores.md` 被删之后 `SKILL.md` 还有三处指着它：路由表那格改成
    「能力与渲染器都在，但格式契约已删」；连续场那节「'结构固定'是被测试守着的」
    改成**明说真值来源没了**（别再假装测试还钉得住）；参考文档节删掉那一行
  - 顺带查出 `assets/templates/README.md` **也是同日被删的**、参考文档节还列着它，
    一并删掉，换成一句「`assets/templates/` 现在只剩五份，连续场的骨架与
    模板目录的 README 都已删」
  - 两份 RMSE 骨架头部「与 `weather_ts_single.md` / `field_scores.md` 的差别」
    去掉后半段；本文件设计原则 3 删掉 `weather_field_scores` 对
    `weather_field_scores.md` 这个已失效的举例，`weather_ts_*` 那条例举按
    single / ens 拆开
  - `weather_rmse_ens.md` 里「本骨架不含球谐带功率」那段改成**指向
    `weather_rmse_wave.md`**：球谐带那两节归谱检验骨架
    （对应 5.1 分频 / 5.2 分变量 / 5.3 配对检验），不在这份重复
  - 历史条目里出现的 `ts.md` / `det.md` / `field_scores.md` **不动**——
    那是当时的实况记录，不是失效链接

- **2026-09-15（深夜）**: 规整 `references/`——补上 RMSE 批次家族那篇
  - 最大的洞是**三份 RMSE 骨架在 `references/` 里一个字都没有**：新增
    `rmse-batch-evaluation.md`，把 `visualization/det_report.py` 的归档形状、
    派生指标（相对 RMSE 的 `100` 是**几何均值基线**、FA 偏差取了绝对值方向就丢了、
    频谱对数 RMS 的底数）、时效分段 `(0,120]/ (120,240]/ (240,360]`、
    Wilcoxon + Holm，以及**跨批次比较必须一致的六个口径参数**写进去；
    并如实标明 `_ens` / `_wave` 的渲染器都还没做、球谐带没有数据来源
  - `evaluation-metrics.md` 从 788 行的杂糅改成**纯指标词典**：删掉「空间聚合与 FSS」
    「业务指标组合」「Case 1-4」「时效典型模式」这些教科书内容和与
    `diagnostic-rules.md` 重复的 BIAS 分级（**冲突一律以 `diagnostic-rules.md` 为准**），
    补上 CRPS / Spread / Spread÷RMSE / AROC / BS / BSS / base_rate / `AVG` 汇总行的定义
  - `diagnostic-rules.md` 新增 §6 概率评分（AROC / BSS），并**明说这节没有代码支撑**
    （`run_ts_ens` 把 `window_aroc` 置空、`weather_ts_ens_prob` 未实现）；
    后面两节顺延为 §7 / §8
  - `precipitation-evaluation.md` 补概率评分表契约（`aroc_bss_<name>.csv`）、
    「概率评分与 TS 口径完全不同」对照表、A/B/C 三支线的**图与表编号表**，
    并把 `weather_ts_ens` 多一节导致整体后移一位这件事写清
  - `field-evaluation.md` 加一句`field_scores.md` 已删、格式契约没有真值来源
    （口径与判读规则不受影响——它们对的是代码，不是骨架）
  - 顺带记下一处**代码与骨架对不上**：纬向谱 log-RMS 实现用 `log10`
    （`det_report.py:551`），`weather_rmse_wave.md` 的骨架文字写的是 `ln`，
    差 ×2.3026，**尚未与实现核对**，两处都标了「以代码为准」

- **2026-09-15（凌晨）**: SKILL.md 按标准 skill 模板重排 + 新增归因框架
  - **起因是核对真实结果目录时发现文档与实况对不上**：`outputs/results/` 下是
    **扁平布局**（`ts_*.csv` + `*_meta.json`），而 SKILL.md 一直描述的是
    `manifest.json` + `diagnostics/categorical_wide.csv`。`categorical_wide.csv`
    **在本仓库里没有任何流程会写**（grep 只有读者没有写者）——今天那份
    `report/weather_ts_single_fgvp_fuxi_aifs/` 是手工 staging 才跑通的
  - SKILL.md 重排成标准形态并**只增不减**：`干什么/不干什么` → `触发场景` →
    **`输入：结果目录`**（扁平布局 + `_meta.json` 逐键说明 + TS 主表契约）→
    **`归类`**（四级判据 + 归类对照表 + 报清单）→ **`工作流`六步** →
    `能力路由` → `输出契约` → `硬性规则`（新增 13/14/15 三条）→
    `常见情形` → `参考文档` → `维护说明`
  - 工作流从原来的"第 1 步跑命令"改成**归类 → 定主模型 → 核对口径 → 跑 →
    转述 → 失败直说**六步：主模型可由用户点名（决定"本模型"的全部说法
    与差值图纵轴方向），未点名则按「目录名标记 → `<capability>.json` → 反问」推断
  - 新增 `references/model-diff-analysis.md`——**"改的这个模块起没起作用"**
    的判断框架：先把改动翻译成预期特征（六类改动的指纹表）→ 在结果里找特征 →
    做**波及面检查**（未改动变量/量级/时效有没有一起动）→ 四条判定规则
    （方向对 / 不是噪声 / **有代价评估** / 跨时效一致）→ 症状词典
    （八种数字形态反推成因，含"单位不匹配长得像技巧超好"那个真坑）
  - 如实标注四处**已知缺口**（SKILL.md 维护说明节）：主模型入口认不得扁平布局、
    `_ens`/`_wave` 渲染器没做、AROC/BSS 取数绘图没做、球谐带不存在
  - 本 README 的能力状态表同步：`weather_field_scores` 从 ✅ 改 ⚠️（骨架已删）、
    补 RMSE 批次家族一行

- **2026-09-16**: 补上 `weather_rmse_ens` 的渲染器，RMSE 三条支线都能跑了
  - 新增 `visualization/ens_report.py` + `scripts/generate_ens_report.py`，
    照 `weather_rmse_wave` 的五段式结构写；`analyze_ens()` 第一句就是
    `det_report.analyze()`，所以归档形状、相对 RMSE、FA 偏差、配对检验全部复用，
    集合特有的只有 CRPS / Spread / Spread÷RMSE / ACC 四块
  - `det_report._pick_metric_file` 的候选名**加一个** `f"{stem}_{date}_ensmean.csv"`
    （排在最末，单成员行为不变），这一行让 `analyze()` 原样吃下集合归档，
    且拿到的正是骨架要求的**集合平均场**口径
  - **两条口径红线写进骨架引用注释**：集合对数用 **ln**、单成员用 **log10**（差 2.3026×）；
    逐日表只读归档自带的 `<口径>_<日期>.csv`，**不用 glob 兜底**——
    否则 `rmse_<日期>_member_000.csv`（单成员）会被当集合平均静默读进来
  - 同步纠正三处**早已过时**的说法：`_wave` 的渲染器其实早就做好了
    （`wave_report.py` + `generate_wave_report.py`），球谐带功率也早就有了
    （`vfc/metrics/spectrum.py:210` 一直在写 `spherical_bands_*.csv`，实测 1308 份）。
    `references/rmse-batch-evaluation.md` §0 的三行表格、SKILL.md 的已知缺口、
    本文件的目录树一并改过来
  - 实测数据现状：三份集合归档（FuXi 349 天 / AIFS 27 天 / FGVP 2 天）
    **三方共同日期为 0**，所以本轮只用 FuXi↔AIFS 的 27 天验证渲染器，
    正式三方报告等 FGVP 数据补齐

- **2026-09-21**: 三份 RMSE 骨架改版 + 文档按 `xmetai_evaluation/` 实况同步
  - **骨架改版**（`weather_rmse_single.md` / `_ens.md` / `_wave.md`）：
    编号契约第一次写明确——正文与必出附录 `图 N：`（`_single` 必出 1–4、
    `_ens` 必出 1–6、`_wave` 必出 1–7，其余按数据可得性出、**出不了就保留节位
    标「本批未出」，允许出现断号**），可选块另开 `图 L*` / `图 S*` 两个命名空间，
    **整块删掉不用重排任何编号**
  - 骨架新增三节：**变量 × 时效 RMSE 热力图**（各自除以本模型首时效的 RMSE
    倍数，单位无关才能同色标横比）、**谱比随时效**、**单起报谱曲线**；
    以及「附 L 分纬度带结果」「附 S 分季节结果」两个可选块
  - 渲染器接纬度带：`det_report.region_table()` 成为 det/ens/wave 共用的汇总，
    正文只留全球行、附 L 只留带行（**两类行混着 `pivot_table(mean)` 会静默
    抹平带间差异**）；并列第一按显示精度判、逐个列出，不再拿 `idxmin` 挑一个
    冒充当赢家
  - 新增图品类 `multi_model_rmse_heatmap` / `ensemble_rmse_heatmap` /
    `spectrum_curve_<变量>_<起报>`；**「谱比随时效」渲染器和骨架都留着但必然
    是本批未出**——产物里没有任何逐 lead × 逐波数的谱（长表那条是标量、
    `spectrum_by_init` 没有 lead 列）
  - **文档按仓库实况重写**：`references/rmse-batch-evaluation.md` 的 §1 归档形状
    与 §4 口径参数整节重写，SKILL.md 的归类判据 / RMSE 家族说明 / 已知缺口同步，
    三个入口脚本的 docstring 同步。作废的东西统一登记在文首那条 ⚠️ 里
  - **三处作废说法**（旧文档写于另一个仓库布局，本仓库里没有对应物）：
    `vfc/regr_ens.py` / `vfc/regr_summary.py` / `vfc/regr_pair.py` 全不存在
    （只有 `xmetai_evaluation/`）；`--summarize-det` / `--summarize-ens`
    两个开关不存在；「批次归档 = `summary.csv` + `batch_meta.json` +
    `<YYYYMMDD>/`、没有 `manifest.json`」不成立——产物就是标准评估产物目录
    （`scores.csv` + `manifest.json` + `diagnostics/`），`pipeline` 也正是
    `weather_field_scores`。RMSE 家族单开入口的原因是**多模型对比**，不是形状不同
  - **实拍发现、写进文档让下次不用重新发现**：
    (a) 长表契约 **24 列**（含 `group`），而现存归档的 `scores.csv` 只有 **23 列**、
    没有 `group` —— 球谐带 5 个频带的行因此认不出来，`wave_report` §4.1–4.3
    会标「本批未出」，**重跑评测即可**；
    (b) `manifest.json` 的 `artifacts` 存的是**写盘时的绝对路径**，
    产物目录拷走后即失效，别拿它定位文件；
    (c) `--model NAME=目录` 的 `NAME` 与目录名、与 `manifest.run_id`
    **三者可以全都不一样**（实测把 FuXi 的产物拷进 `..._fgvp` 目录，
    `run_id` 照样是 `..._fuxi`）

- **2026-09-22**: 七份骨架统一改版——标题、摘要两处（用户提的四条意见）
  - **标题改清楚**：七份骨架一律改成「`{类型}{模型数}模型{维度}评估报告`」，
    去掉原来的括号后缀（`（单成员）` / `（{成员数}成员）` / `（集合）`）。
    新标题：`确定性预报{模型数}模型综合评估报告`（rmse_single）、
    `集合预报{模型数}模型综合评估报告`（rmse_ens）、
    `确定性预报{模型数}模型波谱评估报告`（rmse_wave）、
    `确定性预报{模型数}模型台风评估报告` / `集合预报{模型数}模型台风评估报告`、
    `{模型名}降水评估报告`（ts_single）、`集合预报{模型数}模型降水评估报告`（ts_ens）。
    落地的最烂标题原本是 `XMETAI single（确定性版本）评估检验补充报告`，现在没了
  - **摘要不再分点**：`# 1. 结论摘要` 的正文由「逐条 `•`」改成**一整段连续文字**
    （`"".join(items)`）。**只改了「结论摘要」这一节**——`rmse_ens` 的
    4.1 分项结论 / 5.1 时效结论、`rmse_single` 的 §10 总述结论、台风的 §5 综合结论
    **都还是分点的**，用户没点名，没动
  - **「报告定位与衔接」并进摘要**：这一节整个删掉（TS 那份删的是名字不同的
    「评测对象与对比对象」），内容拆成三段进 `# 1. 结论摘要`——
    ①评估基本信息（评的是谁、多少天 / 多少记录、什么指标、集合口径），
    ②一整段结论，③固定口径声明（RMSE 家族的 FA 定义句、TS 的指标定义式）
  - **渲染器同步**：`ts_report.py`（含 docstring 的章节链）、`det_report.py`
    （`_conclusion_summary` 新增 `first/last/leads` 三个关键字参数）、
    `ens_report.py`（`_section_conclusion` 同上）、`wave_report.py`；
    `tests/unit/visualization/test_ts_report.py` 里两条钉死标题的断言
    （`test_title_uses_run_id_not_forecast_source` / `test_explicit_model_name_wins`）
    跟着改。**台风两份骨架没有渲染器**（全仓没有 typhoon 的 report 模块），只改了骨架
  - **集合成员数进了 reference**：产物**根本不记**这个数——长表与 `manifest.json`
    都只记「算出来的数」，`ens_report._unique_summary_value(archive, "n_members")`
    取不到列，所以报告里「成员数」那一列一直渲染成 **「—」**。
    `references/rmse-batch-evaluation.md` 的 §1.1 写明这件事、§4 加一行
    「集合成员数」并给出实拍取值：**FuXi 集合（`model_id=fuxi_ens`，
    reader `fuxi_ens_phys`）= 51 成员**（1 控制 + 50 扰动）。
    **换 reader 配置后必须回来手改这张表**
  - **顺带同步的文档**：`SKILL.md` 与 `references/precipitation-evaluation.md`
    里的 TS 章节链（少了「评测对象与对比对象」一节，摘要节多了「先交代口径」）
  - **`weather_ts_ens.md` 的骨架仍然领先于渲染器**：`weather_ts_ens` 走的是
    同一个 `ts` 渲染器（`report.py` 的能力路由），骨架里的「集合成员数」
    `{成员数}` 与「集合口径」那一句**渲染器从来不产**——这是本次之前就有的缺口，
    本次没有扩大也没有修

- **2026-09-22（下午）**: 实测订正一个区划数字——`references/rmse-batch-evaluation.md`
  里说热带 `n_valid` 是 **228960**，**是错的**。拿两份归档（`weather_rmse_ens_fuxi`
  的新旧两版）对，热带行实测都是 **231840**。算术也对得上：ERA5 0.25°（721×1440）
  纬圈是从 −90° 起、每 0.25° 一圈，**±20° 闭区间 = 161 圈**，161 × 1440 = 231840；
  228960 对不上任何一个整数圈数（228960 / 1440 = 159）。顺手把三个数都改成加粗，
  并补了一段说明：**这三个数就是「纬圈数 × 经圈数」，边界闭区间、两端带重叠**
  （±20° 那两条纬圈被热带和中高纬各算一次，所以三条带加起来比全球行大），
  **换网格或换边界这三个数就变，别当常数去比不同网格的归档**

- **2026-09-22（傍晚）**: `references/rmse-batch-evaluation.md` 补两条产物的"读法"
  - **逐变量指标矩阵是配置说了算，不是跑丢了**：用户拿
    `weather_rmse_ens_fuxi` 的归档问「是不是没有 acc」——实测 `acc` 有
    **167040 行、全是 `status=success`、空值 0 个**，只是 `VAR_METRICS` 只把 `acc`
    给了 `z500` 和 `msl`；`msl` 反过来不做 FA 与谱。**维护者确认两条都是有意为之**
    （2026-09-22），**不是漏配、不要再当缺口提**。§1.2 新增一张现行取舍表，
    写明「长表里没有的行 = 配置没让它算，先看配置」。
    另注：气候态**确实加载了 `u200`/`v200`**（`ws200` 按 u/v 合成），
    所以这个取舍**不是数据不够**，真要补是「改配置 + 重跑」，不涉及新输入
  - **`scores.csv` 是 24 列不是 23**：§1.2 的列清单漏了 `group`（球谐带 `1_4`
    这类组标签）。补上，并写明**早于该修复的归档只有 23 列**、怎么认（数表头 /
    看 `manifest.score_columns`）、以及**渲染器不依赖 `group` 的有无，
    老归档不用为了补这一列重跑**

- **2026-09-22（晚）**: 自比对演示跑出来的两个渲染器缺陷，均已修
  - **`--archive` 从来没生效过（抬头打印整个 `Archive` repr）**：`det_report` /
    `ens_report` / `wave_report` 三份 `build_*_report` 里都有
    `for archive in archives:`，**循环变量遮蔽了同名函数参数**（`--archive`
    传进来的那个字符串）。循环一跑完，`render*(..., archive=archive)` 拿到的
    是最后一个 `Archive` 对象，报告抬头就变成
    `**产物：Archive(name='伏羲', root=..., summary=<DataFrame>, dates=[348 个日期])**`。
    改法：三处循环变量改名 `item`，各留一行注释防止复发。
    **这个 bug 只有在抬头被注意到时才会暴露，数值一个都不受影响**
  - **并列时不报「并列」**：同一个归档喂两次（模型名 `fuxi1` / `伏羲`）后，
    报告里到处是「综合相对 RMSE 最好 fuxi1，最差 fuxi1」「ACC 最高 fuxi1，
    最低 fuxi1」这种自己跟自己比的话。根因是分节结论直接用了
    `idxmin()` / `idxmax()`，而**「附 L」早就在用一套并列判据**
    （`_best_names` / `_best_cell`，按**显示精度**判并列，digits=3）——
    同一份报告里两套口径。修法：
    - `_best_names` / `_best_cell` 加关键字参数 `mode`（默认 `"min"`，
      `"max"` 取最大），共享一处实现；
    - `ens_report` 新增 `_extreme(series, mode, digits, suffix)`，
      极值句统一走它（`suffix` 管 ACC 的 `%`、FA 的 ` pp`）；
    - 三处「更接近1」/「更优」表格列改用 `_best_cell`；
    - `_overall_notes` 的「口径分歧」改为按**集合**比（`rmse_best ⊆ crps_best`
      才不报），并列冠军里只要有一个不是 CRPS 冠军就算分歧。
    **只在自比对下才现形，但两个模型差距很小时同样会误报领先者**
  - 顺带清掉两处因上面改动而失效的局部变量（`best` / `worst`）
- **2026-09-22（深夜）**: §4 补 ACC 分变量分析 + 去掉「纯结论」小标题
  - **集合指标认 `valid_mask`**（`metrics/ensemble.py`）：CRPS 与 Spread-Error
    以前整段没读这个掩码，而执行层的 `restrict_to_latitude_band` 一个格点都不裁、
    只把带掩码叠进 `valid_mask`——于是全球标量被原样复制进每个纬度带行。
    新增 `_mask_field()`，分子分母一起退出统计。**这是分带口径的改动，不是优化**
  - **日期跨度写成集合式**（`det_report._date_span`）：`20250102，20251215`
    （全角逗号，不带括号，调用方已套括号）；22 处调用点一并生效
  - `ens_report` 新增 **§4.2 分变量ACC**：表头 ACC 列本就是单变量（`acc_variable`），
    这里按变量铺开 + `_acc_notes()` 判读句（逐变量极差、各变量优胜者是否一致、
    本批 ACC 的覆盖范围）。同时把 §4 表头与 §1 摘要的 ACC 句**点名变量**
    （`ACC（z500）`），不再让单变量数字冒充全变量结论
  - 原 `## 4.2 逐变量RMSE` 顺势变 **4.1**，两处「与正文第 4.2 节同源」的引用跟着改
  - **去掉三处「纯结论」小标题**（内容保留，只删标题）：`ens_report` 的
    「4.1 准确度与可靠性分项结论」「5.1 时效结论」、`det_report` 的
    「8.1 逐模型结论」；det 的「关键配对差异」补位成 8.1
  - 骨架同步：`weather_rmse_ens.md`（4.1/4.2 重排 + 新增 ACC 段）、
    `weather_rmse_single.md`（8.1 补位）
- **2026-09-22（续）**: 按需子集对比（「热带 春季 rmse 对比」这类请求）
  - `references/rmse-batch-evaluation.md` **新增 §6「按需子集对比」**
    （原 §6「不要做的事」顺延为 §7，全仓无 §6 交叉引用，改动安全）：
    口语词 → 长表列的词表、季节口径、三条不能破的口径铁律、一段可照抄的取数片段、
    补进报告的节位与形态
  - **两条季节口径当日与维护者确认**：季节按**起报时刻**（`init_time`）切，
    不按 `valid_time`——这套报告的契约是「共同日期 = 共同 `init_date`」，
    一组 `init_date` 必须干净地属于一个季节；DJF 按**气象冬季**归组
    （当年 12 月 + 次年 1、2 月），**跨年批次的两个冬季各只有一截**
    （实测 `20250102–20251215`：DJF(2024/25) 只有 1/2–2/28，DJF(2025/26) 只有
    12/1–12/15），报 DJF 要说清是**哪一个冬季**，不拼凑
  - `SKILL.md`：触发场景与「常见情形」各加一条路由指向 §6
  - 三份骨架的「附 S」与三个 `_season_block()` 的静态文案同步改写：从
    「等两条口径定下来」改成「口径已定 + 指路 §6」。**顺带修掉一处漂移**——
    `wave_report` 的附 S 与 `weather_rmse_wave.md` 不一致（前者写 `Best`、
    且漏了「谱比随季节变化明显」那一整段），现按骨架对齐
  - §6.4 的片段**在真实归档上跑过一遍**（`weather_rmse_ens_fuxi`），修掉三个
    只有跑起来才会现形的问题：`dtype=str` 读进来后 `value` 不转数值会抛
    TypeError；不转 `lead_h` 则 `unstack` 按字典序排；空 `region` 被读成 `NaN`
    会让「全球」静默变空集（需 `keep_default_na=False`）
  - 实测记下 DJF 的**陷阱**：`20250102–20251215` 的 DJF 是 73 天
    （58 天 DJF(2024/25) + 15 天 DJF(2025/26)，两截不相连），与 MAM/JJA/SON
    的 92/92/91 是一个量级，**光看天数发现不了**，报之前必须先看它由哪几个冬季组成

- **2026-09-22（续二）**: 把「热带 × MAM」切片实做进 `report/fuxi1_vs_fuxi/REPORT.md`
  - 拿 §6.4 的工序在两个模型槽上各跑一遍，合成一张 `| 变量 | fuxi1 | 伏羲 | 显著优胜 |`，
    替换掉「附 S」那段静态文案。**结论本身退化**——两个模型槽喂的是同一份
    `weather_rmse_ens_fuxi`，两侧逐位相同、配对检验必然「无显著差异」；已在表下点明
  - **发现归档切不动**：这份归档是修复前产物，**集合口径**的分带没生效——三个纬度带
    的 `n_valid` 完全相同（z500 均 95,518,080），热带与全球逐位相同；**确定性口径**
    才是对的（热带 21,329,280 vs 中高纬 37,226,880）。**两个口径必须分别验**，
    验一个就当整份归档没问题会踩空。本次改用确定性口径切并把这个换口径的事实在
    表下写明
  - `references/rmse-batch-evaluation.md` §6.4 据此补「**切之前先验归档能不能切**」
    一节（含识别代码与实测对照表），并记下 `metric_rows(region="")` **取不到全球行**
    （内部读表把空串吞成 `NaN`），要全球口径得把 `region` 整个省掉——与 §6.4 自己
    用 pandas 读表的 `region == ""` 写法**不一样，别混**
  - §6.5 补两条交付纪律：口径被迫换过必须在表下写明；**手改的 `REPORT.md` 是临时的**，
    下次 `generate_*_report.py` 整份重写会把它冲掉，含手补内容的报告要当面告知用户
  - 顺手修掉一条 2026-09-22（续）里没改净的旧措辞：DJF 不是「缺 12 月」，
    是**两个冬季各只有一截**

- **2026-09-22（续三）**: 三处报告格式改动（用户提的两条 + 一处新增）
  - **抬头「报告日期」改成生成日期**（`det_report.report_date()`，三份渲染器共用）：
    以前写的是 `bundle.dates[-1]`（数据末日期），与正文「覆盖 N 天（起，止）」
    重复，还把「这份报告是什么时候出的」这个只有抬头能承载的信息挤掉了。
    **代价：产物不再逐位可复现**——同一天跑两次一样，隔天跑就不同。
    七份骨架的 `{报告日期}` 占位符含义同步写明
  - **删掉「谱比随时效」一节**（`weather_rmse_single` 附 6 / `weather_rmse_ens` 附 8 /
    `weather_rmse_wave` §5.6）：产物里根本没有逐 lead × 逐波数的谱（长表那条是标量、
    `spectrum_by_init` 没有 lead 列），这一节**永远只能是「本批未出」**，
    留着是噪声。连带删掉三份渲染器的 `_spectrum_ratio_block()` 与
    `det_report.SPECTRUM_RATIO_ABSENT` 常量，后面的图号/节号顺延：
    单起报谱曲线 `_single` 图 7→6、`_ens` 图 9→8、`_wave` §5.7→§5.6 图 10→9
  - **新增「模型能力雷达图」**（三份 RMSE 报告都加，排在编号附录的最后）：
    一根轴一种**综合评分**，每个模型一个多边形，轴取
    `_single` 4 根（综合相对RMSE / ACC / FA偏差 / 频谱对数RMS）、
    `_ens` 6 根（多 CRPS 与离散度偏差）、`_wave` 5 根（多两根谱 logRMS 双口径）。
    逐变量 / 逐时效的细粒度指标**不进雷达图**——一根轴只画一个顶点，
    细粒度指标要么先汇总（那已经是另一个综合评分）、要么把轴撑爆
  - 实现分两处、**归一化只有一处实现**：`det_plots.radar_scores()`（纯函数，
    批内 min-max，方向由 `higher_is_better` 定）与
    `DetPlotter.plot_capability_radar()`（只负责画）；轴的定义与图注在各自的
    `capability_profile()` / `_capability_block()` 里。两个退化情形都写死并
    在正文点明：**某根轴全批打平 → 一律给 1.0**（除零会炸，且「大家一样」
    该画满格）、**某根轴有模型缺值 → 整根剔除**（雷达图没有断点，少一个顶点
    会把多边形拉歪，不能拿 0 顶替）。图注与读法段一并说明**这是批内相对分、
    不是绝对能力分**，而且**只有两个模型时必然是 1 和 0**
  - `_ens` 的 `_check_figure_contract` 新增 `always=(9,)` 参数：雷达图排在
    可选段之后、但**不依赖任何可选数据**，跟图 1–6 一样必出，靠 `required`
    那个前缀区间表达不了。三份骨架的编号契约同步写明「图 N 不允许缺席，
    缺了是渲染器坏了，不是本批未出」
  - 实拍验证：三份渲染器各喂同一份归档的两个模型槽跑通
    （`weather_rmse_ens_fuxi` / `weather_rmse_single_fuxi`），
    图号连续、抬头日期正确、雷达图落盘——**注意这是自比对，
    每根轴都并列、多边形两个模型重合，属预期的退化形状**
  - **ts / typhoon 四份骨架还没加**：typhoon 两份**没有渲染器**（全仓无 typhoon
    report 模块），加了只有骨架没图；`weather_ts_single.md` 的加节还要同时改
    `ts_report.py` 与 `tests/unit/visualization/test_ts_report.py`（后者逐字扫骨架）。
    轴方案已提，等维护者确认
- **2026-09-22（续四）**: ts / typhoon 四份骨架的雷达图补齐（维护者已确认三条口径：
  typhoon 破例并写明、路径误差单占一根轴、ts 连带改渲染器与测试）
  - **`weather_typhoon_single.md` / `weather_typhoon_ens.md`**：新增「模型能力雷达图」
    节（`# 5.` 图 3 / `# 6.` 图 4），轴全部**越低越好**：路径误差 MAE、沿轨 MAE、
    横轨 MAE、最大风速 MAE、最低气压 MAE（集合那份多一根路径离散度偏差，共 6 根）。
    **这是全仓唯一一处「骨架明写不做相对化、却仍出现相对分」的地方**——两处
    都在节内用引用块把边界写死：只回答「这批个例里谁强在哪」，不能与第 3、4 节的
    绝对量并列引用、更不能跨批次比；集合那份还点明风速轴是**逐成员口径**
  - **`ts_report.py` 新增 `capability_profile()` / `_section_capability()`**，
    轴取 4 根**综合评分**（跨等级或跨时效汇总过的能力分）：TS（全等级等权平均，越高
    越好）、强降水 TS（全批 ≥25mm 的**最小**一档，越高越好）、|BIAS−1|（全等级平均，
    越低越好）、跨时效 TS 的 IQR（全等级平均，越低越好）。**POD / FAR 明确不进**
    ——`1/TS = 1/POD + 1/(1−FAR) − 1` 是恒等式，三根同放是同一件事计三次分
  - **只在两个及以上模型时出现**（`CAPABILITY_MIN_MODELS = 2`）：一个多边形没有
    比较对象。所以骨架 A（单模型）没有这一节，B / C 里它是**第七节**，
    建议/口径顺延到八/九。真报告图号：A `图 1..8`、B `图 1..9`、C `图 1..10`
  - 出图放在 `build_ts_report()` 里、**延迟导入** `det_plots`：后者在模块层 import 了
    `precipitation_plots`，而后者又 import `ts_report`——放模块层会绕成环，
    `import ts_report` 直接失败。模型集合与第五节箱线图共用同一个 `models` dict
  - `weather_ts_single.md` / `weather_ts_ens.md` 两个分支各加那一节，
    新增用例 `test_capability_radar_only_when_there_is_something_to_compare`
    （单模型不出图、有对比才进第七节、POD/FAR 不在节内、四根轴与方向）
  - `weather_ts_ens.md` 额外写明**真报告号比骨架少 1**：骨架多算了尚未实现的
    概率评分节，而 `ts_report.py` 现在不出那一节，第六节补上后自动对齐
  - **本轮的 ts 渲染器改动尚未实跑**（手上没有 ts 产物目录），
    验证命令见下：`pytest tests/unit/visualization/test_ts_report.py`

- **2026-09-22（续五）**: TS 系列接通分纬度带（协议 → 执行 → 宽表 → 报告）
  - **起因**：用户要求「weather_ts 系列也要有像 RMSE 那样的分纬度带评估」。
    与 RMSE 那份最大的不同在**切分对象**：RMSE 切的是格点，TS 切的是站点，
    所以走的是 `station_valid_time` 协议、按**站点纬度整站归带**（读 `station_lat`，
    退而求其次 `lat`；都取不到直接 `AlignmentError`，**不静默回退成全球分**
    ——那会把全球分复制成每个带的分，比报错危险）
  - **协议层**：`grid_valid_time` 早就有的 `options["regions"]`，站点那条现在
    同样支持（`test_protocols.py` 新增三个用例，含闭区间与取不到纬度报错）
  - **执行层**：带掩码必须在 `narrow_batch` **之后**叠（它按变量重算 `valid_mask`，
    先叠会被冲掉）；`ensemble.py::_mask_field` 原本对 `valid_mask` 的形状卡得很死，
    而 `ensemble_reduction="none"` 的批次会多挂一条前导 `member` 轴
    ——现在容忍这一维（沿成员轴取「任一成员有效」），并在注释里写明为什么
  - **宽表**：`region` 列终于有值了（带名，全球行为空）——`references/precipitation-evaluation.md`
    里「降水检验里始终为空」那句据此订正
  - **报告**：末尾新增 **「附 L 分纬度带结果（可选）」**。正文只留全球行、附 L 只留
    带行（**两者永不合并**，各带平均再平均 ≠ 全球平均），图走独立编号 `图 L1`
    不动正文图号；表按**南界从低到高**排（按带名排中文是乱序），并列按显示精度
    判定并写明头名是否随时效翻转；没配分带的产物照样保留节位、写「本块本批未出」
    （骨架契约要求章节数**严格相等**，条件性出节会让契约无解）
  - **骨架**：`weather_ts_single.md` / `weather_ts_ens.md` 各三个分支同步（共 6 处）
  - **配置**：`weather_ts_single_fgvp.py` 启用中国四带（南方 / 长江中下游 / 华北 /
    东北），带名就用中文——**报告不翻译、不写死任何带名**，配置里叫什么就显示什么。
    `weather_ts_ens_fuxi.py` **本轮没动**（先把 single 跑通、看过分带结果再决定）
  - 验证：`pytest tests/unit/visualization/test_ts_report.py`（44 passed）与全量
    `pytest`（353 passed / 2 skipped）

- **2026-09-22（续六）**: `weather_ts_ens_fuxi.py` 也启用分纬度带，并订正 execution 注释
  - **配置**：`weather_ts_ens_fuxi.py` 启用与中国四带**同一套**带定义（续五里留的
    「先把 single 跑通再决定」到此收口）。`options` 是**两段共用**的，所以 24h 的 TS 段
    与 6h 的概率段都会出带行；带样本与全球样本共用 `build_batch` 缓存的那个批次
    （全场插值只做一次），代价是**计算量 ×5 而非读盘量 ×5**
  - **报告侧的已知缺口**：附 L 只汇总 24h 那段的 TS。概率段的带行照常落进
    `scores.csv` 与 `diagnostics/probability_wide.csv`（两个宽表的 `region` 都在透视
    键里），但报告不出——第六节（AROC / BSS）本来就没有渲染器，缺口写在
    `assets/templates/weather_ts_ens.md` 顶部
  - **注释订正（只动注释，`execution` 的键一个没碰）**：
    - `mode: processes` 原来的理由是「threads 并发读 nc 撞 HDF5 race」——那是
      `hdf5_guard`（commit `8e45685`）**之前**的说法。配置改进程在 `8b15a23`，两者
      中间隔了 11 个 commit，守卫已经把这条读路径串行化了。**结论不变、理由要换**：
      `hdf5_guard` 是一把全局锁，threads 下所有 netCDF 打开都排队，而本配置一个 run
      要开约 106 万个 nc，读盘正是大头，并发收益会被吃光；进程形态各读各的
    - `n_workers` 补上单块内存算式：读取集 = 采样时效 ∪ 预热时效，24h 段
      `{6,12,18,24}`×51 成员 = 204 场 ≈ 1.7 GB、6h 段 `{0,6,12,18,24}`×51 = 255 场
      ≈ 2.1 GB（单场 8.2 MB / 0.25° 全球 float64，锚点是 `plan.py` 里「15 天 × 51
      成员约 25GB」反推）。**n_workers × ~2 GB 即申请内存下限**：24 进程要 ~50 GB
  - **文档**：`README.md` 与 `references/precipitation-evaluation.md` §9 补上这份配置

- **2026-09-22（续七）**: `weather_rmse_wave` 降成**补充报告**：7 章 12 图 → 4 章 6 图
  - **起因**：用户问「两份报告是互补的吗」，逐章对完发现**约一半是重复的**——
    样本完整性、总体表、逐变量 RMSE、msl 量级检查、复算不足、ACC 风险、选型建议、
    附 L、附 S 全在主报告里，12 张图里有 5 张（热力图 / 单起报谱曲线 / 雷达图 /
    两张分纬度带）是**主报告附 5–7、附 L 的同一张图**。这份报告真正独有的只有
    **球谐带 + 纬向 FFT** 那条链。用户随即要求「简化骨架，变成一份补充」
  - **删掉的**：原 §3.1 排序随目标变化、原 §4 RMSE 与时效表现（含 §4.1）、
    原 §5.5 热力图 / §5.6 单起报谱曲线 / §5.7 雷达图、原 §6 异常与数据质量风险、
    原 §7 验收结论与选型建议、附 L 分纬度带。**`图 L*` 这个命名空间整个消失**
  - **留下的**：`# 1.` 结论摘要（只讲谱）/ `# 2.` 数据范围与检验口径 /
    `# 3.` 总体结果（`图 1`）/ `# 4.` 功率谱检验（§4.1–4.7，`图 2`–`图 6`）/
    附录 + 附 S。第 3 节的综合相对 RMSE / ACC / FA **留着不是再判一次优劣**，
    是给「相对」口径当 100 的基线；判读句一律落在两套谱上
  - **图少了一张不是契约改口径**：原计划的 7 张里 `rmse_by_lead_range` 是
    `_section_rmse` 删掉后**没人引用的孤儿图**，与其出一张正文不指的文件，
    不如连生成块一起删——所以是 **6 张**，编号 `图 1`–`图 6` 连续
  - **雷达图本来就是退化形状**（不是没出图）：用户问「雷达图似乎没放入报告里」，
    实测文件在（原 §5.7 图 10 / 主报告 附 7），但这份产物**两个模型槽喂的是同一份
    归档**，5 根轴全打平，而渲染器对打平的轴一律钳到 1.0（防除零），
    于是两个多边形与单位圆重合、互相盖住，画出来就是个灰五边形。**零信息量**，
    这是删掉它的直接理由。同一次对账还查出同名指标两个值——
    `det_report` 的「频谱对数 RMS×100」= 113.83 与这份的 `log_rms_ln` = 214.09
    差 **×2.3026（log10→ln）再叠加变量数**（前者用 `_pick_variable` 只取 z500，
    后者对 6 个变量先平均）；`det_report.py:47` 的 docstring 写了个 `mean_v`
    而实现并没有做，属文档与代码不符，**尚未处理**
  - **文档同步**：`SKILL.md` / `README.md` / `references/evaluation-metrics.md` /
    `references/rmse-batch-evaluation.md` / `scripts/generate_wave_report.py` /
    `wave_report.py` 六处「球谐带 §5.1–5.3」全部改成 **§4.1–4.3**
  - 骨架与渲染器**一起改**（`assets/templates/weather_rmse_wave.md` +
    `visualization/wave_report.py` 的 `render_wave` / `build_wave_report`），
    删掉 `_spectrum_inclusive_rank` / `_section_rmse` / `_heatmap_block` /
    `_single_init_block` / `capability_profile` / `_capability_block` /
    `_section_risks` / `_section_acceptance` / `_region_block` 九个函数
  - **另一件没解决的事**：这份骨架**没有测试守着**（`test_weather_rmse_wave.py`
    不存在，且 `tests/` 本身没进版本库），改动只能靠实跑对拍

- **2026-10-08**: 新增「降水个例分析」骨架与渲染器（暴雨过程个例，双口径）
  - **新增第四类交付物**：`assets/templates/rainstorm_case.md` +
    `visualization/rainstorm_report.py` + `rainstorm_plots.py` +
    `scripts/generate_rainstorm_report.py`。它吃的是
    `configs/rainstorm_ts_single_*` 的产物根——一个过程两套段
    （`{编号}_24h/{起报日}/` 与 `{编号}_total/`），一次评测 43 个过程 ×
    2 种口径 = 255 段独立产物，报告**按过程分节**（不是按时效分节）
  - **流水线相同、渲染器不同**：每段产物的 `pipeline` 也是 `weather_ts_det`、
    宽表列名也和 TS 系列一样，但 `ts_report.py` 假定"一次连续时段评测、
    一张宽表"，结构上吃不下这批产物。所以它走独立渲染器，
    **不能喂给 `generate_report.py`**（喂得进去、出的却是错的章节口径）。
    这是"模板按家族分、不按流程分"原则之外的第一处例外，已在 SKILL.md 写明
  - **两种口径互不可比**，报告里两套数分节呈现、只在同一口径内跨模型比：
    口径 A（逐日 24h）一天一个 08–08 窗、基准率随每日实况浮动；
    口径 B（过程累积）一根预报累加整段、基准率显著更高。实测 202503 的
    ≥25mm TS：日均 0.210 vs 过程累积 0.313
  - **过程编目收敛到一份真值**：新增 `xmetai_evaluation/rainstorm_catalog.py`
    （43 条：编号 → 起止 / 等级）。此前配置 `rainstorm_ts_single_fuxi.py`、
    `rainstorm_ts_single_fgvp_ctrl.py` 与 `rainstorm_summary.py` **各抄了一份
    46 行的过程表**，改一头另两头不会跟着动；现在三处都 `import` 编目模块，
    重构后逐一核对过 43 个过程 / 255 个 cfg / 输出路径逐项相同
  - **`--process` 必给**：43 个过程全出会变成一本册子，所以要求显式点名；
    先跑 `rainstorm_summary.py`、读它的 `summary_long.csv` 照着挑
  - **跨模型汇总逐行配对，保证 Δ 两边同分母**：一边缺段时「各算各的均值」是两个
    分母，Δ 里混着覆盖率差。首次实跑（FuXi 43 个过程 vs FGVP-ctrl 33 个）就撞上，
    Δ 从 +0.020 被放大到 +0.041。修法是**逐行只取两边都有有效值的过程**
    （`_paired_values`），不是只按「有没有产物」筛——后者仍会漏：
    202534 的 FGVP 只有 `_24h`、没有 `_total`，按产物筛那一行照样是两个分母。
    修后 表 15 的 Δ 与独立复算逐位一致（口径 B +0.020、口径 A +0.034）。
    同步在第六节加了第 3 条口径说明，并把「九条」改成「十条」
  - **不出降水空间分布**：产物里只有站点列联表的聚合统计
    （hits / misses / false_alarms / n_pairs / TS…），没有逐站或逐格降水量，
    画不出实况场也画不出差值场。这件事三种查法都验过（配置的 `writers`、
    `manifest.artifacts`、`.states/*.pkl` 反序列化），报告第六节第 8 条写明
  - **第一节补了「两种口径怎么算」**：把两种口径的算法写进摘要（窗口怎么切、
    往不往上累加、四格表怎么出、「日均」为什么是逐日 TS 的算术平均而不是合并
    四格表），并给出那个最有说服力的反例——「8 天每天 8mm 的连绵小雨」与
    「1 天 60mm 的一场暴雨」在口径 B 眼里完全相同。计算方式是**配置层面的事实**
    （跟模型、跟产物无关）所以写死；只有末尾两个基准率是跟着产物算的
    （`_base_rate` / `_base_rate_mean`），骨架上留 `{…}`。实测一批：
    口径 A 逐日平均 5.1%、口径 B 逐过程平均 28.7%，差 5.6 倍——
    这就是「TS 不可跨口径比」的定量依据
  - **口径 A 的过程级数值改成「汇总重算」**（2026-10-08）：原来是各起报日 TS 的
    算术平均，用户指出应当把该过程所有起报日的 hit/miss/FA 汇到一起重算——
    平均比率会被小样本日拖偏。渲染器新增 `daily_pooled` / `daily_counts` /
    `_from_counts`，`daily_mean` 删掉；`rainstorm_summary.py` 新增 `pooled_ts` /
    `load_wide` / `_wide_counts`。注意 **`scores.csv` 里没有四格表计数**，
    汇总必须读 `diagnostics/categorical_wide.csv`。实测影响：202501 ≥25mm
    0.110→0.191、43 过程整体 0.273→0.316（FuXi）；Δ 从 +0.028 变 +0.022。
    基准率对算法不敏感（逐日平均 5.1% vs 逐过程汇总 5.1%），因为各日起报的
    n_pairs 几乎相同，所以那两个 {…} 数字没变
  - **第二节补了第二张图**：原来只有口径 B 的全过程柱状图，现在口径 A 一张
    （`process_overview_daily.png`，图 1）、口径 B 一张（`process_overview.png`，
    图 2），`plot_process_overview` 加 `metric_label` 参数（只改文字）；
    第三节起的图号整体后移一位
  - **这个骨架没有测试守着**：`tests/` 不在 v2 这个工作副本里，靠实跑对拍
    （已用同一份产物喂两个模型名跑通两个过程，图号表号连续、
    Δ 列全 `+0.000` 属预期的退化形状；改渲染器后又用
    「骨架非占位行必须逐字出现在 REPORT.md 里」扫过一遍：26 行、0 缺失）
  - **改进建议的分支要覆盖「两个口径都领先」**：原来 `if 总量赢且逐日不赢 / elif
    总量输 / else`，`else` 兜底写的是「两种口径都接近持平」——主模型两个口径都
    明显领先时也落进 `else`，实测输出成「接近持平（口径 B 22:3，口径 A 29:3）」。
    已拆成五条分支，「持平」只留给总量真正持平的情形
  - 已知遗留：`rainstorm_summary.py` 缺 `stream.reconfigure(encoding="utf-8")`
    那段，Windows 的 GBK 控制台下中文输出乱码（**产物本身是 UTF-8，不受影响**）

---

**维护者**: 沈哲文 (szw)
**最后更新**: 2026-10-08
