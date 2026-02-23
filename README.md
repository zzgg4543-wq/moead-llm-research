# MOEA/D × LLM 多目标进化系统

> 用**大模型 + 多目标进化算法**做两件事：① 长尾科研知识进化 ② 博后/职业选择优化

**隐私说明**：`.env`、演化结果 JSON/HTML/图表 已加入 `.gitignore`，不会推送。请勿将 API Key 或含个人信息的 profile 提交到仓库。

---

## 一眼看懂

| 项目 | 做什么 | 入口 | 输出 |
|------|--------|------|------|
| **科研知识进化** | LLM 生成 CS/跨学科研究课题 → MOEA/D 多目标优化 → Pareto 前沿 | `moead_cs_exp.py` / `moead_science_v3.py` | 长尾课题 + PDF 报告 |
| **博后职业选择** | LLM 评估博后/大厂/教职选项 → 11 维打分 → 演化研究路线 → Pareto 排序 | `moead_career_exp.py` | 推荐选项 + 演化日志 + PDF |

**共同点**：都是「LLM 生成/评估 + MOEA/D 多目标进化」，只是决策对象不同（科研课题 vs 职业选项）。

---

## 一、博后/职业选择（Career MOEA/D）

**目标**：在 NTU / Stanford / 国内大厂 / 美国其他学校 等选项中，用 11 维目标优化出 Pareto 最优博后/职业路径。

### 11 维目标

| 维度 | 说明 |
|------|------|
| 契合 | 与 MOEA/LLM/AI4Science 背景匹配 |
| 影响力 | 顶会顶刊、领域认可 |
| 职业 | 教职/工业界/创业支持 |
| 资源 | 经费、算力、招聘稳定 |
| 成长 | 新技能、新领域、独立性 |
| 可行 | 拿到 offer 概率 |
| 风险 | 时间、沉没、退出成本 |
| **Agent** | 与 AI agent 时代趋势契合 |
| **格局** | 中美博弈下定位与弹性 |
| **启发** | 对突破性思考的激发 |
| **上限** | 职业与认知天花板 |

### 演化机制

- **交叉**：研究路线交叉（融合两条路线的方法论与应用域）
- **变异**：时间线/产出/延续策略/退出路径微调
- **扩展**：每代生成 3 个新选项（国内大厂、美国其他学校）

### 运行

```bash
export DEEPSEEK_API_KEY="your-key"
python moead_career_exp.py      # 运行演化（约 8 分钟，5 代）
python moead_career_pdf.py      # 生成 PDF 报告
```

### 通用预测系统（Web + 引擎）

通用 MOEA/D 引擎 `moead_engine.py`：可配置目标、Profile、硬约束、超参数；支持 Web 界面或命令行脚本调用。

#### 快速开始

```bash
# 1. 配置 API Key（任选其一，写入 .env）
cp .env.example .env
# 编辑 .env：
#   OPENROUTER_API_KEY=xxx    # 推荐，可调用 DeepSeek/OpenAI 等
#   或 DEEPSEEK_API_KEY=xxx
#   或 OPENAI_API_KEY=xxx

# 2. 安装并启动
pip install -r requirements.txt
python app.py
# 浏览器打开 http://127.0.0.1:5001
```

#### 功能

| 功能 | 说明 |
|------|------|
| **多目标** | 自定义目标名称与定义，0–10 分评估 |
| **硬约束** | 所有选项必须满足，违反者降分/剔除 |
| **预设** | 职业、科研、研究生发展（含分年路径提示） |
| **API** | 优先 OpenRouter → DeepSeek → OpenAI，模型可配置 |

#### 命令行演化脚本

```bash
# 研究生发展路径（自定义 profile 请修改脚本内 PROFILE）
python run_fanfu_evolution.py

# 博后职业选择（赵哲预设）
python run_zhe_evolution.py

# 生成 Pareto 表格 HTML
python gen_pareto_table.py

# 绘制演化曲线
python plot_fanfu_results.py
```

#### Web 输入

- **Profile**：背景、约束、偏好
- **Domain**：预测领域（如博后选择、研究生三年发展路径）
- **硬约束**：每行一条，必满足
- **目标**：名称 + 定义
- **超参数**：种群、代数、交叉对数、变异率

### 产出

| 文件 | 说明 |
|------|------|
| `moead_career_results.json` | Pareto 排序、推荐理由、各代统计 |
| `career_logs/gen_XX_population.json` | 每代完整种群（选项+路线+得分） |
| `moead_career_report.pdf` | 演化曲线、分数变化、Pareto Top5（需运行 pdf 脚本生成） |

### 最新结论（5 代演化）

- **Top 1**：MIT CSAIL 优化与学习组（契合 10、启发 10、上限 10）
- **Top 2**：阿里达摩院 AutoML（契合 10、Agent 9、格局 9）
- **新加坡 NTU**：演化中被 Stanford/MIT 选项支配，主要因契合度、Agent 契合、上限较低

---

## 二、科研知识进化（Science MOEA/D）

**目标**：LLM 生成长尾科研课题 → MOEA/D 多目标进化 → Pareto 前沿的「新颖+可行+合理」研究方向。

### 7 维目标

知识价值、社会影响、长尾度、跨学科性、前沿性、**可行性**、**合理性**

### 实验版本

| 版本 | 文件 | 说明 |
|------|------|------|
| v1 | `moead_cs_research.py` | 3 目标，CS 长尾知识 |
| v2 | `moead_science.py` | 5 目标，全科学域 |
| v3 | `moead_science_v2.py` | 7 目标，Claude |
| v4 | `moead_science_v3.py` | 7 目标 + 5 项算法改进，DeepSeek |
| 对比 | `moead_compare_exp.py` | 单目标 vs 四目标受控实验 |
| CS+方案 | `moead_cs_exp.py` | CS 主题 + 研究方案同步进化 |

### 运行

```bash
python moead_science_v3.py      # v4 强化版
python moead_compare_exp.py     # 对比实验
python moead_cs_exp.py          # CS + 研究方案
```

### 核心发现

- **长尾度 ↔ 可行性**：强对抗（r≈−0.5）
- **前沿性 ↔ 合理性**：几乎独立（r≈−0.09）— 前沿不必牺牲严谨
- 单目标（仅新颖）→ 科幻方向；四目标 → 可实验的前沿研究

---

## 文件结构

```
moead-research/
├── README.md                   本说明
├── .env.example                API Key 模板（复制为 .env 使用，勿提交）
├── requirements.txt
│
├── # ═══ 通用预测系统 ═══
├── moead_engine.py             通用 MOEA/D 引擎（目标/约束/archive/all_options）
├── app.py                      Web API（Flask）
├── web/index.html              前端：表单 + 结果 + Pareto 排序
├── run_fanfu_evolution.py      研究生发展路径命令行演化
├── run_zhe_evolution.py        博后职业选择命令行演化
├── gen_pareto_table.py         从 JSON 生成 Pareto 表格 HTML
├── plot_fanfu_results.py       绘制演化曲线与雷达图
│
├── # ═══ 博后/职业选择 ═══
├── moead_career_exp.py         主程序（5 代，扩展变异）
├── moead_career_pdf.py         PDF 报告生成
├── career_logs/                每代种群日志
│
├── # ═══ 科研知识进化 ═══
├── moead_cs_research.py        v1 基础
├── moead_science*.py           v2–v4
├── moead_compare_exp.py        对比实验
├── moead_cs_exp.py             CS + 研究方案
├── moead_*_pdf.py / moead_*_report.py   各版 PDF/报告
├── moead_*_results.json        各版结果
│
├── # ═══ 权衡分析 ═══
├── moead_tradeoff_pdf.py
└── latex/tradeoff_report.tex
```

---

## 依赖

```bash
pip install -r requirements.txt
# openai, python-dotenv, numpy, flask, reportlab, anthropic
```

- **openai**：DeepSeek / OpenAI / OpenRouter 兼容
- **python-dotenv**：加载 .env
- **numpy**：权重、Pareto
- **flask**：Web API
- **reportlab**：PDF
