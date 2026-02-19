# MOEA/D × LLM — 跨学科长尾科研知识多目标进化系统

本项目探索将**多目标进化算法（MOEA/D）**与**大语言模型（LLM）**结合，用于生成、评估和进化跨学科长尾科研知识。

## 项目概览

LLM 充当进化算子：**生成初始种群** → **多维度评分** → **交叉变异** → **迭代进化**。

```
初始种群 (LLM生成)
    │
    ▼
┌─────────────────────────────┐
│  MOEA/D 多目标进化框架        │
│  ① 偏置权重向量 (Dirichlet) │
│  ② 惩罚约束 Tchebycheff     │
│  ③ 精英保留                 │
│  LLM 交叉变异 + 评估         │
└─────────────────────────────┘
    │
    ▼
Pareto 最优解集（最优权衡面）
```

## 实验版本

| 版本 | 文件 | 模型 | 目标数 | 说明 |
|------|------|------|--------|------|
| v1 | `moead_cs_research.py` | DeepSeek | 3 | 计算机科学长尾知识，基础版 |
| v2 | `moead_science.py` | DeepSeek | 5 | 扩展至全科学域，5目标 |
| v3 | `moead_science_v2.py` | Claude-sonnet-4-5 | 7 | 加入可行性+合理性，7目标 |
| v4 | `moead_science_v3.py` | DeepSeek | 7 | 算法强化版（5项改进） |
| 对比 | `moead_compare_exp.py` | DeepSeek | 1 vs 4 | 单目标 vs 多目标受控实验 |

## 7 个优化目标（全部最大化，0-10分）

| 目标 | 说明 |
|------|------|
| 知识价值 | 对基础科学理论的贡献深度 |
| 社会影响 | 对人类社会的长远正向影响 |
| 长尾度 | 研究的稀缺性/小众程度 |
| 跨学科性 | 跨领域连接与融合潜力 |
| 前沿性 | 新颖程度 / 突破已知边界 |
| **可行性** | 当前技术条件下可开展研究的程度 |
| **合理性** | 科学假设的逻辑严谨性与证伪可能性 |

## 算法改进（v4 强化版）

1. **偏置权重向量**：Dirichlet(α=[1,1,1,1,1,**3,3**])，可行性/合理性获得 3× 关注
2. **惩罚约束 Tchebycheff**：低于阈值(5.5分)时施加软约束惩罚
3. **精英保留**：每代锁定最优可行性/合理性个体不被替换
4. **强化 LLM 提示**：交叉时硬要求后代可行≥6、合理≥6
5. **初始种群偏置**：要求 LLM 生成"当前可开展+理论有依据"的知识

## 核心发现（权衡关系分析）

| 关系 | r 值 | 类型 |
|------|------|------|
| 可行性 ↔ 合理性 | +0.69 | 强协同（验证轴） |
| 长尾度 ↔ 可行性 | −0.50 | 强对抗（核心矛盾） |
| 前沿性 ↔ 合理性 | −0.09 | **独立**（最反直觉） |

**最反直觉发现**：前沿性与合理性几乎不相关——前沿不必牺牲严谨，制约两者共存的真正瓶颈是**可行性**。

## 对比实验结果

|  | 单目标（仅新颖） | 四目标 MOEA/D |
|--|---------|---------|
| 新颖性 | **9.8** | 7.6 |
| 可行性 | 2.1 | **6.6** |
| 合理性 | 3.5 | **7.7** |

单目标进化收敛到「暗物质多元宇宙通信生物学」等科幻方向；四目标 MOEA/D 收敛到可实验的前沿研究。

## 文件结构

```
moead-research/
├── README.md
├── .gitignore
│
├── # ── v1：3目标 CS 长尾知识 ──
├── moead_cs_research.py        DeepSeek，3目标
├── moead_pdf_report.py         PDF 报告生成器
│
├── # ── v2：5目标全科学域 ──
├── moead_science.py            DeepSeek，5目标
├── moead_science_pdf.py        PDF 报告生成器
├── moead_science_results.json  运行结果
│
├── # ── v3：7目标 Claude ──
├── moead_science_v2.py         Claude-sonnet-4-5，7目标
├── moead_science_v2_pdf.py     PDF 报告生成器
├── moead_science_v2_results.json
│
├── # ── v4：7目标强化版 ──
├── moead_science_v3.py         DeepSeek，7目标 + 5项算法改进
├── moead_science_v3_pdf.py     PDF 报告生成器
├── moead_science_v3_results.json   Claude 版结果
├── moead_science_v3ds_results.json DeepSeek 版结果
│
├── # ── 对比实验 ──
├── moead_compare_exp.py        单目标 vs 四目标受控对比
├── moead_compare_pdf.py        对比报告生成器
├── moead_compare_results.json  对比实验结果
│
├── # ── 权衡分析 ──
├── moead_tradeoff_pdf.py       权衡关系分析 PDF（ReportLab）
│
└── latex/
    └── tradeoff_report.tex     权衡分析报告（LaTeX 版）
```

## 快速开始

```bash
# 安装依赖
pip install openai anthropic numpy reportlab

# 设置 API Key
export DEEPSEEK_API_KEY="your-key-here"

# 运行最新版本（v4 强化版）
python moead_science_v3.py

# 运行对比实验
python moead_compare_exp.py

# 生成权衡分析 PDF
python moead_tradeoff_pdf.py

# 编译 LaTeX 报告
cd latex && xelatex tradeoff_report.tex
```

## 依赖

- Python 3.9+
- `openai` — DeepSeek / OpenAI API
- `anthropic` — Claude API
- `numpy` — 权重向量生成、Pareto 计算
- `reportlab` — PDF 报告生成
- XeLaTeX + PingFang SC — LaTeX 报告编译（macOS）
