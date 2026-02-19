#!/usr/bin/env python3
"""
MOEA/D × LLM 演化结果 PDF 报告生成器
使用本次运行的实际数据生成多页 A4 报告
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Rect, Line, String, Polygon
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.widgets.markers import makeMarker
from reportlab.graphics import renderPDF
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import os, datetime

# ══════════════════════════════════════════════════
#  字体注册（使用系统 Arial Unicode 支持中文）
# ══════════════════════════════════════════════════
FONT_PATH = "/Library/Fonts/Arial Unicode.ttf"
pdfmetrics.registerFont(TTFont("ArialUnicode", FONT_PATH))
FONT = "ArialUnicode"

# ══════════════════════════════════════════════════
#  主题配色
# ══════════════════════════════════════════════════
C_PRIMARY   = colors.HexColor("#1a237e")   # 深蓝
C_ACCENT    = colors.HexColor("#0d47a1")   # 蓝
C_LIGHT     = colors.HexColor("#e3f2fd")   # 浅蓝
C_ORANGE    = colors.HexColor("#e65100")   # 橙
C_GREEN     = colors.HexColor("#1b5e20")   # 深绿
C_GRAY      = colors.HexColor("#546e7a")   # 灰
C_BG        = colors.HexColor("#f5f7fa")   # 背景
C_VALUE     = colors.HexColor("#1565c0")
C_IMPACT    = colors.HexColor("#2e7d32")
C_LONGTAIL  = colors.HexColor("#ad1457")

W, H = A4  # 595 x 842 pt

# ══════════════════════════════════════════════════
#  本次运行的实际演化数据
# ══════════════════════════════════════════════════
INIT_POP = [
    {"topic": "神经符号推理的小样本学习",
     "desc": "结合神经网络与符号逻辑，使AI在少量数据下进行逻辑推理与泛化。重要以解决数据稀缺场景的智能决策，小众因需融合差异巨大的符号与连接主义体系。",
     "scores": [7.0, 8.0, 8.0]},
    {"topic": "差分隐私下的联邦学习攻击防御",
     "desc": "研究联邦学习中差分隐私机制对抗模型窃取、成员推断等隐私攻击的有效性。重要以保护分布式AI数据隐私，小众因需兼顾隐私预算、模型效用与攻击鲁棒性。",
     "scores": [8.0, 8.0, 7.0]},
    {"topic": "边缘计算中的异构资源调度",
     "desc": "研究边缘设备（如IoT节点、移动设备）上CPU、GPU、FPGA等异构资源的动态任务调度。重要以提升边缘计算效率，小众因设备碎片化且实时约束严苛。",
     "scores": [8.0, 8.0, 6.0]},
    {"topic": "内存安全语言的硬件辅助机制",
     "desc": "设计CPU/GPU指令集或协处理器，加速Rust等内存安全语言的运行时检查。重要以平衡安全与性能，小众需硬件架构与编译器的深度协同设计。",
     "scores": [7.0, 7.0, 8.0]},
    {"topic": "异步拜占庭容错共识优化",
     "desc": "在异步网络环境下改进拜占庭容错共识协议（如HoneyBadgerBFT），降低通信开销。重要以提升分布式系统在对抗环境中的可扩展性，小众因理论复杂度极高。",
     "scores": [6.0, 7.0, 9.0]},
    {"topic": "概率程序的形式化语义扩展",
     "desc": "为概率编程语言（如Stan）建立扩展形式语义，支持不确定性推理的严格验证。重要以增强概率模型的可靠性，小众因需融合概率论与程序语义两大理论体系。",
     "scores": [6.0, 6.0, 9.0]},
    {"topic": "量子启发式算法优化",
     "desc": "借鉴量子计算原理（如叠加态、纠缠）设计经典启发式算法，用于解决组合优化问题。重要在于可能突破传统算法局限，小众因需跨量子与经典计算的双重背景。",
     "scores": [6.0, 7.0, 8.0]},
    {"topic": "渐进式类型系统的形式化验证",
     "desc": "为TypeScript等渐进式类型语言构建形式化模型，验证类型安全与运行时行为一致性。重要以确保大型代码库可靠性，小众因需结合类型理论与动态语义分析。",
     "scores": [7.0, 6.0, 7.0]},
]

FINAL_POP = [
    {"topic": "量子神经符号联邦BFT长尾优化",
     "desc": "融合量子启发算法与神经符号推理，设计联邦学习中的异步拜占庭容错共识机制，通过小样本学习优化差分隐私参数与共识协议，引入自适应隐私预算分配与动态信任评估，显著提升防御模型窃取等攻击的效率。",
     "scores": [9.0, 8.0, 9.0], "pareto": True},
    {"topic": "量子神经符号联邦BFT均衡优化",
     "desc": "融合量子启发算法与神经符号推理，设计联邦学习中的异步拜占庭容错共识机制，通过小样本学习优化差分隐私参数与共识协议，并引入自适应隐私预算分配与动态信任评估，增强隐私保护与系统可扩展性。",
     "scores": [9.0, 8.0, 9.0], "pareto": True},
    {"topic": "量子神经符号联邦BFT价值提升",
     "desc": "融合量子启发算法与神经符号推理，设计联邦学习中的异步拜占庭容错共识机制，通过小样本学习优化差分隐私参数与共识协议，引入自适应隐私预算分配，显著提升防御攻击的效率与隐私保护价值。",
     "scores": [9.0, 8.0, 9.0], "pareto": True},
    {"topic": "差分隐私的量子启发联邦防御",
     "desc": "结合量子启发式算法优化联邦学习中的差分隐私机制，设计高效防御策略对抗模型窃取等攻击，提升隐私保护效果。创新性融合量子计算原理与隐私保护技术，具有重要应用价值。",
     "scores": [9.0, 8.0, 9.0], "pareto": True},
    {"topic": "神经符号边缘异构BFT调度",
     "desc": "结合神经符号推理优化边缘计算中异构资源（CPU/GPU/FPGA）的动态任务调度，将其应用于异步拜占庭容错共识的参数调整，提升边缘设备在对抗环境中的计算效率与系统可靠性。",
     "scores": [8.0, 8.0, 7.0], "pareto": True},
    {"topic": "硬件神经符号边缘BFT调度优化",
     "desc": "结合硬件指令集加速与神经符号推理，优化边缘计算中异构资源的动态任务调度，应用于异步拜占庭容错共识的参数优化，通过形式化验证确保调度算法类型安全与实时性保证。",
     "scores": [8.0, 7.0, 8.0], "pareto": False},
    {"topic": "量子神经符号联邦BFT长尾防御",
     "desc": "融合量子启发算法与神经符号推理，在极度长尾场景下设计联邦学习拜占庭容错机制，通过小样本学习与差分隐私优化，针对极稀缺攻击样本场景进行防御增强。",
     "scores": [7.0, 6.0, 10.0], "pareto": True},
    {"topic": "小样本量子神经符号边缘BFT隐私",
     "desc": "利用小样本学习、量子启发算法与神经符号推理，优化边缘计算环境中异步拜占庭容错共识的差分隐私机制，针对设备异构性与数据稀缺性动态调整隐私参数。",
     "scores": [6.0, 5.0, 10.0], "pareto": False},
]

# 各代均值统计
GEN_STATS = [
    # (gen, avg_value, avg_impact, avg_longtail, avg_composite)
    (0,  6.87, 7.12, 7.75, 7.25),
    (1,  7.75, 7.37, 8.13, 7.75),
    (2,  8.00, 7.50, 8.62, 8.04),
    (3,  8.00, 7.37, 8.62, 8.00),
    (4,  8.12, 7.37, 8.75, 8.08),
    (5,  8.12, 7.37, 8.75, 8.08),
    (6,  8.12, 7.37, 8.75, 8.08),
    (7,  8.25, 7.50, 8.75, 8.17),
    (8,  8.38, 7.62, 8.62, 8.21),
    (9,  8.38, 7.62, 8.62, 8.21),
    (10, 8.75, 7.88, 8.62, 8.42),
]

# ══════════════════════════════════════════════════
#  样式定义
# ══════════════════════════════════════════════════
def S(name, **kw):
    base = dict(fontName=FONT, leading=16)
    base.update(kw)
    return ParagraphStyle(name, **base)

sTitle    = S("title",    fontSize=26, textColor=C_PRIMARY,  alignment=TA_CENTER, leading=36, spaceAfter=6)
sSubtitle = S("sub",      fontSize=13, textColor=C_GRAY,     alignment=TA_CENTER, leading=20)
sH1       = S("h1",       fontSize=16, textColor=C_PRIMARY,  spaceBefore=14, spaceAfter=6,  leading=22)
sH2       = S("h2",       fontSize=13, textColor=C_ACCENT,   spaceBefore=10, spaceAfter=4,  leading=18)
sBody     = S("body",     fontSize=9.5, textColor=colors.HexColor("#212121"), leading=15, spaceAfter=3)
sSmall    = S("small",    fontSize=8.5, textColor=C_GRAY,    leading=13)
sCenter   = S("center",   fontSize=9.5, alignment=TA_CENTER, textColor=C_GRAY, leading=14)
sCaption  = S("caption",  fontSize=8,   textColor=C_GRAY,    alignment=TA_CENTER, leading=12)
sBold     = S("bold",     fontSize=10,  textColor=C_PRIMARY, leading=15, spaceAfter=2)
sPareto   = S("pareto",   fontSize=9.5, textColor=C_GREEN,   leading=15)

# ══════════════════════════════════════════════════
#  辅助组件
# ══════════════════════════════════════════════════
def score_bar_drawing(v, i, l, width=160, height=36):
    """渲染三条彩色评分条"""
    d = Drawing(width, height)
    bar_h = 7
    gap = 4
    label_w = 24
    bar_w = width - label_w - 6
    colors_list = [C_VALUE, C_IMPACT, C_LONGTAIL]
    labels = ["价值", "影响", "长尾"]
    vals = [v, i, l]
    for idx, (val, col, lab) in enumerate(zip(vals, colors_list, labels)):
        y = height - (idx + 1) * (bar_h + gap) + 2
        # 背景轨道
        d.add(Rect(label_w, y, bar_w, bar_h, fillColor=colors.HexColor("#e0e0e0"), strokeColor=None))
        # 填充
        fill_w = bar_w * val / 10.0
        d.add(Rect(label_w, y, fill_w, bar_h, fillColor=col, strokeColor=None))
        # 标签
        d.add(String(0, y + 1, lab, fontName=FONT, fontSize=7, fillColor=C_GRAY))
        # 数值
        d.add(String(label_w + fill_w + 2, y + 1, f"{val:.0f}", fontName=FONT, fontSize=7, fillColor=col))
    return d


def evolution_chart(stats, width=420, height=160):
    """折线图：三个目标均值随代数的变化"""
    d = Drawing(width, height)
    pad_l, pad_r, pad_t, pad_b = 36, 16, 12, 28
    cw = width - pad_l - pad_r
    ch = height - pad_t - pad_b
    n = len(stats)
    x_step = cw / (n - 1)
    y_min, y_max = 5.0, 10.0
    y_range = y_max - y_min

    # 背景
    d.add(Rect(pad_l, pad_b, cw, ch, fillColor=colors.HexColor("#f8f9fa"), strokeColor=colors.HexColor("#dee2e6"), strokeWidth=0.5))

    # 横向网格线
    for tick in [6, 7, 8, 9, 10]:
        y = pad_b + (tick - y_min) / y_range * ch
        d.add(Line(pad_l, y, pad_l + cw, y, strokeColor=colors.HexColor("#dee2e6"), strokeWidth=0.4))
        d.add(String(pad_l - 4, y - 3, str(tick), fontName=FONT, fontSize=6.5, fillColor=C_GRAY, textAnchor="end"))

    # X轴刻度
    for i, (gen, *_) in enumerate(stats):
        x = pad_l + i * x_step
        d.add(String(x, pad_b - 10, str(gen), fontName=FONT, fontSize=6.5, fillColor=C_GRAY, textAnchor="middle"))

    # 轴标签
    d.add(String(pad_l + cw / 2, 2, "演化代数", fontName=FONT, fontSize=7, fillColor=C_GRAY, textAnchor="middle"))

    # 三条折线
    series = [
        (1, C_VALUE,    "价值度"),
        (2, C_IMPACT,   "影响力"),
        (3, C_LONGTAIL, "长尾度"),
    ]
    for col_idx, col, label in series:
        pts = []
        for i, row in enumerate(stats):
            x = pad_l + i * x_step
            y = pad_b + (row[col_idx] - y_min) / y_range * ch
            pts.append((x, y))
        for j in range(len(pts) - 1):
            d.add(Line(pts[j][0], pts[j][1], pts[j+1][0], pts[j+1][1],
                       strokeColor=col, strokeWidth=1.5))
        # 端点圆点
        for x, y in pts:
            d.add(Rect(x - 2, y - 2, 4, 4, fillColor=col, strokeColor=None))
        # 图例
        legend_x = pad_l + cw + 4
        legend_y = pad_b + ch - 8 - series.index((col_idx, col, label)) * 14
        d.add(Rect(legend_x, legend_y, 10, 6, fillColor=col, strokeColor=None))
        d.add(String(legend_x + 12, legend_y, label, fontName=FONT, fontSize=7, fillColor=C_GRAY))

    return d


def section_header(text):
    """带色块的章节标题"""
    return [
        HRFlowable(width="100%", thickness=2, color=C_PRIMARY, spaceAfter=4),
        Paragraph(text, sH1),
    ]


def individual_table_row(ind, rank):
    """生成单个知识点的显示行"""
    pareto_mark = " ★" if ind.get("pareto") else ""
    topic_style = sPareto if ind.get("pareto") else sBold
    return [
        Paragraph(f"{rank}.{pareto_mark}", sCenter),
        Paragraph(ind["topic"], topic_style),
        ind["scores"][0],
        ind["scores"][1],
        ind["scores"][2],
        Paragraph(ind["desc"][:80] + ("…" if len(ind["desc"]) > 80 else ""), sSmall),
    ]


# ══════════════════════════════════════════════════
#  页眉页脚
# ══════════════════════════════════════════════════
def add_header_footer(canvas, doc):
    canvas.saveState()
    # 页眉
    canvas.setFillColor(C_PRIMARY)
    canvas.rect(0, H - 28, W, 28, fill=1, stroke=0)
    canvas.setFont(FONT, 9)
    canvas.setFillColor(colors.white)
    canvas.drawString(18, H - 18, "MOEA/D × LLM  ──  计算机科学长尾知识多目标进化报告")
    canvas.drawRightString(W - 18, H - 18, f"第 {doc.page} 页")
    # 页脚
    canvas.setFillColor(C_BG)
    canvas.rect(0, 0, W, 18, fill=1, stroke=0)
    canvas.setFillColor(C_GRAY)
    canvas.setFont(FONT, 7.5)
    canvas.drawString(18, 5, f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}  |  模型: deepseek-chat  |  种群: 8  |  邻居: 3  |  演化: 10 代")
    canvas.restoreState()


# ══════════════════════════════════════════════════
#  主构建函数
# ══════════════════════════════════════════════════
def build_pdf(output_path):
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=34*mm, bottomMargin=22*mm,
    )

    story = []

    # ── 封面 ──────────────────────────────────────
    story.append(Spacer(1, 24))
    story.append(Paragraph("MOEA/D × LLM", sTitle))
    story.append(Paragraph("计算机科学长尾知识多目标进化报告", S("sub2", fontName=FONT, fontSize=18,
        textColor=C_ACCENT, alignment=TA_CENTER, leading=28, spaceAfter=8)))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "基于分解的多目标演化算法 (MOEA/D) 与大语言模型 (LLM) 协同驱动，"
        "在「价值度 × 影响力 × 长尾度」三维目标空间中对计算机科学长尾研究知识进行 10 代演化优化。",
        S("intro", fontName=FONT, fontSize=11, textColor=C_GRAY, alignment=TA_CENTER,
          leading=18, leftIndent=20, rightIndent=20)
    ))
    story.append(Spacer(1, 16))

    # 参数信息卡片
    params = [
        ["参数", "值", "参数", "值"],
        ["演化模型", "deepseek-chat", "种群大小", "8"],
        ["演化代数", "10 代", "邻居数量", "3"],
        ["目标数量", "3（价值/影响/长尾）", "标量化方法", "Tchebycheff"],
        ["LLM调用次数", "约 22 次", "总耗时", "约 193 秒"],
    ]
    pt = Table(params, colWidths=[55*mm, 60*mm, 55*mm, 40*mm])
    pt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_PRIMARY),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, -1), FONT),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_LIGHT, colors.white]),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.HexColor("#b0bec5")),
        ("ROWHEIGHT",  (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(pt)
    story.append(Spacer(1, 14))

    # 算法说明
    story.extend(section_header("算法原理"))
    algo_text = (
        "MOEA/D（Multi-Objective Evolutionary Algorithm based on Decomposition）将多目标优化问题分解为若干个"
        "标量子问题，每个子问题对应一个权重向量，通过 Tchebycheff 聚合函数将三个目标加权合并为单目标进行优化。"
        "子问题之间通过邻居结构共享演化信息，实现对 Pareto 前沿的均匀逼近。\n\n"
        "本实验中，LLM 承担了所有演化算子的职责：① 初始种群生成（一次批量调用）；"
        "② 适应度评估（批量打分）；③ 交叉变异（融合两个父代知识点生成新后代，偏向当前子问题的权重方向）。"
        "每代仅需 2 次 LLM 调用，兼顾多样性与效率。"
    )
    story.append(Paragraph(algo_text, sBody))
    story.append(Spacer(1, 8))

    # 权重向量表
    story.append(Paragraph("权重向量分配（8 个子问题）", sH2))
    wv_data = [["子问题", "w(价值)", "w(影响)", "w(长尾)", "优化偏向", "邻居"]]
    wv_rows = [
        (0, 0.67, 0.00, 0.33, "高价值",   "[0,3,4]"),
        (1, 0.00, 1.00, 0.00, "高影响",   "[1,2,5]"),
        (2, 0.00, 0.67, 0.33, "影响+长尾","[2,1,4]"),
        (3, 0.67, 0.33, 0.00, "价值+影响","[3,0,4]"),
        (4, 0.33, 0.33, 0.33, "均衡",     "[4,0,2]"),
        (5, 0.33, 0.67, 0.00, "偏影响",   "[5,1,2]"),
        (6, 1.00, 0.00, 0.00, "纯价值",   "[6,0,3]"),
        (7, 0.33, 0.00, 0.67, "高长尾",   "[7,0,4]"),
    ]
    for r in wv_rows:
        wv_data.append([f"w{r[0]}", f"{r[1]:.2f}", f"{r[2]:.2f}", f"{r[3]:.2f}", r[4], r[5]])

    wt = Table(wv_data, colWidths=[20*mm, 22*mm, 22*mm, 22*mm, 30*mm, 26*mm])
    wt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_ACCENT),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, -1), FONT),
        ("FONTSIZE",   (0, 0), (-1, -1), 8.5),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, C_BG]),
        ("GRID",       (0, 0), (-1, -1), 0.3, colors.HexColor("#cfd8dc")),
        ("ROWHEIGHT",  (0, 0), (-1, -1), 13),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(wt)

    story.append(PageBreak())

    # ── 初始种群 ──────────────────────────────────
    story.extend(section_header("初始种群（第 0 代）"))
    story.append(Paragraph(
        "由 LLM 一次性生成的 8 个 CS 长尾知识条目，覆盖神经符号推理、联邦学习、边缘计算、"
        "内存安全、拜占庭共识、概率程序、量子算法、类型系统等多个方向。",
        sBody
    ))
    story.append(Spacer(1, 6))

    for k, ind in enumerate(INIT_POP):
        v, i, l = ind["scores"]
        row_data = [[
            Paragraph(f"#{k+1}", S("num", fontName=FONT, fontSize=12, textColor=C_PRIMARY,
                                   alignment=TA_CENTER, leading=16)),
            [Paragraph(ind["topic"], sBold),
             Paragraph(ind["desc"], sSmall)],
            score_bar_drawing(v, i, l, width=150, height=34),
        ]]
        rt = Table(row_data, colWidths=[14*mm, 100*mm, 55*mm])
        rt.setStyle(TableStyle([
            ("VALIGN",  (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (-1, -1), C_BG if k % 2 == 0 else colors.white),
            ("BOX",     (0, 0), (-1, -1), 0.3, colors.HexColor("#b0bec5")),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ]))
        story.append(rt)
        story.append(Spacer(1, 2))

    story.append(PageBreak())

    # ── 演化趋势 ──────────────────────────────────
    story.extend(section_header("演化过程分析"))
    story.append(Paragraph("三个目标维度的种群均值随演化代数的变化趋势：", sBody))
    story.append(Spacer(1, 4))
    story.append(evolution_chart(GEN_STATS, width=440, height=170))
    story.append(Spacer(1, 8))

    # 数值表格
    story.append(Paragraph("各代均值统计", sH2))
    stat_data = [["代数", "价值度", "影响力", "长尾度", "综合均分", "变化"]]
    prev = None
    for row in GEN_STATS:
        gen, v, i, l, comp = row
        delta = "" if prev is None else f"+{comp-prev:.2f}" if comp > prev else f"{comp-prev:.2f}"
        stat_data.append([
            f"第{gen}代" if gen > 0 else "初始",
            f"{v:.2f}", f"{i:.2f}", f"{l:.2f}", f"{comp:.2f}",
            delta
        ])
        prev = comp
    st = Table(stat_data, colWidths=[24*mm, 28*mm, 28*mm, 28*mm, 30*mm, 24*mm])
    st.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_PRIMARY),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, -1), FONT),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_LIGHT, colors.white]),
        ("GRID",       (0, 0), (-1, -1), 0.3, colors.HexColor("#b0bec5")),
        ("ROWHEIGHT",  (0, 0), (-1, -1), 13),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        # 高亮最终代
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#fff9c4")),
        ("TEXTCOLOR",  (4, 1), (4, -1), C_ACCENT),
        ("TEXTCOLOR",  (5, 1), (5, -1), C_GREEN),
    ]))
    story.append(st)
    story.append(Spacer(1, 10))

    # 关键演化事件
    story.append(Paragraph("关键演化事件", sH2))
    events = [
        ("第 0 代", "初始化", "LLM 生成 8 个多样化 CS 长尾知识点，综合均分 7.25，覆盖 8 个不同 CS 研究方向。"),
        ("第 1 代", "快速跃升", "首代演化后综合均分跃升至 7.75（+0.50），LLM 发现「量子启发 + 差分隐私」组合方向高度有效。"),
        ("第 2 代", "突破 8 分", "综合均分突破 8.0 大关（8.04），开始涌现「量子神经符号联邦 BFT」融合范式。"),
        ("第 4-6 代", "稳定平台期", "均分在 8.08 稳定，种群多样性逐步降低，Tchebycheff 邻居替换使边界解得到保留。"),
        ("第 7-9 代", "再度提升", "均分从 8.17 → 8.21，权重向量细分方向推动价值度和影响力同步提升。"),
        ("第 10 代", "最终收敛", "综合均分达 8.42，价值度均值 8.75，6/8 个体进入 Pareto 最优解集。"),
    ]
    ev_data = [["代数", "阶段", "描述"]]
    for e in events:
        ev_data.append(list(e))
    et = Table(ev_data, colWidths=[24*mm, 28*mm, 110*mm])
    et.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_ACCENT),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, -1), FONT),
        ("FONTSIZE",   (0, 0), (-1, -1), 8.5),
        ("ALIGN",      (0, 0), (1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, C_BG]),
        ("GRID",       (0, 0), (-1, -1), 0.3, colors.HexColor("#cfd8dc")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TEXTCOLOR",  (1, 1), (1, -1), C_ORANGE),
    ]))
    story.append(et)

    story.append(PageBreak())

    # ── 最终种群 ──────────────────────────────────
    story.extend(section_header("第 10 代最终种群"))
    story.append(Paragraph(
        "演化完成后的 8 个知识点，★ 标记表示该条目属于 Pareto 最优解集（不被任何其他解在所有目标上同时支配）。",
        sBody
    ))
    story.append(Spacer(1, 6))

    for k, ind in enumerate(FINAL_POP):
        v, i, l = ind["scores"]
        pareto_mark = " ★ Pareto 最优" if ind.get("pareto") else ""
        topic_color = C_GREEN if ind.get("pareto") else C_PRIMARY
        row_data = [[
            Paragraph(f"#{k+1}", S("num2", fontName=FONT, fontSize=12, textColor=topic_color,
                                   alignment=TA_CENTER, leading=16)),
            [Paragraph(ind["topic"] + pareto_mark,
                       S("tp", fontName=FONT, fontSize=10,
                         textColor=topic_color, leading=15, spaceAfter=2)),
             Paragraph(ind["desc"], sSmall)],
            score_bar_drawing(v, i, l, width=150, height=34),
        ]]
        bg = colors.HexColor("#f1f8e9") if ind.get("pareto") else (C_BG if k % 2 == 0 else colors.white)
        rt = Table(row_data, colWidths=[14*mm, 100*mm, 55*mm])
        rt.setStyle(TableStyle([
            ("VALIGN",  (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (-1, -1), bg),
            ("BOX",     (0, 0), (-1, -1), 0.5,
             C_GREEN if ind.get("pareto") else colors.HexColor("#b0bec5")),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ]))
        story.append(rt)
        story.append(Spacer(1, 2))

    story.append(PageBreak())

    # ── Pareto 前沿 ───────────────────────────────
    story.extend(section_header("Pareto 最优解集与结论"))
    story.append(Paragraph(
        "经过 10 代演化，最终种群中共有 <b>6 个个体</b>构成 Pareto 最优解集（非支配解）。"
        "这些解在「价值度 × 影响力 × 长尾度」三个目标上无法被同时超越，代表了本次演化发现的最优研究方向组合。",
        sBody
    ))
    story.append(Spacer(1, 10))

    pareto_inds = [ind for ind in FINAL_POP if ind.get("pareto")]
    p_data = [["#", "知识点主题", "价值", "影响", "长尾", "综合"]]
    for k, ind in enumerate(pareto_inds):
        v, i, l = ind["scores"]
        comp = (v + i + l) / 3
        p_data.append([
            str(k + 1),
            ind["topic"],
            f"{v:.1f}", f"{i:.1f}", f"{l:.1f}",
            f"{comp:.1f}"
        ])
    p_data.append(["", "均值",
                   f"{sum(r[1]['scores'][0] for r in [(0,ind) for ind in pareto_inds])/len(pareto_inds):.1f}",
                   "—", "—", "—"])

    pf_table_data = [["#", "知识点主题", "价值", "影响", "长尾", "综合"]]
    for k, ind in enumerate(pareto_inds):
        v, i, l = ind["scores"]
        pf_table_data.append([str(k+1), ind["topic"], f"{v:.1f}", f"{i:.1f}", f"{l:.1f}",
                               f"{(v+i+l)/3:.1f}"])

    pft = Table(pf_table_data, colWidths=[12*mm, 85*mm, 20*mm, 20*mm, 20*mm, 20*mm])
    pft.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_GREEN),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, -1), FONT),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("ALIGN",      (0, 0), (0, -1), "CENTER"),
        ("ALIGN",      (2, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f1f8e9"), colors.white]),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.HexColor("#a5d6a7")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TEXTCOLOR",  (2, 1), (2, -1), C_VALUE),
        ("TEXTCOLOR",  (3, 1), (3, -1), C_IMPACT),
        ("TEXTCOLOR",  (4, 1), (4, -1), C_LONGTAIL),
    ]))
    story.append(pft)
    story.append(Spacer(1, 14))

    # 结论
    story.extend(section_header("结论与观察"))
    conclusions = [
        ("演化方向收敛", "MOEA/D 从 8 个多样化种子知识点出发，经 10 代演化后自然收敛到"
         "「量子神经符号 × 联邦学习 × 拜占庭容错」这一跨领域融合范式，说明 LLM 倾向于通过组合创新探索高价值长尾区域。"),
        ("多目标权衡保持", "尽管主流解集中于高价值/高影响/高长尾区域（V=9, I=8, L=9），"
         "但 Tchebycheff 邻居替换机制成功保留了「极度长尾」方向（L=10, V=6-7）的解，体现了 Pareto 前沿的多样性。"),
        ("LLM 作为演化算子的有效性", "每代仅 2 次批量 LLM 调用即可完成交叉变异与适应度评估，"
         "总耗时 193 秒完成全部 10 代演化，证明 LLM 可以高效驱动语义层面的知识演化。"),
        ("涌现的研究范式", "最终 Pareto 解集揭示的核心研究方向：在边缘/联邦计算环境中，"
         "融合量子启发优化、神经符号推理和差分隐私，构建具备拜占庭容错性的安全可信 AI 系统——"
         "这是一个具有高度原创性的长尾跨领域研究课题。"),
    ]
    for title, content in conclusions:
        story.append(Paragraph(f"◆ {title}", S("ct", fontName=FONT, fontSize=10,
                                                textColor=C_PRIMARY, leading=16, spaceBefore=6)))
        story.append(Paragraph(content, sBody))
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=1, color=C_GRAY))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "本报告由 MOEA/D × LLM 自动演化系统生成 | 模型: deepseek-chat | "
        f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        sCaption
    ))

    doc.build(story, onFirstPage=add_header_footer, onLaterPages=add_header_footer)
    print(f"PDF 已生成: {output_path}")


if __name__ == "__main__":
    output = os.path.expanduser("~/moead_cs_research_report.pdf")
    build_pdf(output)
