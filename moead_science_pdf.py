#!/usr/bin/env python3
"""MOEA/D × LLM 跨学科科研知识 PDF 报告生成器（5目标版）"""

import json, os, datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Rect, Line, String, Circle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# ── 字体 ──────────────────────────────────────────
pdfmetrics.registerFont(TTFont("U", "/Library/Fonts/Arial Unicode.ttf"))
F = "U"

# ── 配色 ──────────────────────────────────────────
CP   = colors.HexColor("#1a237e")   # 深蓝（主色）
CA   = colors.HexColor("#283593")   # 蓝（副色）
CL   = colors.HexColor("#e8eaf6")   # 浅蓝背景
COr  = colors.HexColor("#bf360c")   # 橙红
CG   = colors.HexColor("#1b5e20")   # 深绿
CGr  = colors.HexColor("#546e7a")   # 灰
CBg  = colors.HexColor("#f5f7fa")
# 5目标颜色
OBJ_COLORS = [
    colors.HexColor("#1565c0"),  # 知识价值 - 蓝
    colors.HexColor("#2e7d32"),  # 社会影响 - 绿
    colors.HexColor("#ad1457"),  # 长尾度   - 紫红
    colors.HexColor("#e65100"),  # 跨学科性 - 橙
    colors.HexColor("#6a1b9a"),  # 前沿性   - 紫
]
OBJ_NAMES = ["知识价值", "社会影响", "长尾度", "跨学科性", "前沿性"]
OBJ_SHORT = ["知识", "社会", "长尾", "跨学", "前沿"]

W, H = A4

# ── 样式 ──────────────────────────────────────────
def S(name, **kw):
    d = dict(fontName=F, leading=15)
    d.update(kw)
    return ParagraphStyle(name, **d)

sTitle  = S("t",  fontSize=24, textColor=CP, alignment=TA_CENTER, leading=32, spaceAfter=4)
sSub    = S("s",  fontSize=12, textColor=CGr, alignment=TA_CENTER, leading=18)
sH1     = S("h1", fontSize=15, textColor=CP, spaceBefore=12, spaceAfter=5, leading=20)
sH2     = S("h2", fontSize=11, textColor=CA, spaceBefore=8,  spaceAfter=3, leading=16)
sBody   = S("b",  fontSize=9,  textColor=colors.HexColor("#212121"), leading=14, spaceAfter=2)
sSmall  = S("sm", fontSize=8,  textColor=CGr, leading=12)
sCtr    = S("c",  fontSize=9,  alignment=TA_CENTER, textColor=CGr, leading=13)
sBold   = S("bo", fontSize=9.5, textColor=CP, leading=14, spaceAfter=1)
sGreen  = S("g",  fontSize=9.5, textColor=CG, leading=14)
sTag    = S("tg", fontSize=7.5, textColor=colors.HexColor("#5c35a3"), leading=12)
sCap    = S("cp", fontSize=7.5, textColor=CGr, alignment=TA_CENTER, leading=11)


# ── 组件 ──────────────────────────────────────────
def score_bars_5(scores, width=155, height=50):
    """5条彩色评分条"""
    d = Drawing(width, height)
    bar_h = 6
    gap   = 3
    lw    = 22
    bw    = width - lw - 8
    for idx in range(5):
        y = height - (idx + 1) * (bar_h + gap) + 1
        v = scores[idx]
        d.add(Rect(lw, y, bw, bar_h, fillColor=colors.HexColor("#eeeeee"), strokeColor=None))
        d.add(Rect(lw, y, bw * v, bar_h, fillColor=OBJ_COLORS[idx], strokeColor=None))
        d.add(String(0, y + 1, OBJ_SHORT[idx], fontName=F, fontSize=6.5, fillColor=CGr))
        d.add(String(lw + bw * v + 2, y + 1, f"{v*10:.0f}",
                     fontName=F, fontSize=6.5, fillColor=OBJ_COLORS[idx]))
    return d


def radar_drawing(scores, size=90):
    """五维雷达图"""
    import math
    d = Drawing(size, size)
    cx, cy, r = size / 2, size / 2, size * 0.38
    n = 5
    angles = [math.pi / 2 + 2 * math.pi * i / n for i in range(n)]

    # 背景网格
    for lvl in [0.25, 0.5, 0.75, 1.0]:
        pts = []
        for a in angles:
            pts.extend([cx + r * lvl * math.cos(a), cy + r * lvl * math.sin(a)])
        # draw polygon
        for i in range(n):
            x1 = cx + r * lvl * math.cos(angles[i])
            y1 = cy + r * lvl * math.sin(angles[i])
            x2 = cx + r * lvl * math.cos(angles[(i + 1) % n])
            y2 = cy + r * lvl * math.sin(angles[(i + 1) % n])
            d.add(Line(x1, y1, x2, y2, strokeColor=colors.HexColor("#e0e0e0"), strokeWidth=0.4))

    # 轴线
    for a in angles:
        d.add(Line(cx, cy, cx + r * math.cos(a), cy + r * math.sin(a),
                   strokeColor=colors.HexColor("#bdbdbd"), strokeWidth=0.5))

    # 数据多边形
    data_pts = [(cx + r * scores[i] * math.cos(angles[i]),
                 cy + r * scores[i] * math.sin(angles[i])) for i in range(n)]
    for i in range(n):
        x1, y1 = data_pts[i]
        x2, y2 = data_pts[(i + 1) % n]
        d.add(Line(x1, y1, x2, y2, strokeColor=colors.HexColor("#3949ab"), strokeWidth=1.2))
    for x, y in data_pts:
        d.add(Circle(x, y, 2, fillColor=colors.HexColor("#3949ab"), strokeColor=None))

    # 标签
    for i, (a, name) in enumerate(zip(angles, OBJ_SHORT)):
        lx = cx + (r + 9) * math.cos(a)
        ly = cy + (r + 9) * math.sin(a)
        d.add(String(lx - 6, ly - 3, name, fontName=F, fontSize=5.5,
                     fillColor=OBJ_COLORS[i], textAnchor="middle"))
    return d


def evolution_chart_5(gen_stats, width=430, height=170):
    """5条折线演化趋势图"""
    d = Drawing(width, height)
    PL, PR, PT, PB = 38, 70, 12, 28
    cw = width - PL - PR
    ch = height - PT - PB
    n  = len(gen_stats)
    xs = cw / max(n - 1, 1)
    ymin, ymax = 0.0, 10.0
    yr = ymax - ymin

    # 背景
    d.add(Rect(PL, PB, cw, ch, fillColor=colors.HexColor("#f8f9fa"),
               strokeColor=colors.HexColor("#dee2e6"), strokeWidth=0.5))
    # 网格
    for tick in [2, 4, 6, 8, 10]:
        y = PB + tick / yr * ch
        d.add(Line(PL, y, PL + cw, y, strokeColor=colors.HexColor("#e0e0e0"), strokeWidth=0.4))
        d.add(String(PL - 4, y - 3, str(tick), fontName=F, fontSize=6.5,
                     fillColor=CGr, textAnchor="end"))
    # X轴
    for i, row in enumerate(gen_stats):
        x = PL + i * xs
        lbl = "初" if row["gen"] == 0 else str(row["gen"])
        d.add(String(x, PB - 9, lbl, fontName=F, fontSize=6.5, fillColor=CGr, textAnchor="middle"))
    d.add(String(PL + cw / 2, 2, "演化代数", fontName=F, fontSize=7, fillColor=CGr, textAnchor="middle"))

    # 5条折线
    for obj_i, (col, name) in enumerate(zip(OBJ_COLORS, OBJ_SHORT)):
        pts = []
        for i, row in enumerate(gen_stats):
            x = PL + i * xs
            y = PB + row["avgs"][obj_i] / yr * ch
            pts.append((x, y))
        for j in range(len(pts) - 1):
            d.add(Line(pts[j][0], pts[j][1], pts[j+1][0], pts[j+1][1],
                       strokeColor=col, strokeWidth=1.4))
        for x, y in pts:
            d.add(Rect(x - 1.5, y - 1.5, 3, 3, fillColor=col, strokeColor=None))
        # 图例
        lx = PL + cw + 4
        ly = PB + ch - 6 - obj_i * 16
        d.add(Rect(lx, ly, 10, 5, fillColor=col, strokeColor=None))
        d.add(String(lx + 12, ly, name, fontName=F, fontSize=6.5, fillColor=CGr))

    # 综合均分（黑色虚线）
    pts_comp = []
    for i, row in enumerate(gen_stats):
        x = PL + i * xs
        y = PB + row["composite"] / yr * ch
        pts_comp.append((x, y))
    for j in range(len(pts_comp) - 1):
        d.add(Line(pts_comp[j][0], pts_comp[j][1], pts_comp[j+1][0], pts_comp[j+1][1],
                   strokeColor=colors.HexColor("#212121"), strokeWidth=1.6,
                   strokeDashArray=[3, 2]))
    lx = PL + cw + 4
    ly = PB + ch - 6 - 5 * 16
    d.add(Line(lx, ly + 2, lx + 10, ly + 2, strokeColor=colors.HexColor("#212121"),
               strokeWidth=1.6, strokeDashArray=[3, 2]))
    d.add(String(lx + 12, ly, "综合", fontName=F, fontSize=6.5, fillColor=colors.HexColor("#212121")))
    return d


def sec_header(text):
    return [
        HRFlowable(width="100%", thickness=2, color=CP, spaceAfter=3),
        Paragraph(text, sH1),
    ]


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(CP)
    canvas.rect(0, H - 28, W, 28, fill=1, stroke=0)
    canvas.setFont(F, 9)
    canvas.setFillColor(colors.white)
    canvas.drawString(18, H - 18, "MOEA/D × LLM  ──  跨学科长尾科研知识多目标进化报告（扩展版）")
    canvas.drawRightString(W - 18, H - 18, f"第 {doc.page} 页")
    canvas.setFillColor(CBg)
    canvas.rect(0, 0, W, 18, fill=1, stroke=0)
    canvas.setFillColor(CGr)
    canvas.setFont(F, 7.5)
    canvas.drawString(18, 5,
        f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
        f"模型: deepseek-chat  |  种群: 15  |  目标: 5  |  邻居: 5  |  演化: 10 代")
    canvas.restoreState()


def ind_card(ind, rank, is_pareto=False):
    """个体展示卡片（含雷达图）"""
    v = ind["scores"]
    pareto_flag = " ★" if is_pareto else ""
    tc = CG if is_pareto else CP
    row = [[
        Paragraph(f"#{rank}{pareto_flag}", S("rk", fontName=F, fontSize=11,
            textColor=tc, alignment=TA_CENTER, leading=15)),
        [
            Paragraph(ind["topic"],
                      S("tp2", fontName=F, fontSize=9.5, textColor=tc, leading=14, spaceAfter=1)),
            Paragraph(f"[{ind['domain']}]", sTag),
            Paragraph(ind["description"][:90] + ("…" if len(ind["description"]) > 90 else ""), sSmall),
        ],
        score_bars_5(v, width=140, height=48),
        radar_drawing(v, size=60),
    ]]
    bg = colors.HexColor("#f1f8e9") if is_pareto else (CBg if rank % 2 == 0 else colors.white)
    border_c = CG if is_pareto else colors.HexColor("#b0bec5")
    t = Table(row, colWidths=[13*mm, 84*mm, 52*mm, 22*mm])
    t.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND",    (0, 0), (-1, -1), bg),
        ("BOX",           (0, 0), (-1, -1), 0.5, border_c),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 3),
    ]))
    return [t, Spacer(1, 2)]


# ══════════════════════════════════════════════════
#  主构建
# ══════════════════════════════════════════════════
def build_pdf(json_path, output_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    cfg        = data["config"]
    init_pop   = data["init_pop"]
    gen_stats  = data["gen_stats"]
    final_pop  = data["final_pop"]
    pareto     = data["pareto"]
    pareto_set = {(p["topic"], p["domain"]) for p in pareto}

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=16*mm, rightMargin=16*mm,
        topMargin=34*mm, bottomMargin=22*mm,
    )
    story = []

    # ══ 封面 ══════════════════════════════════════
    story.append(Spacer(1, 20))
    story.append(Paragraph("MOEA/D × LLM", sTitle))
    story.append(Paragraph("跨学科长尾科研知识多目标进化报告", S("st2",
        fontName=F, fontSize=16, textColor=CA, alignment=TA_CENTER, leading=24, spaceAfter=6)))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "本报告基于「基于分解的多目标演化算法（MOEA/D）」与「大语言模型（LLM）」协同驱动，"
        "在五维科研评价目标空间中对跨学科长尾研究知识进行 10 代演化优化，"
        "探索具有高知识价值、深远社会影响、强跨学科潜力的长尾前沿研究方向。",
        S("intro2", fontName=F, fontSize=10.5, textColor=CGr, alignment=TA_CENTER,
          leading=17, leftIndent=15, rightIndent=15, spaceAfter=10)
    ))

    # 参数卡片
    params = [
        ["参数", "值", "参数", "值", "参数", "值"],
        ["演化模型",  "deepseek-chat", "种群大小",   "15",  "演化代数", "10 代"],
        ["目标数量",  "5 个",          "邻居数量",   "5",   "标量化",  "Tchebycheff"],
        ["总LLM调用", "约 22 次",       "总耗时",    "~464s", "Pareto解", f"{len(pareto)} 个"],
    ]
    pt = Table(params, colWidths=[28*mm, 40*mm, 28*mm, 22*mm, 24*mm, 28*mm])
    pt.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), CP),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, -1), F),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [CL, colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#b0bec5")),
        ("ROWHEIGHT",     (0, 0), (-1, -1), 13),
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(pt)
    story.append(Spacer(1, 12))

    # 5目标说明
    story.extend(sec_header("五维评价目标"))
    obj_data = [["目标", "名称", "说明", "优化方向"]]
    obj_descs = [
        ("f1", "知识价值", "对基础科学理论的贡献深度，能否推进对自然/社会规律的根本性理解", "最大化"),
        ("f2", "社会影响", "若研究取得突破，对人类社会、技术、环境的长远正向影响程度",     "最大化"),
        ("f3", "长尾度",   "当前研究的稀缺性和小众程度（0=极主流，10=极小众罕见）",        "最大化"),
        ("f4", "跨学科性", "与其他学科深度融合的潜力，跨越单一学科边界的广度与深度",        "最大化"),
        ("f5", "前沿性",   "知识的新颖度，突破已知科学边界、与现有研究的差异化程度",        "最大化"),
    ]
    for row in obj_descs:
        obj_data.append(list(row))

    ot = Table(obj_data, colWidths=[12*mm, 22*mm, 115*mm, 20*mm])
    ot.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), CA),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, -1), F),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("ALIGN",         (0, 0), (1, -1), "CENTER"),
        ("ALIGN",         (3, 0), (3, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, CBg]),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#cfd8dc")),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    # 目标颜色
    for i, col in enumerate(OBJ_COLORS):
        ot.setStyle(TableStyle([("TEXTCOLOR", (0, i+1), (1, i+1), col)]))
    story.append(ot)

    story.append(PageBreak())

    # ══ 初始种群 ══════════════════════════════════
    story.extend(sec_header("初始种群（第 0 代） — 15 个跨学科长尾知识条目"))
    story.append(Paragraph(
        "由 LLM 一次性生成，覆盖量子生物学、天体化学、材料科学、神经工程、"
        "生物计算、计算社会科学、极地地球化学、生物物理学等多个交叉学科方向。"
        "★ 标记的条目最终进入 Pareto 最优解集。",
        sBody
    ))
    story.append(Spacer(1, 5))
    for k, ind in enumerate(init_pop):
        is_p = (ind["topic"], ind["domain"]) in pareto_set
        story.extend(ind_card(ind, k + 1, is_p))

    story.append(PageBreak())

    # ══ 演化趋势 ══════════════════════════════════
    story.extend(sec_header("演化过程分析"))
    story.append(Paragraph("五个目标维度的种群均值随演化代数的变化（单位：0-10分）：", sBody))
    story.append(Spacer(1, 4))
    story.append(evolution_chart_5(gen_stats, width=430, height=170))
    story.append(Spacer(1, 8))

    # 数值表
    story.append(Paragraph("各代均值统计", sH2))
    stat_header = ["代数"] + [f"{n[:3]}" for n in OBJ_NAMES] + ["综合均分", "变化"]
    stat_data = [stat_header]
    prev = None
    for row in gen_stats:
        delta = ("" if prev is None
                 else f"+{row['composite']-prev:.2f}" if row['composite'] > prev
                 else f"{row['composite']-prev:.2f}")
        gen_label = "初始" if row["gen"] == 0 else f"第{row['gen']}代"
        stat_data.append(
            [gen_label] +
            [f"{a:.2f}" for a in row["avgs"]] +
            [f"{row['composite']:.2f}", delta]
        )
        prev = row["composite"]

    col_w = [20*mm] + [22*mm] * 5 + [24*mm, 18*mm]
    st = Table(stat_data, colWidths=col_w)
    style_cmds = [
        ("BACKGROUND",    (0, 0), (-1, 0), CP),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, -1), F),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [CL, colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#b0bec5")),
        ("ROWHEIGHT",     (0, 0), (-1, -1), 13),
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("BACKGROUND",    (0, -1), (-1, -1), colors.HexColor("#fff9c4")),
        ("TEXTCOLOR",     (6, 1), (6, -1), CA),
        ("TEXTCOLOR",     (7, 1), (7, -1), CG),
    ]
    for i, col in enumerate(OBJ_COLORS):
        style_cmds.append(("TEXTCOLOR", (i+1, 1), (i+1, -1), col))
    st.setStyle(TableStyle(style_cmds))
    story.append(st)
    story.append(Spacer(1, 10))

    # 演化洞察
    story.append(Paragraph("演化关键洞察", sH2))
    # 计算各代变化
    insights = [
        ("第 0 代 → 第 1 代", "最大单代跃升",
         f"综合均分从 {gen_stats[0]['composite']:.2f} 跃升至 {gen_stats[1]['composite']:.2f}，"
         f"长尾度（{gen_stats[0]['avgs'][2]:.1f}→{gen_stats[1]['avgs'][2]:.1f}）与跨学科性大幅提升，"
         "LLM 迅速发现量子拓扑与神经工程的交叉融合方向。"),
        ("前沿性目标", "持续单调上升",
         f"前沿性均值从初始 {gen_stats[0]['avgs'][4]:.1f} 稳步提升至第10代 {gen_stats[-1]['avgs'][4]:.1f}，"
         "是5个目标中增长最稳定的，体现了 LLM 持续探索新颖知识组合的能力。"),
        ("社会影响目标", "张力最大的目标",
         "社会影响与知识价值/前沿性呈现明显的 Pareto 张力——高理论价值的宇宙量子研究社会影响低，"
         "而高社会影响的研究（如全球量子健康网络）理论深度相对有限，体现了真实的多目标权衡。"),
        ("Pareto 前沿多样性", "11 个非支配解",
         f"最终 Pareto 前沿共 {len(pareto)} 个解，覆盖从「极高知识价值/极低社会影响」"
         "到「极高社会影响/较低长尾度」的完整谱系，说明 MOEA/D 成功维持了解的多样性。"),
    ]
    ins_data = [["代次/维度", "现象", "分析"]]
    for a, b, c in insights:
        ins_data.append([a, b, c])
    it = Table(ins_data, colWidths=[30*mm, 30*mm, 110*mm])
    it.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), CA),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, -1), F),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("ALIGN",         (0, 0), (1, -1), "CENTER"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, CBg]),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#cfd8dc")),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TEXTCOLOR",     (1, 1), (1, -1), COr),
    ]))
    story.append(it)

    story.append(PageBreak())

    # ══ 最终种群 ══════════════════════════════════
    story.extend(sec_header("第 10 代最终种群（全 15 个个体）"))
    story.append(Paragraph(
        "★ 标记为 Pareto 最优解（绿色边框）。每个条目附有五维评分条和雷达图。",
        sBody
    ))
    story.append(Spacer(1, 5))
    for k, ind in enumerate(sorted(final_pop, key=lambda x: sum(x["scores"]), reverse=True)):
        is_p = (ind["topic"], ind["domain"]) in pareto_set
        story.extend(ind_card(ind, k + 1, is_p))

    story.append(PageBreak())

    # ══ Pareto 前沿 ══════════════════════════════
    story.extend(sec_header(f"Pareto 最优解集（共 {len(pareto)} 个非支配解）"))
    story.append(Paragraph(
        "以下解集中任何一个无法在所有五个目标上同时被另一个解超越，"
        "共同构成本次演化发现的最优科研方向前沿面。",
        sBody
    ))
    story.append(Spacer(1, 8))

    # 汇总表
    pf_hdr = ["#", "研究主题", "领域"] + OBJ_SHORT + ["综合"]
    pf_data = [pf_hdr]
    pareto_sorted = sorted(pareto, key=lambda x: sum(x["scores"]), reverse=True)
    # 去重
    seen = set()
    pareto_unique = []
    for p in pareto_sorted:
        key = p["topic"]
        if key not in seen:
            seen.add(key)
            pareto_unique.append(p)

    for k, p in enumerate(pareto_unique):
        v = p["scores"]
        pf_data.append(
            [str(k+1), p["topic"], p["domain"][:20]] +
            [f"{s*10:.0f}" for s in v] +
            [f"{sum(v)/5*10:.1f}"]
        )

    pft = Table(pf_data, colWidths=[10*mm, 60*mm, 38*mm] + [16*mm]*5 + [16*mm])
    pft_style = [
        ("BACKGROUND",    (0, 0), (-1, 0), CG),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, -1), F),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("ALIGN",         (0, 0), (0, -1), "CENTER"),
        ("ALIGN",         (3, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f1f8e9"), colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#a5d6a7")),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for i, col in enumerate(OBJ_COLORS):
        pft_style.append(("TEXTCOLOR", (3+i, 1), (3+i, -1), col))
    pft.setStyle(TableStyle(pft_style))
    story.append(pft)
    story.append(Spacer(1, 14))

    # 详细描述
    story.append(Paragraph("Pareto 解集详细描述", sH2))
    story.append(Spacer(1, 4))
    for k, p in enumerate(pareto_unique):
        v = p["scores"]
        scores_str = "  ".join(
            f"{OBJ_SHORT[i]}={v[i]*10:.0f}"
            for i in range(5)
        )
        card = [
            Paragraph(f"◆  {p['topic']}", S("pn", fontName=F, fontSize=10,
                textColor=CG, leading=15, spaceAfter=1)),
            Paragraph(f"[{p['domain']}]", sTag),
            Paragraph(scores_str + f"  综合={sum(v)/5*10:.1f}", sSmall),
            Paragraph(p["description"], sBody),
            Spacer(1, 6),
        ]
        story.extend(card)

    # ══ 结论 ══════════════════════════════════════
    story.extend(sec_header("结论与科学洞察"))
    conclusions = [
        ("涌现的跨学科研究范式",
         "MOEA/D 从 15 个多样化种子出发，经 10 代演化后自然涌现出"
         "「量子拓扑 × 神经工程 × 天体生物学」的核心融合范式，"
         "以及「声学超材料 × 量子神经科学 × 宇宙探测」的新兴方向，"
         "体现了 LLM 在跨学科知识组合探索中的强大创造力。"),
        ("多目标 Pareto 张力分析",
         "五维目标间存在真实的权衡关系：知识价值与社会影响之间（基础研究 vs 应用价值）、"
         "长尾度与社会影响之间（越小众越难落地）、前沿性与可理解性之间均呈现典型的 Pareto 张力，"
         f"最终形成覆盖完整谱系的 {len(pareto_unique)} 个非支配解。"),
        ("MOEA/D 权重分解的有效性",
         "15 个子问题对应 15 种不同的研究价值取向，Tchebycheff 标量化确保了"
         "每个子问题在自己的偏好方向上找到局部最优解，"
         "邻居替换机制（T=5）保证了种群多样性与收敛速度的平衡。"),
        ("LLM 作为科研创新引擎",
         "本实验证明 LLM 可以作为语义层面的演化算子，"
         "在「知识生成—评估—交叉变异」的演化循环中产生真正的跨学科创新知识点，"
         "且仅需约 22 次 API 调用、464 秒即可完成 15 个个体的 10 代演化，"
         "展示了 LLM 驱动的科研知识探索系统的可行性。"),
    ]
    for title, content in conclusions:
        story.append(Paragraph(f"◆ {title}", S("ct2", fontName=F, fontSize=10,
            textColor=CP, leading=16, spaceBefore=6, spaceAfter=2)))
        story.append(Paragraph(content, sBody))

    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=1, color=CGr))
    story.append(Spacer(1, 5))
    story.append(Paragraph(
        f"MOEA/D × LLM 自动演化系统  |  模型: deepseek-chat  |  "
        f"生成: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        sCap
    ))

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"✓ PDF 已生成: {output_path}")


if __name__ == "__main__":
    json_path   = os.path.expanduser("~/moead_science_results.json")
    output_path = os.path.expanduser("~/moead_science_report.pdf")
    build_pdf(json_path, output_path)
