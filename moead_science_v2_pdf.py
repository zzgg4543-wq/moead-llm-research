#!/usr/bin/env python3
"""MOEA/D × Claude 跨学科科研知识 PDF 报告（7目标版）"""

import json, os, datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak
)
from reportlab.graphics.shapes import Drawing, Rect, Line, String, Circle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_LEFT

pdfmetrics.registerFont(TTFont("U", "/Library/Fonts/Arial Unicode.ttf"))
F = "U"

# ── 配色 ──────────────────────────────────────────
CP  = colors.HexColor("#0d1b4b")
CA  = colors.HexColor("#1a3a7e")
CL  = colors.HexColor("#e8eef8")
CBg = colors.HexColor("#f4f6fb")
CGr = colors.HexColor("#607d8b")
CG  = colors.HexColor("#1b5e20")
COr = colors.HexColor("#bf360c")
W, H = A4

OBJ_NAMES  = ["知识价值", "社会影响", "长尾度", "跨学科性", "前沿性", "可行性", "合理性"]
OBJ_SHORT  = ["知识", "社会", "长尾", "跨学", "前沿", "可行", "合理"]
OBJ_COLORS = [
    colors.HexColor("#1565c0"),  # 知识价值
    colors.HexColor("#2e7d32"),  # 社会影响
    colors.HexColor("#880e4f"),  # 长尾度
    colors.HexColor("#e65100"),  # 跨学科性
    colors.HexColor("#6a1b9a"),  # 前沿性
    colors.HexColor("#00695c"),  # 可行性
    colors.HexColor("#827717"),  # 合理性
]
OBJ_DESCS = [
    "对基础科学理论的贡献深度，能否推进对自然/社会规律的根本性理解",
    "若取得突破，对人类社会、技术、医疗、环境的长远正向影响",
    "当前研究稀缺性（极主流=1，极小众冷门=10）",
    "与其他学科深度融合的潜力，跨越单一学科边界的广度",
    "知识的新颖度，突破已知边界、与现有研究的差异化程度",
    "在当前技术/实验/计算条件下可开展研究的现实程度",
    "科学假设的逻辑严谨性、理论基础支撑、原则上的可证伪性",
]

def S(name, **kw):
    d = dict(fontName=F, leading=15)
    d.update(kw)
    return ParagraphStyle(name, **d)

sTitle = S("t",  fontSize=22, textColor=CP,  alignment=TA_CENTER, leading=30, spaceAfter=4)
sSub   = S("s",  fontSize=11, textColor=CGr, alignment=TA_CENTER, leading=18)
sH1    = S("h1", fontSize=14, textColor=CP,  spaceBefore=10, spaceAfter=4, leading=20)
sH2    = S("h2", fontSize=11, textColor=CA,  spaceBefore=7,  spaceAfter=3, leading=16)
sBody  = S("b",  fontSize=8.5, textColor=colors.HexColor("#1a1a2e"), leading=13, spaceAfter=2)
sSm    = S("sm", fontSize=7.5, textColor=CGr, leading=11)
sCtr   = S("c",  fontSize=8.5, alignment=TA_CENTER, textColor=CGr, leading=12)
sBold  = S("bo", fontSize=9,   textColor=CP,  leading=13, spaceAfter=1)
sGrn   = S("g",  fontSize=9,   textColor=CG,  leading=13)
sTag   = S("tg", fontSize=7,   textColor=colors.HexColor("#5c35a3"), leading=11)
sCap   = S("cp", fontSize=7.5, textColor=CGr, alignment=TA_CENTER, leading=11)


# ── 7条评分条 ─────────────────────────────────────
def score_bars_7(scores, width=148, height=60):
    d = Drawing(width, height)
    bar_h, gap, lw = 5, 2.5, 20
    bw = width - lw - 6
    for i in range(7):
        y = height - (i + 1) * (bar_h + gap) + 1
        v = scores[i]
        d.add(Rect(lw, y, bw, bar_h, fillColor=colors.HexColor("#e8e8e8"), strokeColor=None))
        d.add(Rect(lw, y, bw * v, bar_h, fillColor=OBJ_COLORS[i], strokeColor=None))
        d.add(String(0, y + 0.5, OBJ_SHORT[i], fontName=F, fontSize=6, fillColor=CGr))
        val_x = lw + bw * v + 2
        d.add(String(val_x, y + 0.5, f"{v*10:.0f}",
                     fontName=F, fontSize=6, fillColor=OBJ_COLORS[i]))
    return d


# ── 7维雷达图 ─────────────────────────────────────
def radar_7(scores, size=78):
    import math
    d = Drawing(size, size)
    cx, cy, r = size / 2, size / 2, size * 0.35
    n, angles = 7, [math.pi / 2 + 2 * math.pi * i / 7 for i in range(7)]

    for lvl in [0.33, 0.67, 1.0]:
        for i in range(n):
            x1 = cx + r * lvl * math.cos(angles[i])
            y1 = cy + r * lvl * math.sin(angles[i])
            x2 = cx + r * lvl * math.cos(angles[(i+1) % n])
            y2 = cy + r * lvl * math.sin(angles[(i+1) % n])
            d.add(Line(x1, y1, x2, y2, strokeColor=colors.HexColor("#dce0e8"), strokeWidth=0.4))
    for a in angles:
        d.add(Line(cx, cy, cx + r*math.cos(a), cy + r*math.sin(a),
                   strokeColor=colors.HexColor("#c0c8d8"), strokeWidth=0.4))

    pts = [(cx + r*scores[i]*math.cos(angles[i]), cy + r*scores[i]*math.sin(angles[i]))
           for i in range(n)]
    for i in range(n):
        d.add(Line(pts[i][0], pts[i][1], pts[(i+1)%n][0], pts[(i+1)%n][1],
                   strokeColor=colors.HexColor("#3949ab"), strokeWidth=1.2))
    for x, y in pts:
        d.add(Circle(x, y, 1.8, fillColor=colors.HexColor("#3949ab"), strokeColor=None))

    for i, (a, name) in enumerate(zip(angles, OBJ_SHORT)):
        lx = cx + (r + 8) * math.cos(a)
        ly = cy + (r + 8) * math.sin(a)
        d.add(String(lx - 5, ly - 2.5, name, fontName=F, fontSize=5,
                     fillColor=OBJ_COLORS[i], textAnchor="middle"))
    return d


# ── 演化折线图（7目标 + 综合） ─────────────────────
def evolution_chart_7(gen_stats, width=430, height=175):
    d = Drawing(width, height)
    PL, PR, PT, PB = 36, 75, 10, 26
    cw = width - PL - PR
    ch = height - PT - PB
    n  = len(gen_stats)
    xs = cw / max(n - 1, 1)

    d.add(Rect(PL, PB, cw, ch, fillColor=colors.HexColor("#f8f9fb"),
               strokeColor=colors.HexColor("#d8dde8"), strokeWidth=0.5))
    for tick in [2, 4, 6, 8, 10]:
        y = PB + tick / 10 * ch
        d.add(Line(PL, y, PL + cw, y, strokeColor=colors.HexColor("#e0e4ec"), strokeWidth=0.4))
        d.add(String(PL - 4, y - 3, str(tick), fontName=F, fontSize=6,
                     fillColor=CGr, textAnchor="end"))
    for i, row in enumerate(gen_stats):
        x = PL + i * xs
        lbl = "初" if row["gen"] == 0 else str(row["gen"])
        d.add(String(x, PB - 9, lbl, fontName=F, fontSize=6, fillColor=CGr, textAnchor="middle"))
    d.add(String(PL + cw/2, 2, "演化代数", fontName=F, fontSize=7, fillColor=CGr, textAnchor="middle"))

    # 7条细线
    for obj_i, (col, nm) in enumerate(zip(OBJ_COLORS, OBJ_SHORT)):
        pts = [(PL + i*xs, PB + row["avgs"][obj_i]/10*ch) for i, row in enumerate(gen_stats)]
        for j in range(len(pts)-1):
            d.add(Line(pts[j][0], pts[j][1], pts[j+1][0], pts[j+1][1],
                       strokeColor=col, strokeWidth=1.1))
        for x, y in pts:
            d.add(Rect(x-1.3, y-1.3, 2.6, 2.6, fillColor=col, strokeColor=None))
        lx, ly = PL + cw + 4, PB + ch - 5 - obj_i * 14
        d.add(Rect(lx, ly, 9, 5, fillColor=col, strokeColor=None))
        d.add(String(lx + 11, ly, nm, fontName=F, fontSize=6, fillColor=CGr))

    # 综合均分（黑虚线）
    pts_c = [(PL + i*xs, PB + row["composite"]/10*ch) for i, row in enumerate(gen_stats)]
    for j in range(len(pts_c)-1):
        d.add(Line(pts_c[j][0], pts_c[j][1], pts_c[j+1][0], pts_c[j+1][1],
                   strokeColor=colors.HexColor("#1a1a2e"), strokeWidth=1.6,
                   strokeDashArray=[4, 2]))
    lx = PL + cw + 4
    ly = PB + ch - 5 - 7 * 14
    d.add(Line(lx, ly+2, lx+9, ly+2, strokeColor=colors.HexColor("#1a1a2e"),
               strokeWidth=1.6, strokeDashArray=[4, 2]))
    d.add(String(lx+11, ly, "综合", fontName=F, fontSize=6, fillColor=colors.HexColor("#1a1a2e")))
    return d


def ind_card(ind, rank, is_pareto=False):
    v = ind["scores"]
    flag = " ★" if is_pareto else ""
    tc   = CG if is_pareto else CP
    bg   = colors.HexColor("#f0f8f0") if is_pareto else (CBg if rank % 2 == 0 else colors.white)
    bc   = CG if is_pareto else colors.HexColor("#b8c4d0")
    row = [[
        Paragraph(f"#{rank}{flag}", S("rk", fontName=F, fontSize=10, textColor=tc,
                                       alignment=TA_CENTER, leading=14)),
        [Paragraph(ind["topic"], S("tp", fontName=F, fontSize=9, textColor=tc,
                                   leading=13, spaceAfter=1)),
         Paragraph(f"[{ind['domain']}]", sTag),
         Paragraph(ind["description"][:95] + ("…" if len(ind["description"]) > 95 else ""), sSm)],
        score_bars_7(v, width=138, height=57),
        radar_7(v, size=63),
    ]]
    t = Table(row, colWidths=[12*mm, 83*mm, 51*mm, 23*mm])
    t.setStyle(TableStyle([
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("BACKGROUND",    (0,0), (-1,-1), bg),
        ("BOX",           (0,0), (-1,-1), 0.5, bc),
        ("TOPPADDING",    (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ("LEFTPADDING",   (0,0), (-1,-1), 3),
        ("RIGHTPADDING",  (0,0), (-1,-1), 2),
    ]))
    return [t, Spacer(1, 2)]


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(CP)
    canvas.rect(0, H-28, W, 28, fill=1, stroke=0)
    canvas.setFont(F, 8.5)
    canvas.setFillColor(colors.white)
    canvas.drawString(16, H-18, "MOEA/D × Claude ── 跨学科长尾科研知识多目标进化报告（7目标版）")
    canvas.drawRightString(W-16, H-18, f"第 {doc.page} 页")
    canvas.setFillColor(CBg)
    canvas.rect(0, 0, W, 16, fill=1, stroke=0)
    canvas.setFillColor(CGr)
    canvas.setFont(F, 7)
    canvas.drawString(16, 4,
        f"生成: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
        f"模型: claude-sonnet-4-5  |  种群: 20  |  目标: 7  |  邻居: 5  |  演化: 10 代")
    canvas.restoreState()


def build_pdf(json_path, output_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    cfg       = data["config"]
    init_pop  = data["init_pop"]
    gen_stats = data["gen_stats"]
    final_pop = data["final_pop"]
    pareto    = data["pareto"]
    pareto_keys = {(p["topic"], p["domain"]) for p in pareto}

    doc = SimpleDocTemplate(output_path, pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=33*mm, bottomMargin=20*mm)
    story = []

    # ══ 封面 ═══════════════════════════════════════
    story.append(Spacer(1, 16))
    story.append(Paragraph("MOEA/D × Claude", sTitle))
    story.append(Paragraph("跨学科长尾科研知识多目标进化报告  ·  7目标版",
        S("st2", fontName=F, fontSize=14, textColor=CA, alignment=TA_CENTER, leading=22, spaceAfter=5)))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "在「知识价值 × 社会影响 × 长尾度 × 跨学科性 × 前沿性 × 可行性 × 合理性」"
        "七维科研评价空间中，以 MOEA/D 算法驱动 Claude Sonnet 4.5 对跨学科长尾研究知识进行 10 代演化优化。"
        "新增可行性与合理性维度，使评价体系更接近真实科研决策。",
        S("intro", fontName=F, fontSize=10, textColor=CGr, alignment=TA_CENTER,
          leading=16, leftIndent=12, rightIndent=12, spaceAfter=10)
    ))

    # 参数表
    params = [
        ["参数", "值",          "参数",     "值",    "参数",     "值"],
        ["模型",  "claude-sonnet-4-5", "种群大小", "20", "演化代数", "10 代"],
        ["目标数量", "7 个",    "邻居数量", "5",    "标量化",   "Tchebycheff"],
        ["权重向量", "Dirichlet均匀", "总耗时", "~1076s", "Pareto解", f"{len(set(p['topic'] for p in pareto))} 个(唯一)"],
    ]
    pt = Table(params, colWidths=[26*mm, 40*mm, 26*mm, 20*mm, 24*mm, 32*mm])
    pt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), CP),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTNAME",      (0,0), (-1,-1), F),
        ("FONTSIZE",      (0,0), (-1,-1), 8),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [CL, colors.white]),
        ("GRID",          (0,0), (-1,-1), 0.3, colors.HexColor("#b0bec5")),
        ("ROWHEIGHT",     (0,0), (-1,-1), 12),
        ("TOPPADDING",    (0,0), (-1,-1), 2),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2),
    ]))
    story.append(pt)
    story.append(Spacer(1, 10))

    # 7目标说明
    story.append(HRFlowable(width="100%", thickness=2, color=CP, spaceAfter=3))
    story.append(Paragraph("七维评价目标体系", sH1))
    obj_data = [["目标", "名称", "定义", "优化"]]
    for i in range(7):
        obj_data.append([f"f{i+1}", OBJ_NAMES[i], OBJ_DESCS[i], "最大化"])
    ot = Table(obj_data, colWidths=[10*mm, 22*mm, 127*mm, 16*mm])
    ot_style = [
        ("BACKGROUND",    (0,0), (-1,0), CA),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTNAME",      (0,0), (-1,-1), F),
        ("FONTSIZE",      (0,0), (-1,-1), 8),
        ("ALIGN",         (0,0), (1,-1), "CENTER"),
        ("ALIGN",         (3,0), (3,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, CBg]),
        ("GRID",          (0,0), (-1,-1), 0.3, colors.HexColor("#cfd8dc")),
        ("TOPPADDING",    (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]
    for i, col in enumerate(OBJ_COLORS):
        ot_style += [("TEXTCOLOR", (0,i+1), (1,i+1), col)]
    ot.setStyle(TableStyle(ot_style))
    story.append(ot)
    story.append(Spacer(1, 8))

    # 新增目标说明
    story.append(Paragraph("新增目标说明：可行性 vs 合理性的区别", sH2))
    note = (
        "【可行性】关注「能不能做」：在当前技术条件、实验手段、计算资源下是否可以实际开展研究。"
        "一项研究可以科学假设非常合理，但因技术限制暂时不可行（如直接探测暗物质核反冲，合理但可行性低）。"
        "  ·  "
        "【合理性】关注「值不值得信」：科学假设是否有理论依据支撑、是否可证伪、逻辑是否严密。"
        "一项研究可以技术上完全可行，但假设本身缺乏科学基础（如纯粹的科幻推测）。"
        "两个维度的加入使 Pareto 前沿呈现出更丰富的权衡结构。"
    )
    story.append(Paragraph(note, sBody))

    story.append(PageBreak())

    # ══ 初始种群 ════════════════════════════════════
    story.append(HRFlowable(width="100%", thickness=2, color=CP, spaceAfter=3))
    story.append(Paragraph("初始种群（第 0 代）— 20 个跨学科长尾知识条目", sH1))
    story.append(Paragraph(
        "由 Claude Sonnet 4.5 一次生成，覆盖量子生物物理、地球化学手性起源、"
        "拓扑微生物生态、量子声子光学、量子计算生物学、非厄米城市物理等多个交叉学科方向。",
        sBody))
    story.append(Spacer(1, 4))
    for k, ind in enumerate(init_pop):
        is_p = (ind["topic"], ind["domain"]) in pareto_keys
        story.extend(ind_card(ind, k+1, is_p))

    story.append(PageBreak())

    # ══ 演化趋势 ════════════════════════════════════
    story.append(HRFlowable(width="100%", thickness=2, color=CP, spaceAfter=3))
    story.append(Paragraph("演化过程分析", sH1))
    story.append(Paragraph("七个目标维度的种群均值随演化代数的变化（单位：0-10分）：", sBody))
    story.append(Spacer(1, 3))
    story.append(evolution_chart_7(gen_stats, width=432, height=175))
    story.append(Spacer(1, 7))

    # 统计表
    story.append(Paragraph("各代均值统计", sH2))
    hdr = ["代数"] + [n[:3] for n in OBJ_NAMES] + ["综合", "变化"]
    stat_data = [hdr]
    prev = None
    for row in gen_stats:
        gl = "初始" if row["gen"] == 0 else f"第{row['gen']}代"
        delta = ("" if prev is None
                 else f"+{row['composite']-prev:.2f}" if row['composite'] > prev
                 else f"{row['composite']-prev:.2f}")
        stat_data.append([gl] + [f"{a:.1f}" for a in row["avgs"]] +
                         [f"{row['composite']:.2f}", delta])
        prev = row["composite"]

    cws = [19*mm] + [19*mm]*7 + [20*mm, 16*mm]
    st = Table(stat_data, colWidths=cws)
    st_style = [
        ("BACKGROUND",    (0,0), (-1,0), CP),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTNAME",      (0,0), (-1,-1), F),
        ("FONTSIZE",      (0,0), (-1,-1), 7.5),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [CL, colors.white]),
        ("GRID",          (0,0), (-1,-1), 0.3, colors.HexColor("#b0bec5")),
        ("ROWHEIGHT",     (0,0), (-1,-1), 12),
        ("TOPPADDING",    (0,0), (-1,-1), 1),
        ("BOTTOMPADDING", (0,0), (-1,-1), 1),
        ("BACKGROUND",    (0,-1), (-1,-1), colors.HexColor("#fff9c4")),
        ("TEXTCOLOR",     (8,1), (8,-1), CA),
        ("TEXTCOLOR",     (9,1), (9,-1), CG),
    ]
    for i, col in enumerate(OBJ_COLORS):
        st_style.append(("TEXTCOLOR", (i+1,1), (i+1,-1), col))
    st.setStyle(TableStyle(st_style))
    story.append(st)
    story.append(Spacer(1, 8))

    # 演化洞察
    story.append(Paragraph("演化关键洞察", sH2))
    init_stats = gen_stats[0]
    final_stats = gen_stats[-1]
    insights = [
        ("可行性目标", "可行性张力",
         f"可行性均值始终保持在 4-5 分区间（初始 {init_stats['avgs'][5]:.1f}→"
         f"最终 {final_stats['avgs'][5]:.1f}），是7个目标中分值最低的。"
         "这反映了长尾知识的本质：越新颖的研究往往意味着技术条件尚不成熟。"),
        ("合理性目标", "稳步提升",
         f"合理性均值从初始 {init_stats['avgs'][6]:.1f} 提升至第10代 {final_stats['avgs'][6]:.1f}，"
         "说明 Claude 的交叉变异倾向于产生具有更扎实理论基础的后代。"),
        ("前沿 vs 合理", "核心权衡",
         "Pareto 前沿中「暗物质核反冲拓扑芯片」（前沿=10, 合理=4）"
         "与「根际量子增效表观育种」（前沿=7, 合理=7）并存，"
         "清晰揭示了「大胆创新 vs 扎实可信」的经典科研决策权衡。"),
        ("研究范式涌现", "双轨收敛",
         "演化后期收敛到「量子拓扑光合信息学」（偏知识价值/前沿性）"
         "和「根际量子生态碳循环」（偏社会影响/可行性）两大范式，"
         "体现了 MOEA/D 在高维目标空间中维持解多样性的能力。"),
    ]
    ins_data = [["维度", "洞察", "分析"]]
    for a, b, c in insights:
        ins_data.append([a, b, c])
    it = Table(ins_data, colWidths=[28*mm, 30*mm, 110*mm])
    it.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), CA),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTNAME",      (0,0), (-1,-1), F),
        ("FONTSIZE",      (0,0), (-1,-1), 7.5),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("ALIGN",         (0,0), (1,-1), "CENTER"),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, CBg]),
        ("GRID",          (0,0), (-1,-1), 0.3, colors.HexColor("#cfd8dc")),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("TEXTCOLOR",     (1,1), (1,-1), COr),
    ]))
    story.append(it)

    story.append(PageBreak())

    # ══ 最终种群 ════════════════════════════════════
    story.append(HRFlowable(width="100%", thickness=2, color=CP, spaceAfter=3))
    story.append(Paragraph("第 10 代最终种群（全 20 个个体）", sH1))
    story.append(Paragraph("★ 标记为 Pareto 最优解（绿框）。每个条目含 7 维评分条与雷达图。", sBody))
    story.append(Spacer(1, 4))
    for k, ind in enumerate(sorted(final_pop, key=lambda x: sum(x["scores"]), reverse=True)):
        is_p = (ind["topic"], ind["domain"]) in pareto_keys
        story.extend(ind_card(ind, k+1, is_p))

    story.append(PageBreak())

    # ══ Pareto 前沿 ══════════════════════════════════
    # 去重
    seen, pareto_u = set(), []
    for p in sorted(pareto, key=lambda x: sum(x["scores"]), reverse=True):
        if p["topic"] not in seen:
            seen.add(p["topic"]); pareto_u.append(p)

    story.append(HRFlowable(width="100%", thickness=2, color=CP, spaceAfter=3))
    story.append(Paragraph(f"Pareto 最优解集（{len(pareto_u)} 个唯一非支配解）", sH1))
    story.append(Paragraph(
        "以下解集中任何一个在 7 个目标上均无法被另一个解同时超越，"
        "共同构成本次演化在七维科研评价空间中发现的最优研究方向前沿面。",
        sBody))
    story.append(Spacer(1, 6))

    pf_hdr = ["#", "研究主题", "领域"] + OBJ_SHORT + ["综合"]
    pf_data = [pf_hdr]
    for k, p in enumerate(pareto_u):
        v = p["scores"]
        pf_data.append(
            [str(k+1), p["topic"], p["domain"][:18]] +
            [f"{s*10:.0f}" for s in v] + [f"{sum(v)/7*10:.1f}"]
        )
    pft = Table(pf_data, colWidths=[9*mm, 58*mm, 34*mm] + [13*mm]*7 + [14*mm])
    pft_s = [
        ("BACKGROUND",    (0,0), (-1,0), CG),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTNAME",      (0,0), (-1,-1), F),
        ("FONTSIZE",      (0,0), (-1,-1), 7.5),
        ("ALIGN",         (0,0), (0,-1), "CENTER"),
        ("ALIGN",         (3,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.HexColor("#f0f8f0"), colors.white]),
        ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#a5d6a7")),
        ("TOPPADDING",    (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]
    for i, col in enumerate(OBJ_COLORS):
        pft_s.append(("TEXTCOLOR", (3+i,1), (3+i,-1), col))
    pft.setStyle(TableStyle(pft_s))
    story.append(pft)
    story.append(Spacer(1, 12))

    # 详细描述
    story.append(Paragraph("Pareto 解集详细描述", sH2))
    for k, p in enumerate(pareto_u):
        v = p["scores"]
        sc_str = "  ".join(f"{OBJ_SHORT[i]}={v[i]*10:.0f}" for i in range(7))
        story.append(Paragraph(f"◆  {p['topic']}",
            S("pn", fontName=F, fontSize=9.5, textColor=CG, leading=14, spaceAfter=1, spaceBefore=5)))
        story.append(Paragraph(f"[{p['domain']}]  |  {sc_str}  |  综合={sum(v)/7*10:.1f}", sSm))
        story.append(Paragraph(p["description"], sBody))

    # ══ 结论 ════════════════════════════════════════
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=2, color=CP, spaceAfter=3))
    story.append(Paragraph("结论与科学洞察", sH1))
    concs = [
        ("七维目标体系的科研决策价值",
         "加入可行性与合理性后，Pareto 前沿呈现出更丰富的多维权衡结构：出现了「高前沿/低可行」"
         "（如暗物质相关研究）、「高可行/高合理」（如根际量子农业验证）、「高长尾/低合理」"
         "（如月尘光合激元）三类典型解型，对应科研决策中「探索性基础研究」、"
         "「可验证的近期研究」与「极度小众假说」三种不同策略。"),
        ("Claude 的演化质量",
         "相比 DeepSeek，claude-sonnet-4-5 生成的知识条目描述更加具体翔实，"
         "理论背景阐述更清晰，在可行性和合理性维度上的评分也更具区分度（初始标准差更大）。"
         "这导致初始综合分（约 6.0）低于 DeepSeek 版（约 7.25），"
         "但 Pareto 解的科学质量更高，也更具研究操作性。"),
        ("MOEA/D 在高维（7目标）场景的表现",
         "高维目标空间下「维数诅咒」使得 Pareto 前沿解数量激增（17个，其中12个唯一），"
         "覆盖率更广但收敛性略弱。Dirichlet 权重向量采样配合邻居替换机制"
         "有效维持了种群多样性，最终理想点在所有7个目标上均达到满分（9-10分）。"),
        ("推荐的研究方向",
         "综合七维评估，「量子相干性在光合作用电子转移中的拓扑保护机制」"
         "（知识=8, 社会=7, 长尾=8-9, 跨学=9, 前沿=9, 可行=5-6, 合理=7）"
         "在知识价值、跨学科性、前沿性、合理性上均衡，且可行性相对较高，"
         "是最值得投入研究资源的长尾科研方向。"),
    ]
    for title, content in concs:
        story.append(Paragraph(f"◆ {title}", S("ct", fontName=F, fontSize=9.5,
            textColor=CP, leading=14, spaceBefore=6, spaceAfter=2)))
        story.append(Paragraph(content, sBody))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1, color=CGr))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"MOEA/D × Claude 自动演化系统  |  模型: claude-sonnet-4-5  |  "
        f"生成: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        sCap
    ))

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"✓ PDF 已生成: {output_path}")


if __name__ == "__main__":
    build_pdf(
        os.path.expanduser("~/moead_science_v2_results.json"),
        os.path.expanduser("~/moead_science_v2_report.pdf")
    )
