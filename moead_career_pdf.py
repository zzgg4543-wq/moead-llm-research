#!/usr/bin/env python3
"""
博后/职业选择 MOEA/D PDF 报告
包含：演化路线、分数变化曲线、Pareto 最优选项
"""

import json, os, math
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Rect, Line, String
from reportlab.graphics import renderPDF

# 尝试多个字体路径
for fontpath in ["/Library/Fonts/Arial Unicode.ttf", "/System/Library/Fonts/PingFang.ttc",
                 "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"]:
    if os.path.exists(fontpath):
        try:
            pdfmetrics.registerFont(TTFont("U", fontpath))
            break
        except Exception:
            continue
else:
    pdfmetrics.registerFont(TTFont("U", "Helvetica"))

F = "U"
W, H = A4
PW = W - 28*mm

CA  = colors.HexColor("#1565c0")
CB  = colors.HexColor("#2e7d32")
CR  = colors.HexColor("#c62828")
CAm = colors.HexColor("#e65100")
CDk = colors.HexColor("#212121")
CGy = colors.HexColor("#607d8b")
CLg = colors.HexColor("#eceff1")

OBJ_NAMES = ["契合","影响力","职业","资源","成长","可行","风险","Agent","格局","启发","上限"]
OBJ_FOCUS = [0, 7, 8, 9, 10]  # 契合 Agent 格局 启发 上限

def ps(name, fs, tc, ld=14, **kw):
    return ParagraphStyle(name, fontName=F, fontSize=fs, textColor=tc, leading=ld, **kw)

ST = {
    "title":   ps("t", 18, CA, 22, spaceAfter=3, spaceBefore=4, alignment=1),
    "sub":     ps("s",  9, CGy, 13, spaceAfter=2, alignment=1),
    "h2":      ps("h2", 9, colors.white, 13),
    "body":    ps("bd", 8.5, CDk, 13, spaceAfter=3),
    "caption": ps("cp", 7.5, CGy, 11, alignment=1, spaceAfter=3),
    "small":   ps("sm", 7, CGy, 10),
}

def sec(title, color=CA, h=16):
    d = Drawing(PW, h)
    d.add(Rect(0, 2, PW, h-2, fillColor=color, strokeColor=None))
    d.add(String(6, 5.5, title, fontName=F, fontSize=9.5, fillColor=colors.white))
    return d

def add_hf(canvas, doc):
    canvas.saveState()
    canvas.setFont(F, 7.5); canvas.setFillColor(CGy)
    canvas.drawString(14*mm, H-9*mm, "博后/职业选择 MOEA/D · 演化路线与分数变化报告")
    canvas.drawRightString(W-14*mm, H-9*mm, f"第 {doc.page} 页")
    canvas.line(14*mm, H-10*mm, W-14*mm, H-10*mm)
    canvas.restoreState()

# ── 演化分数曲线 ──────────────────────────────────────
def evo_score_chart(gen_stats, obj_indices, obj_labels, w=PW, h=110):
    d = Drawing(w, h)
    ox, oy = 35, 12; cw, ch = w-ox-18, h-oy-28
    mg = len(gen_stats) - 1
    cols = [CA, CB, CAm, colors.HexColor("#6a1b9a"), colors.HexColor("#00838f")]
    for yv in [2, 4, 6, 8, 10]:
        y = oy + (yv/10)*ch
        d.add(Line(ox, y, ox+cw, y, strokeColor=CLg, strokeWidth=0.5))
        d.add(String(ox-4, y-3, str(yv), fontName=F, fontSize=6, fillColor=CGy, textAnchor="end"))
    for ki, (idx, lbl) in enumerate(zip(obj_indices, obj_labels)):
        col = cols[ki % len(cols)]
        pts = [(ox + g*(cw/mg), oy + (gen_stats[g]["avgs"][idx]/10)*ch) for g in range(len(gen_stats))]
        for i in range(len(pts)-1):
            d.add(Line(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1], strokeColor=col, strokeWidth=2))
        d.add(String(pts[-1][0]+3, pts[-1][1]-3, lbl, fontName=F, fontSize=6, fillColor=col))
    for g in range(len(gen_stats)):
        x = ox + g*(cw/mg)
        d.add(String(x, oy-10, str(g), fontName=F, fontSize=6, fillColor=CGy, textAnchor="middle"))
    d.add(String(ox+cw/2, oy-18, "代数", fontName=F, fontSize=7, fillColor=CDk, textAnchor="middle"))
    return d

# ── 11维均值表 ────────────────────────────────────────
def gen_stats_table(gen_stats, w=PW):
    rows = [["代数"] + OBJ_NAMES]
    for gs in gen_stats:
        rows.append([str(gs["gen"])] + [f"{v:.1f}" for v in gs["avgs"]])
    tw = 14*mm + 11*12*mm
    colWidths = [14*mm] + [11*mm]*11
    t = Table(rows, colWidths=colWidths)
    t.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), F), ("FONTSIZE", (0,0), (-1,-1), 7),
        ("BACKGROUND", (0,0), (-1,0), CDk), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.3, CLg),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 2), ("BOTTOMPADDING", (0,0), (-1,-1), 2),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, CLg]),
    ]))
    return t

# ── 演化路线（各代 option_id 出现情况）────────────────
def route_table(log_dir, n_gens):
    rows = [["代数", "选项 ID", "机构/导师", "契合", "Agent", "格局", "启发", "上限"]]
    for g in range(n_gens + 1):
        fp = os.path.join(log_dir, f"gen_{g:02d}_population.json")
        if not os.path.exists(fp):
            continue
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        pop = data.get("population", [])
        for i, ind in enumerate(pop[:5]):  # 每代前5个
            anc = ind.get("anchor", {})
            inst = anc.get("institution_or_company", anc.get("institution", anc.get("mentor", "")))
            adv = anc.get("advisor_or_team", anc.get("mentor", ""))[:12] if anc.get("advisor_or_team") else ""
            loc = f"{inst} {adv}".strip()[:28]
            s = ind.get("scores", [0]*11)
            rows.append([str(g) if i==0 else "", ind.get("option_id","")[:24], loc[:24],
                         f"{s[0]:.0f}" if s else "-", f"{s[7]:.0f}" if len(s)>7 else "-",
                         f"{s[8]:.0f}" if len(s)>8 else "-", f"{s[9]:.0f}" if len(s)>9 else "-",
                         f"{s[10]:.0f}" if len(s)>10 else "-"])
    colWidths = [10*mm, 50*mm, 45*mm, 12*mm, 12*mm, 12*mm, 12*mm, 12*mm]
    t = Table(rows, colWidths=colWidths)
    t.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), F), ("FONTSIZE", (0,0), (-1,-1), 6.5),
        ("BACKGROUND", (0,0), (-1,0), CA), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.25, CLg),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, CLg]),
    ]))
    return t

# ── Pareto 选项卡片 ───────────────────────────────────
def option_card(ind, rank, rec=None):
    anc = ind.get("anchor", {})
    inst = anc.get("institution_or_company", anc.get("institution", ""))
    adv = anc.get("advisor_or_team", anc.get("mentor", ""))
    route = ind.get("research_route", {})
    s = ind.get("scores", [0]*11)
    sc_str = "  ".join(f"{OBJ_NAMES[j]}={s[j]*10:.0f}" for j in OBJ_FOCUS) if len(s)>=11 else ""
    data = [[Paragraph(f"<b>#{rank} {ind.get('option_id','')}</b>", ps("c1",9,CA,11))],
            [Paragraph(f"{inst} | {adv}", ps("c2",8,CDk,11))],
            [Paragraph(f"得分: {sc_str}", ps("c3",7.5,CGy,10))],
            [Paragraph(f"<b>Y1</b>: {route.get('year1_focus','')[:120]}...", ps("c4",7.5,CDk,10))],
            [Paragraph(f"<b>Y2</b>: {route.get('year2_focus','')[:120]}...", ps("c5",7.5,CDk,10))]]
    if rec:
        data.append([Paragraph(f"推荐: {rec.get('brief','')[:100]}...", ps("c6",7,CGy,10))])
    t = Table(data, colWidths=[PW])
    t.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), F),
        ("BOX", (0,0), (-1,-1), 0.5, CLg),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e3f2fd")),
        ("TOPPADDING", (0,0), (-1,-1), 3), ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]))
    return t

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base, "moead_career_results.json")) as f:
        data = json.load(f)

    cfg = data["config"]
    gen_stats = data["gen_stats"]
    options = data["evaluated_options"]
    ranking = data.get("pareto_ranking", [])
    recs = {r["option_id"]: r for r in data.get("recommendations", [])}

    # 去重 Pareto 排序
    seen = set()
    pareto_order = []
    for oid in ranking:
        if oid not in seen:
            seen.add(oid)
            pareto_order.append(oid)
    opts_by_id = {o["option_id"]: o for o in options}
    top_opts = [opts_by_id[oid] for oid in pareto_order[:6] if oid in opts_by_id]

    doc = SimpleDocTemplate(os.path.join(base, "moead_career_report.pdf"),
                            pagesize=A4, leftMargin=14*mm, rightMargin=14*mm,
                            topMargin=14*mm, bottomMargin=14*mm)
    story = []

    # 封面
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph("博后/职业选择 MOEA/D 报告", ST["title"]))
    story.append(Paragraph("演化路线与分数变化 · Pareto 最优选项", ST["sub"]))
    story.append(Paragraph(f"种群={cfg['pop_size']} 代数={cfg['n_gens']} 目标=11维", ST["sub"]))
    story.append(Spacer(1, 4*mm))

    # 1. 分数演化曲线
    story.append(renderPDF.GraphicsFlowable(sec("01  分数演化曲线（关键维度）")))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph("契合、Agent、格局、启发、上限 五维均值随代数变化。", ST["body"]))
    story.append(Spacer(1, 2*mm))
    story.append(renderPDF.GraphicsFlowable(evo_score_chart(
        gen_stats, OBJ_FOCUS,
        ["契合", "Agent", "格局", "启发", "上限"], w=PW, h=115)))
    story.append(Paragraph("图1  关键目标均值演化曲线", ST["caption"]))
    story.append(Spacer(1, 3*mm))

    # 2. 11维均值表
    story.append(renderPDF.GraphicsFlowable(sec("02  各代 11 维均值表")))
    story.append(Spacer(1, 2*mm))
    story.append(gen_stats_table(gen_stats))
    story.append(Paragraph("表1  每代种群 11 维目标均值", ST["caption"]))
    story.append(Spacer(1, 4*mm))

    # 3. 演化路线
    log_dir = os.path.join(base, "career_logs")
    story.append(renderPDF.GraphicsFlowable(sec("03  演化路线（各代代表选项）")))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph("各代种群中部分选项及得分，展示选项类型随演化的变化。", ST["body"]))
    story.append(Spacer(1, 2*mm))
    story.append(route_table(log_dir, cfg["n_gens"]))
    story.append(Paragraph("表2  演化路线：各代代表选项与得分", ST["caption"]))
    story.append(Spacer(1, 4*mm))

    # 4. Pareto Top 选项
    story.append(renderPDF.GraphicsFlowable(sec("04  Pareto 最优选项（Top 5）")))
    story.append(Spacer(1, 2*mm))
    for i, ind in enumerate(top_opts[:5], 1):
        rec = recs.get(ind["option_id"])
        story.append(option_card(ind, i, rec))
        story.append(Spacer(1, 3*mm))

    story.append(Paragraph("以上为综合 11 维 Pareto 最优的职业选项。", ST["small"]))
    doc.build(story, onFirstPage=add_hf, onLaterPages=add_hf)
    print(f"✓ PDF 报告已生成: {os.path.join(base, 'moead_career_report.pdf')}")


if __name__ == "__main__":
    main()
