#!/usr/bin/env python3
"""
对比实验 PDF 报告生成器
单目标（仅新颖性）vs 四目标 MOEA/D
"""

import json, os, math
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, KeepTogether)
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Rect, Line, Circle, String, Polygon
from reportlab.graphics import renderPDF

pdfmetrics.registerFont(TTFont("U", "/Library/Fonts/Arial Unicode.ttf"))
F = "U"
W, H = A4
PW = W - 28*mm

# ── 颜色 ──────────────────────────────────────────────
CA  = colors.HexColor("#1565c0")   # exp A 主色：蓝
CB  = colors.HexColor("#2e7d32")   # exp B 主色：绿
CR  = colors.HexColor("#c62828")   # 红
CAm = colors.HexColor("#e65100")   # 橙
CPk = colors.HexColor("#ad1457")   # 粉
CDk = colors.HexColor("#212121")   # 深色文字
CGy = colors.HexColor("#546e7a")   # 灰
CLg = colors.HexColor("#eceff1")   # 浅灰背景
CBgA = colors.HexColor("#e3f2fd")  # A 背景
CBgB = colors.HexColor("#e8f5e9")  # B 背景

OBJ_ALL   = ["知识价值", "社会影响", "长尾度", "跨学科性", "新颖性", "可行性", "合理性"]
OBJ_SHORT = ["知识", "社会", "长尾", "跨学", "新颖", "可行", "合理"]
OBJ_FOCUS = [4, 0, 5, 6]  # 新颖, 知识, 可行, 合理

def ps(name, fs, tc, ld=14, **kw):
    return ParagraphStyle(name, fontName=F, fontSize=fs, textColor=tc, leading=ld, **kw)

ST = {
    "title":   ps("title",  20, CA, 24, spaceAfter=4, spaceBefore=6, alignment=1),
    "sub":     ps("sub",    10, CGy,14, spaceAfter=2, alignment=1),
    "h1":      ps("h1",    10.5,colors.white, 14, spaceAfter=0),
    "body":    ps("body",  8.5, CDk, 13, spaceAfter=3),
    "caption": ps("caption",7.5,CGy, 11, alignment=1, spaceAfter=4),
    "small":   ps("small",  7,  CGy, 10),
}

def section_bar(title_a, title_b=None, h=16):
    d = Drawing(PW, h)
    if title_b:
        d.add(Rect(0, 2, PW*0.485, h-2, fillColor=CA, strokeColor=None))
        d.add(Rect(PW*0.515, 2, PW*0.485, h-2, fillColor=CB, strokeColor=None))
        d.add(String(PW*0.242, 5, title_a, fontName=F, fontSize=9, fillColor=colors.white, textAnchor="middle"))
        d.add(String(PW*0.758, 5, title_b, fontName=F, fontSize=9, fillColor=colors.white, textAnchor="middle"))
    else:
        d.add(Rect(0, 2, PW, h-2, fillColor=CA, strokeColor=None))
        d.add(String(6, 5, title_a, fontName=F, fontSize=9.5, fillColor=colors.white))
    return d


# ═══════════════════════════════════════════════════════
#  图形：演化趋势对比（A 和 B，同一图中）
# ═══════════════════════════════════════════════════════
def evolution_chart(stats_a, stats_b, w=PW, h=90):
    d = Drawing(w, h)
    ox, oy = 32, 10
    cw, ch = w - ox - 20, h - oy - 22
    max_gen = max(len(stats_a), len(stats_b)) - 1

    for yv in [2, 4, 6, 8, 10]:
        y = oy + ((yv - 1) / 9) * ch
        d.add(Line(ox, y, ox+cw, y, strokeColor=CLg, strokeWidth=0.5))
        d.add(String(ox-4, y-3, str(yv), fontName=F, fontSize=6, fillColor=CGy, textAnchor="end"))

    lines = [
        (stats_a, 4, CA,     [],    "A-新颖"),
        (stats_a, 5, CA,     [4,2], "A-可行"),
        (stats_a, 6, CPk,    [4,2], "A-合理"),
        (stats_b, 4, CB,     [],    "B-新颖"),
        (stats_b, 5, CB,     [4,2], "B-可行"),
        (stats_b, 6, colors.HexColor("#00695c"), [4,2], "B-合理"),
    ]
    for stats, idx, col, dash, lbl in lines:
        pts = [(ox + g*(cw/max_gen), oy + ((stats[g]['avgs'][idx]-1)/9)*ch)
               for g in range(len(stats))]
        for i in range(len(pts)-1):
            d.add(Line(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1],
                       strokeColor=col, strokeWidth=1.8, strokeDashArray=dash))
        d.add(String(pts[-1][0]+2, pts[-1][1]-3, lbl, fontName=F, fontSize=6, fillColor=col))

    for g in range(max_gen+1):
        x = ox + g*(cw/max_gen)
        d.add(String(x, oy-10, f"{g}", fontName=F, fontSize=6, fillColor=CGy, textAnchor="middle"))
    d.add(String(ox+cw/2, oy-18, "代数", fontName=F, fontSize=7, fillColor=CDk, textAnchor="middle"))
    return d


# ═══════════════════════════════════════════════════════
#  图形：雷达对比（最终种群均值）
# ═══════════════════════════════════════════════════════
def radar_compare(avgs_a, avgs_b, w=None, h=110):
    w = w or PW * 0.46
    d = Drawing(w, h)
    focus = [4, 0, 5, 6]  # 新颖, 知识, 可行, 合理
    labels = ["新颖性", "知识价值", "可行性", "合理性"]
    n = len(focus)
    cx, cy = w/2, h/2
    r_max = min(cx, cy) - 18

    for lv in [0.25, 0.5, 0.75, 1.0]:
        pts = []
        for k in range(n):
            angle = math.pi/2 + 2*math.pi*k/n
            pts += [cx + r_max*lv*math.cos(angle), cy + r_max*lv*math.sin(angle)]
        for i in range(0, len(pts)-2, 2):
            j = (i+2) % len(pts)
            d.add(Line(pts[i], pts[i+1], pts[j], pts[j+1], strokeColor=CLg, strokeWidth=0.5))
    for k in range(n):
        angle = math.pi/2 + 2*math.pi*k/n
        d.add(Line(cx, cy, cx+r_max*math.cos(angle), cy+r_max*math.sin(angle),
                   strokeColor=CLg, strokeWidth=0.5))
        lx = cx + (r_max+12)*math.cos(angle)
        ly = cy + (r_max+12)*math.sin(angle)
        d.add(String(lx, ly-3.5, labels[k], fontName=F, fontSize=7, fillColor=CDk, textAnchor="middle"))

    for avgs, col, lbl in [(avgs_a, CA, "实验A"), (avgs_b, CB, "实验B")]:
        pts = []
        for k, fi in enumerate(focus):
            angle = math.pi/2 + 2*math.pi*k/n
            r = (avgs[fi]/10) * r_max
            pts += [cx + r*math.cos(angle), cy + r*math.sin(angle)]
        for i in range(0, len(pts)-2, 2):
            j = (i+2) % len(pts)
            d.add(Line(pts[i], pts[i+1], pts[j], pts[j+1], strokeColor=col, strokeWidth=2))
        # 收尾
        d.add(Line(pts[-2], pts[-1], pts[0], pts[1], strokeColor=col, strokeWidth=2))
        # 标注
        angle0 = math.pi/2
        r0 = (avgs[focus[0]]/10) * r_max
        d.add(Circle(cx+r0*math.cos(angle0), cy+r0*math.sin(angle0), 3,
                     fillColor=col, strokeColor=colors.white, strokeWidth=0.5))

    # 图例
    lx, ly = 4, 10
    for col, lbl in [(CA,"实验A（仅新颖）"), (CB,"实验B（四目标）")]:
        d.add(Line(lx, ly+3, lx+12, ly+3, strokeColor=col, strokeWidth=2))
        d.add(String(lx+15, ly, lbl, fontName=F, fontSize=6.5, fillColor=CDk))
        ly += 12
    return d


# ═══════════════════════════════════════════════════════
#  图形：条形对比（最终均值 A vs B）
# ═══════════════════════════════════════════════════════
def bar_compare(avgs_a, avgs_b, w=None, h=90):
    w = w or PW * 0.52
    d = Drawing(w, h)
    focus = [4, 0, 5, 6]
    names = ["新颖性", "知识价值", "可行性", "合理性"]
    cols  = [CAm, CA, CR, CPk]

    n = len(focus)
    ox, oy = 28, 8
    cw, ch = w - ox - 14, h - oy - 20
    group_w = cw / n
    bar_w = group_w * 0.3

    for yv in [0, 5, 10]:
        y = oy + (yv/10)*ch
        d.add(Line(ox, y, ox+cw, y, strokeColor=CLg, strokeWidth=0.5))
        d.add(String(ox-4, y-3, str(yv), fontName=F, fontSize=6, fillColor=CGy, textAnchor="end"))

    for i, (fi, name, col) in enumerate(zip(focus, names, cols)):
        gx = ox + i*group_w + group_w/2
        va = avgs_a[fi]
        vb = avgs_b[fi]
        # 实验A
        bx_a = gx - bar_w - 1
        d.add(Rect(bx_a, oy, bar_w, (va/10)*ch, fillColor=CA, strokeColor=None))
        d.add(String(bx_a+bar_w/2, oy+(va/10)*ch+1.5, f"{va:.1f}",
                     fontName=F, fontSize=6.5, fillColor=CA, textAnchor="middle"))
        # 实验B
        bx_b = gx + 1
        d.add(Rect(bx_b, oy, bar_w, (vb/10)*ch, fillColor=CB, strokeColor=None))
        d.add(String(bx_b+bar_w/2, oy+(vb/10)*ch+1.5, f"{vb:.1f}",
                     fontName=F, fontSize=6.5, fillColor=CB, textAnchor="middle"))
        # 差值标注
        delta = vb - va
        dcol = CB if delta > 0 else CR
        sign = "▲" if delta > 0 else "▼"
        d.add(String(gx, oy-1, f"{sign}{abs(delta):.1f}", fontName=F, fontSize=6.5,
                     fillColor=dcol, textAnchor="middle"))
        d.add(String(gx, oy-12, name, fontName=F, fontSize=7,
                     fillColor=CDk, textAnchor="middle"))

    # 图例
    lx, ly = ox, h-10
    d.add(Rect(lx, ly, 10, 7, fillColor=CA, strokeColor=None))
    d.add(String(lx+13, ly+1, "实验A（仅新颖）", fontName=F, fontSize=6.5, fillColor=CDk))
    d.add(Rect(lx+80, ly, 10, 7, fillColor=CB, strokeColor=None))
    d.add(String(lx+93, ly+1, "实验B（四目标）", fontName=F, fontSize=6.5, fillColor=CDk))
    return d


# ═══════════════════════════════════════════════════════
#  图形：散点图（新颖性 vs 可行性）
# ═══════════════════════════════════════════════════════
def scatter_novelty_feasibility(pop_a, pop_b, w=PW, h=100):
    d = Drawing(w, h)
    ox, oy = 34, 12
    cw, ch = w - ox - 20, h - oy - 24

    # 背景象限
    mx, my = ox + cw/2, oy + ch/2
    d.add(Rect(mx, oy, cw/2, ch/2, fillColor=colors.HexColor("#fff9e0"), strokeColor=CLg))
    d.add(Rect(ox, oy, cw/2, ch/2, fillColor=colors.HexColor("#e8f5e9"), strokeColor=CLg))
    d.add(Rect(mx, my, cw/2, ch/2, fillColor=colors.HexColor("#fce4ec"), strokeColor=CLg))
    d.add(Rect(ox, my, cw/2, ch/2, fillColor=colors.HexColor("#e3f2fd"), strokeColor=CLg))

    # 象限标签
    for (qx,qy,lbl) in [(ox+cw*0.75,oy+ch*0.75,"高新颖+低可行\n天马行空"),
                         (ox+cw*0.25,oy+ch*0.75,"低新颖+低可行"),
                         (ox+cw*0.75,oy+ch*0.25,"高新颖+高可行\n★理想区域"),
                         (ox+cw*0.25,oy+ch*0.25,"低新颖+高可行")]:
        for k, line in enumerate(lbl.split("\n")):
            d.add(String(qx, qy+5-k*9, line, fontName=F, fontSize=6,
                         fillColor=CGy, textAnchor="middle"))

    d.add(Line(mx, oy, mx, oy+ch, strokeColor=CGy, strokeWidth=0.5, strokeDashArray=[3,3]))
    d.add(Line(ox, my, ox+cw, my, strokeColor=CGy, strokeWidth=0.5, strokeDashArray=[3,3]))

    for pop, col, marker_col in [(pop_a, CA, colors.HexColor("#bbdefb")),
                                  (pop_b, CB, colors.HexColor("#c8e6c9"))]:
        for ind in pop:
            fr = ind['scores'][4]  # 新颖性
            fe = ind['scores'][5]  # 可行性
            ri = ind['scores'][6]  # 合理性（圆圈大小）
            x = ox + fr * cw
            y = oy + fe * ch
            x = max(ox+2, min(ox+cw-2, x))
            y = max(oy+2, min(oy+ch-2, y))
            r = 2.5 + ri * 4
            d.add(Circle(x, y, r, fillColor=col, strokeColor=colors.white, strokeWidth=0.5))

    d.add(String(ox+cw/2, oy-14, "← 新颖性 →", fontName=F, fontSize=7, fillColor=CDk, textAnchor="middle"))
    for yf, lbl in [(0,"0"),(0.5,"5"),(1,"10")]:
        y = oy + yf*ch
        d.add(String(ox-4, y-3, lbl, fontName=F, fontSize=6, fillColor=CGy, textAnchor="end"))
    d.add(String(14, oy+ch/2, "可行性↑", fontName=F, fontSize=7, fillColor=CDk, textAnchor="middle"))

    lx, ly = ox+cw+4, oy+ch-10
    for col, lbl in [(CA,"实验A"), (CB,"实验B")]:
        d.add(Circle(lx+4, ly+3, 4, fillColor=col, strokeColor=colors.white))
        d.add(String(lx+11, ly, lbl, fontName=F, fontSize=6.5, fillColor=CDk))
        ly -= 14
    d.add(String(lx, ly+4, "圆大=合理高", fontName=F, fontSize=5.5, fillColor=CGy))
    return d


def add_hf(canvas, doc):
    canvas.saveState()
    canvas.setFont(F, 7.5)
    canvas.setFillColor(CGy)
    canvas.drawString(14*mm, H-9*mm, "对比实验报告：单目标新颖性 vs 四目标 MOEA/D")
    canvas.drawRightString(W-14*mm, H-9*mm, f"第 {doc.page} 页")
    canvas.line(14*mm, H-10*mm, W-14*mm, H-10*mm)
    canvas.setFillColor(CGy)
    canvas.drawCentredString(W/2, 7*mm, "MOEA/D × DeepSeek  ·  新颖性单目标 vs 四目标对比实验")
    canvas.restoreState()


def build_pdf(path):
    with open(os.path.expanduser("~/moead_compare_results.json")) as f:
        data = json.load(f)

    cfg    = data["config"]
    sa     = data["exp_a"]["gen_stats"]
    sb     = data["exp_b"]["gen_stats"]
    pop_a  = data["exp_a"]["final_pop"]
    pop_b  = data["exp_b"]["final_pop"]
    avgs_a = sa[-1]["avgs"]
    avgs_b = sb[-1]["avgs"]

    doc = SimpleDocTemplate(path, pagesize=A4,
                            leftMargin=14*mm, rightMargin=14*mm,
                            topMargin=14*mm, bottomMargin=14*mm)
    story = []

    # ── 标题 ───────────────────────────────────────────
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph("单目标 vs 多目标优化对比实验", ST["title"]))
    story.append(Paragraph("新颖性单独优化 vs 新颖性+知识价值+可行性+合理性联合优化", ST["sub"]))
    story.append(Paragraph(
        f"模型: {cfg['model']}  ·  种群: {cfg['pop_size']}  ·  代数: {cfg['n_gens']}  ·"
        " 两实验完全相同的超参数，相同评估标准", ST["sub"]))
    story.append(Spacer(1, 3*mm))

    # ── 核心数字摘要 ───────────────────────────────────
    delta = [avgs_b[j] - avgs_a[j] for j in range(7)]
    summary = [
        ["目标", "实验A（仅新颖）", "实验B（四目标）", "差值 B−A", "解读"],
        ["新颖性",   f"{avgs_a[4]:.1f}", f"{avgs_b[4]:.1f}", f"{delta[4]:+.1f}",
         "A压倒性领先，多目标代价约−2.2分"],
        ["知识价值", f"{avgs_a[0]:.1f}", f"{avgs_b[0]:.1f}", f"{delta[0]:+.1f}",
         "两者相近，B略低"],
        ["可行性",   f"{avgs_a[5]:.1f}", f"{avgs_b[5]:.1f}", f"{delta[5]:+.1f}",
         "★ B大幅领先 +4.5，A几乎不可行"],
        ["合理性",   f"{avgs_a[6]:.1f}", f"{avgs_b[6]:.1f}", f"{delta[6]:+.1f}",
         "★ B大幅领先 +4.1，A偏向科幻假说"],
        ["长尾度",   f"{avgs_a[2]:.1f}", f"{avgs_b[2]:.1f}", f"{delta[2]:+.1f}",
         "A更小众，B倾向主流可实验方向"],
        ["跨学科性", f"{avgs_a[3]:.1f}", f"{avgs_b[3]:.1f}", f"{delta[3]:+.1f}",
         "A稍高，聚焦宇宙级跨学科猜想"],
        ["社会影响", f"{avgs_a[1]:.1f}", f"{avgs_b[1]:.1f}", f"{delta[1]:+.1f}",
         "B更高，可落地研究社会价值更清晰"],
    ]
    st = Table(summary, colWidths=[20*mm, 22*mm, 22*mm, 18*mm, 86*mm])
    st.setStyle(TableStyle([
        ("FONTNAME",      (0,0), (-1,-1), F),
        ("FONTSIZE",      (0,0), (-1,-1), 8),
        ("BACKGROUND",    (0,0), (-1,0),  CDk),
        ("TEXTCOLOR",     (0,0), (-1,0),  colors.white),
        ("GRID",          (0,0), (-1,-1), 0.3, CLg),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 2),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, CLg]),
        ("TEXTCOLOR",     (1,1), (1,-1),  CA),
        ("TEXTCOLOR",     (2,1), (2,-1),  CB),
        ("TEXTCOLOR",     (3,3), (3,4),   CB),
        ("TEXTCOLOR",     (3,1), (3,2),   CR),
    ]))
    story.append(st)
    story.append(Paragraph("表1  最终种群7维均值对比（共6代，种群15）", ST["caption"]))
    story.append(Spacer(1, 3*mm))

    # ── 演化趋势 ───────────────────────────────────────
    story.append(renderPDF.GraphicsFlowable(section_bar("01  演化趋势对比（新颖 / 可行 / 合理）")))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "实线=新颖性，虚线=可行性/合理性。蓝色=实验A（仅新颖），绿色=实验B（四目标）。",
        ST["body"]))
    story.append(Spacer(1, 2*mm))
    story.append(renderPDF.GraphicsFlowable(evolution_chart(sa, sb, w=PW, h=100)))
    story.append(Paragraph("图1  各目标均值演化曲线（代数0=初始种群）", ST["caption"]))
    story.append(Spacer(1, 3*mm))

    # ── 散点图 ─────────────────────────────────────────
    story.append(renderPDF.GraphicsFlowable(section_bar("02  新颖性 × 可行性 散点图（最终种群）")))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "横轴=新颖性，纵轴=可行性，圆圈大小=合理性。"
        "实验A（蓝）聚集在右下（高新颖+低可行），实验B（绿）聚集在中上（中新颖+高可行）。",
        ST["body"]))
    story.append(Spacer(1, 2*mm))
    story.append(renderPDF.GraphicsFlowable(scatter_novelty_feasibility(pop_a, pop_b, w=PW, h=105)))
    story.append(Paragraph("图2  新颖性 vs 可行性 散点图（圆大小=合理性）", ST["caption"]))
    story.append(Spacer(1, 3*mm))

    # ── 雷达 + 柱状 ────────────────────────────────────
    story.append(renderPDF.GraphicsFlowable(section_bar("03  四维均值：雷达图 + 柱状对比")))
    story.append(Spacer(1, 2*mm))

    rdr = radar_compare(avgs_a, avgs_b, w=PW*0.46, h=115)
    bar = bar_compare(avgs_a, avgs_b, w=PW*0.52, h=115)
    side = Table([[renderPDF.GraphicsFlowable(rdr), renderPDF.GraphicsFlowable(bar)]],
                 colWidths=[PW*0.48, PW*0.50])
    side.setStyle(TableStyle([
        ("VALIGN",       (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING",  (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
    ]))
    story.append(side)
    story.append(Paragraph("图3  四目标雷达对比（左）与均值柱状图（右）", ST["caption"]))
    story.append(Spacer(1, 3*mm))

    # ── 代表性案例 ─────────────────────────────────────
    story.append(renderPDF.GraphicsFlowable(section_bar("04  代表案例对比")))
    story.append(Spacer(1, 2*mm))

    # 实验A Top3（按新颖性）
    top_a = sorted(pop_a, key=lambda x: x['scores'][4], reverse=True)[:3]
    # 实验B Top3（按综合）
    top_b = sorted(pop_b, key=lambda x: sum(x['scores'][i] for i in OBJ_FOCUS), reverse=True)[:3]

    case_data = [["实验", "研究主题", "领域", "新颖", "知识", "可行", "合理"]]
    for ind in top_a:
        s = ind['scores']
        case_data.append(["A", ind['topic'][:22], ind.get('domain','')[:24],
                          f"{s[4]*10:.0f}", f"{s[0]*10:.0f}", f"{s[5]*10:.0f}", f"{s[6]*10:.0f}"])
    for ind in top_b:
        s = ind['scores']
        case_data.append(["B", ind['topic'][:22], ind.get('domain','')[:24],
                          f"{s[4]*10:.0f}", f"{s[0]*10:.0f}", f"{s[5]*10:.0f}", f"{s[6]*10:.0f}"])

    ct = Table(case_data, colWidths=[10*mm, 50*mm, 52*mm,
                                      14*mm, 14*mm, 14*mm, 14*mm])
    ct.setStyle(TableStyle([
        ("FONTNAME",     (0,0), (-1,-1), F),
        ("FONTSIZE",     (0,0), (-1,-1), 7.5),
        ("BACKGROUND",   (0,0), (-1,0),  CDk),
        ("TEXTCOLOR",    (0,0), (-1,0),  colors.white),
        ("GRID",         (0,0), (-1,-1), 0.3, CLg),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",   (0,0), (-1,-1), 2),
        ("BOTTOMPADDING",(0,0), (-1,-1), 2),
        ("ALIGN",        (3,0), (-1,-1), "CENTER"),
        ("BACKGROUND",   (0,1), (-1,3),  CBgA),
        ("BACKGROUND",   (0,4), (-1,6),  CBgB),
        ("TEXTCOLOR",    (0,1), (0,3),   CA),
        ("TEXTCOLOR",    (0,4), (0,6),   CB),
    ]))
    story.append(ct)
    story.append(Paragraph("表2  实验A Top3（按新颖性）与实验B Top3（按四目标综合）", ST["caption"]))
    story.append(Spacer(1, 3*mm))

    # ── 结论 ───────────────────────────────────────────
    story.append(renderPDF.GraphicsFlowable(section_bar("05  实验结论")))
    story.append(Spacer(1, 2*mm))

    conclusions = [
        ("单目标的极端化效应",
         f"仅优化新颖性6代后，新颖性均值达到 {avgs_a[4]:.1f} 分，"
         f"但可行性跌至 {avgs_a[5]:.1f}、合理性跌至 {avgs_a[6]:.1f}。"
         "种群收敛到「暗物质通信生物学」「时空晶体网络」等极端科幻方向，"
         "几乎无法在现实中开展研究。"),
        ("多目标的均衡约束力",
         f"四目标 MOEA/D 以牺牲约2.2分新颖性为代价，"
         f"将可行性从 {avgs_a[5]:.1f} 提升到 {avgs_b[5]:.1f}（+{avgs_b[5]-avgs_a[5]:.1f}），"
         f"合理性从 {avgs_a[6]:.1f} 提升到 {avgs_b[6]:.1f}（+{avgs_b[6]-avgs_a[6]:.1f}）。"
         "种群收敛到「纳米技术×环境治理」「量子×合成生物学」等实验室可开展方向。"),
        ("Pareto 权衡的核心价值",
         "多目标优化不是「让所有目标都最大」，而是找到在冲突目标之间的合理权衡面。"
         "实验证明：如果只关心新颖性，算法会毫不犹豫地抛弃可行性和合理性；"
         "加入约束后，系统自动在「大胆但可研究」与「新颖但科幻」之间找到平衡点。"),
        ("科研选题的实践启示",
         "纯追求「酷炫新颖」的研究方向会产生无法发表的论文；"
         "把可行性+合理性作为显式优化目标，演化系统会自动筛选出"
         "「当前可开展的前沿方向」——这正是高影响力研究的核心特征。"),
    ]
    con_data = [["#", "结论维度", "内容"]]
    for i, (title, text) in enumerate(conclusions, 1):
        con_data.append([str(i), title, text])
    con_t = Table(con_data, colWidths=[8*mm, 32*mm, 128*mm])
    con_t.setStyle(TableStyle([
        ("FONTNAME",     (0,0), (-1,-1), F),
        ("FONTSIZE",     (0,0), (-1,-1), 8),
        ("BACKGROUND",   (0,0), (-1,0),  CDk),
        ("TEXTCOLOR",    (0,0), (-1,0),  colors.white),
        ("GRID",         (0,0), (-1,-1), 0.3, CLg),
        ("VALIGN",       (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",   (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, CLg]),
        ("ALIGN",        (0,0), (0,-1),  "CENTER"),
        ("TEXTCOLOR",    (1,1), (1,1),   CR),
        ("TEXTCOLOR",    (1,2), (1,2),   CB),
        ("TEXTCOLOR",    (1,3), (1,3),   CAm),
        ("TEXTCOLOR",    (1,4), (1,4),   colors.HexColor("#00695c")),
    ]))
    story.append(con_t)
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(
        f"数据来源：DeepSeek {cfg['model']} · 种群={cfg['pop_size']} · "
        f"{cfg['n_gens']}代 · 两实验独立运行", ST["small"]))

    doc.build(story, onFirstPage=add_hf, onLaterPages=add_hf)
    print(f"✓ PDF 已生成: {path}")


if __name__ == "__main__":
    build_pdf(os.path.expanduser("~/moead_compare_report.pdf"))
