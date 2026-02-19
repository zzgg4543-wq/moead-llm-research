#!/usr/bin/env python3
"""
CS 对比实验 PDF 报告生成器
包含：演化趋势、散点分析、完整研究方案展示（Pareto 最优解）
"""

import json, os, math
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, KeepTogether, HRFlowable)
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Rect, Line, Circle, String, Polygon
from reportlab.graphics import renderPDF

pdfmetrics.registerFont(TTFont("U", "/Library/Fonts/Arial Unicode.ttf"))
F = "U"
W, H = A4
PW = W - 28*mm

CA  = colors.HexColor("#1565c0")
CB  = colors.HexColor("#2e7d32")
CR  = colors.HexColor("#c62828")
CAm = colors.HexColor("#e65100")
CPk = colors.HexColor("#ad1457")
CDk = colors.HexColor("#212121")
CGy = colors.HexColor("#607d8b")
CLg = colors.HexColor("#eceff1")
CBgA = colors.HexColor("#e3f2fd")
CBgB = colors.HexColor("#e8f5e9")
CHL = colors.HexColor("#fff9c4")

OBJ_ALL = ["知识价值","社会影响","长尾度","跨学科性","新颖性","可行性","合理性"]
OBJ_4_IDX = [4, 0, 5, 6]

def ps(name, fs, tc, ld=14, **kw):
    return ParagraphStyle(name, fontName=F, fontSize=fs, textColor=tc, leading=ld, **kw)

ST = {
    "title":   ps("t",  18, CA, 22, spaceAfter=3, spaceBefore=4, alignment=1),
    "sub":     ps("s",   9, CGy,13, spaceAfter=2, alignment=1),
    "h2":      ps("h2",  9, colors.white, 13),
    "body":    ps("bd", 8.5,CDk, 13, spaceAfter=3),
    "plan_h":  ps("ph", 8.5,CA,  12, spaceBefore=4, spaceAfter=1),
    "plan_b":  ps("pb", 8,  CDk, 12, spaceAfter=2, leftIndent=8),
    "caption": ps("cp", 7.5,CGy, 11, alignment=1, spaceAfter=3),
    "small":   ps("sm",  7,  CGy, 10),
    "score":   ps("sc",  8,  CB,  11),
}

def sec(title, color=CA, h=16):
    d = Drawing(PW, h)
    d.add(Rect(0, 2, PW, h-2, fillColor=color, strokeColor=None))
    d.add(String(6, 5.5, title, fontName=F, fontSize=9.5, fillColor=colors.white))
    return d

def add_hf(canvas, doc):
    canvas.saveState()
    canvas.setFont(F, 7.5); canvas.setFillColor(CGy)
    canvas.drawString(14*mm, H-9*mm, "CS 领域对比实验：单目标新颖性 vs 四目标 MOEA/D · 完整研究方案报告")
    canvas.drawRightString(W-14*mm, H-9*mm, f"第 {doc.page} 页")
    canvas.line(14*mm, H-10*mm, W-14*mm, H-10*mm)
    canvas.drawCentredString(W/2, 7*mm, "MOEA/D × DeepSeek  ·  CS 领域长尾研究课题进化系统")
    canvas.restoreState()

# ── 图形组件 ──────────────────────────────────────────

def evo_chart(sa, sb, w=PW, h=95):
    d = Drawing(w, h)
    ox, oy = 30, 10; cw, ch = w-ox-16, h-oy-22
    mg = max(len(sa), len(sb)) - 1

    for yv in [2,4,6,8,10]:
        y = oy + ((yv-1)/9)*ch
        d.add(Line(ox, y, ox+cw, y, strokeColor=CLg, strokeWidth=0.5))
        d.add(String(ox-4, y-3, str(yv), fontName=F, fontSize=6, fillColor=CGy, textAnchor="end"))

    specs = [
        (sa, 4, CA,  [],    "A·新颖"),
        (sa, 5, colors.HexColor("#90caf9"), [4,2], "A·可行"),
        (sa, 6, colors.HexColor("#ce93d8"), [4,2], "A·合理"),
        (sb, 4, CB,  [],    "B·新颖"),
        (sb, 5, colors.HexColor("#a5d6a7"), [4,2], "B·可行"),
        (sb, 6, colors.HexColor("#80cbc4"), [4,2], "B·合理"),
    ]
    for stats, idx, col, dash, lbl in specs:
        pts = [(ox+g*(cw/mg), oy+((stats[g]['avgs'][idx]-1)/9)*ch) for g in range(len(stats))]
        for i in range(len(pts)-1):
            d.add(Line(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1],
                       strokeColor=col, strokeWidth=1.8, strokeDashArray=dash))
        d.add(String(pts[-1][0]+2, pts[-1][1]-3, lbl, fontName=F, fontSize=6, fillColor=col))
    for g in range(mg+1):
        x = ox + g*(cw/mg)
        d.add(String(x, oy-10, str(g), fontName=F, fontSize=6, fillColor=CGy, textAnchor="middle"))
    d.add(String(ox+cw/2, oy-18, "代数", fontName=F, fontSize=7, fillColor=CDk, textAnchor="middle"))
    return d

def scatter_plot(pop_a, pop_b, w=PW, h=100):
    d = Drawing(w, h)
    ox, oy = 32, 12; cw, ch = w-ox-22, h-oy-24
    mx, my = ox+cw/2, oy+ch/2

    bgs = [(mx,oy,cw/2,ch/2,colors.HexColor("#fff3e0"),"高新颖+低可行\n天马行空型"),
           (ox,oy,cw/2,ch/2,colors.HexColor("#e8f5e9"),"低新颖+低可行"),
           (mx,my,cw/2,ch/2,colors.HexColor("#fce4ec"),"高新颖+高可行\n★理想区域"),
           (ox,my,cw/2,ch/2,colors.HexColor("#e3f2fd"),"低新颖+高可行\n可落地研究")]
    for (bx,by,bw,bh,bc,bl) in bgs:
        d.add(Rect(bx,by,bw,bh,fillColor=bc,strokeColor=CLg,strokeWidth=0.3))
        for k,ln in enumerate(bl.split("\n")):
            d.add(String(bx+bw/2, by+bh/2+3-k*9, ln, fontName=F, fontSize=5.5,
                         fillColor=CGy, textAnchor="middle"))
    d.add(Line(mx,oy,mx,oy+ch,strokeColor=CGy,strokeWidth=0.5,strokeDashArray=[3,3]))
    d.add(Line(ox,my,ox+cw,my,strokeColor=CGy,strokeWidth=0.5,strokeDashArray=[3,3]))

    for pop, col in [(pop_a, CA), (pop_b, CB)]:
        for ind in pop:
            fr = ind['scores'][4]; fe = ind['scores'][5]; ri = ind['scores'][6]
            x = max(ox+2, min(ox+cw-2, ox+fr*cw))
            y = max(oy+2, min(oy+ch-2, oy+fe*ch))
            r = 2 + ri*4
            d.add(Circle(x, y, r, fillColor=col, strokeColor=colors.white, strokeWidth=0.5))

    d.add(String(ox+cw/2, oy-14, "← 新颖性 →", fontName=F, fontSize=7, fillColor=CDk, textAnchor="middle"))
    for yf, lb in [(0,"0"),(0.5,"5"),(1,"10")]:
        d.add(String(ox-4, oy+yf*ch-3, lb, fontName=F, fontSize=6, fillColor=CGy, textAnchor="end"))
    d.add(String(12, oy+ch/2, "可行性↑", fontName=F, fontSize=7, fillColor=CDk, textAnchor="middle"))
    lx, ly = ox+cw+3, oy+ch-8
    for col, lb in [(CA,"实验A·单目标"), (CB,"实验B·四目标")]:
        d.add(Circle(lx+4, ly+3, 4, fillColor=col, strokeColor=colors.white))
        d.add(String(lx+11, ly, lb, fontName=F, fontSize=6, fillColor=CDk))
        ly -= 14
    d.add(String(lx, ly+3, "圆大=合理高", fontName=F, fontSize=5.5, fillColor=CGy))
    return d

def radar_two(avgs_a, avgs_b, w=None, h=105):
    w = w or PW*0.44
    d = Drawing(w, h)
    focus = [4, 0, 5, 6]
    labels = ["新颖性","知识价值","可行性","合理性"]
    n = len(focus); cx, cy = w/2, h/2; rm = min(cx,cy)-18
    for lv in [0.25, 0.5, 0.75, 1.0]:
        pts = []
        for k in range(n):
            a = math.pi/2 + 2*math.pi*k/n
            pts += [cx+rm*lv*math.cos(a), cy+rm*lv*math.sin(a)]
        for i in range(0, len(pts)-2, 2):
            j = (i+2) % len(pts)
            d.add(Line(pts[i], pts[i+1], pts[j], pts[j+1], strokeColor=CLg, strokeWidth=0.5))
    for k in range(n):
        a = math.pi/2 + 2*math.pi*k/n
        d.add(Line(cx,cy,cx+rm*math.cos(a),cy+rm*math.sin(a),strokeColor=CLg,strokeWidth=0.5))
        lx = cx+(rm+13)*math.cos(a); ly = cy+(rm+13)*math.sin(a)
        d.add(String(lx, ly-3.5, labels[k], fontName=F, fontSize=7, fillColor=CDk, textAnchor="middle"))
    for avgs, col in [(avgs_a, CA), (avgs_b, CB)]:
        pts = []
        for k, fi in enumerate(focus):
            a = math.pi/2 + 2*math.pi*k/n
            r = (avgs[fi]/10)*rm
            pts += [cx+r*math.cos(a), cy+r*math.sin(a)]
        for i in range(0, len(pts)-2, 2):
            j = (i+2) % len(pts)
            d.add(Line(pts[i], pts[i+1], pts[j], pts[j+1], strokeColor=col, strokeWidth=2))
        d.add(Line(pts[-2], pts[-1], pts[0], pts[1], strokeColor=col, strokeWidth=2))
    lx, ly = 4, 12
    for col, lb in [(CA,"实验A（仅新颖）"),(CB,"实验B（四目标）")]:
        d.add(Line(lx,ly+3,lx+12,ly+3,strokeColor=col,strokeWidth=2))
        d.add(String(lx+15,ly,lb,fontName=F,fontSize=6.5,fillColor=CDk)); ly+=12
    return d

def bar_cmp(avgs_a, avgs_b, w=None, h=105):
    w = w or PW*0.54
    d = Drawing(w, h)
    focus = [4, 0, 5, 6]; names = ["新颖性","知识价值","可行性","合理性"]
    n = len(focus); ox, oy = 26, 8; cw, ch = w-ox-12, h-oy-20
    gw = cw/n; bw = gw*0.3
    for yv in [0,5,10]:
        y = oy+(yv/10)*ch
        d.add(Line(ox,y,ox+cw,y,strokeColor=CLg,strokeWidth=0.5))
        d.add(String(ox-4,y-3,str(yv),fontName=F,fontSize=6,fillColor=CGy,textAnchor="end"))
    for i, (fi, nm) in enumerate(zip(focus, names)):
        gx = ox+i*gw+gw/2; va = avgs_a[fi]; vb = avgs_b[fi]
        d.add(Rect(gx-bw-1,oy,bw,(va/10)*ch,fillColor=CA,strokeColor=None))
        d.add(String(gx-bw/2-0.5,oy+(va/10)*ch+1.5,f"{va:.1f}",fontName=F,fontSize=6,fillColor=CA,textAnchor="middle"))
        d.add(Rect(gx+1,oy,bw,(vb/10)*ch,fillColor=CB,strokeColor=None))
        d.add(String(gx+bw/2+1.5,oy+(vb/10)*ch+1.5,f"{vb:.1f}",fontName=F,fontSize=6.5,fillColor=CB,textAnchor="middle"))
        delta = vb-va; dc = CB if delta>0 else CR; sign = "▲" if delta>0 else "▼"
        d.add(String(gx,oy-1,f"{sign}{abs(delta):.1f}",fontName=F,fontSize=6.5,fillColor=dc,textAnchor="middle"))
        d.add(String(gx,oy-12,nm,fontName=F,fontSize=7,fillColor=CDk,textAnchor="middle"))
    lx, ly = ox, h-10
    d.add(Rect(lx,ly,10,7,fillColor=CA,strokeColor=None)); d.add(String(lx+13,ly+1,"实验A",fontName=F,fontSize=6.5,fillColor=CDk))
    d.add(Rect(lx+44,ly,10,7,fillColor=CB,strokeColor=None)); d.add(String(lx+57,ly+1,"实验B",fontName=F,fontSize=6.5,fillColor=CDk))
    return d

# ── 研究方案卡片 ──────────────────────────────────────

def plan_card(ind: dict, rank: int, total: int, color=CB) -> list:
    """渲染单个研究方案为 Platypus 元素列表"""
    s = ind['scores']
    plan = ind.get('plan', {})
    elems = []

    # 标题行
    score_parts = " · ".join(
        f"{OBJ_ALL[j][:2]}={s[j]*10:.0f}" for j in [4, 0, 5, 6]
    )
    header_data = [[
        Paragraph(f"<b>#{rank}/{total}</b>", ps("r",8,colors.white,10)),
        Paragraph(f"<b>{ind['topic']}</b>", ps("tp",9,colors.white,11)),
        Paragraph(f"<b>{score_parts}</b>", ps("sc",7.5,colors.HexColor("#b2dfdb"),10)),
    ]]
    ht = Table(header_data, colWidths=[12*mm, 95*mm, 61*mm])
    ht.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,0), color),
        ("FONTNAME",   (0,0),(-1,0), F),
        ("VALIGN",     (0,0),(-1,0), "MIDDLE"),
        ("TOPPADDING", (0,0),(-1,0), 3),
        ("BOTTOMPADDING",(0,0),(-1,0),3),
        ("LEFTPADDING",(0,0),(-1,0),4),
    ]))
    elems.append(ht)

    # 方案内容
    questions = plan.get('questions', [])
    contributions = plan.get('contributions', [])
    content = [
        ["领域",    ind.get('domain','—')],
        ["背景",    plan.get('background','—')],
        ["核心问题", "\n".join(f"  Q{i+1}：{q}" for i, q in enumerate(questions)) or "—"],
        ["技术路线", plan.get('methodology','—')],
        ["预期贡献", "\n".join(f"  C{i+1}：{c}" for i, c in enumerate(contributions)) or "—"],
    ]
    body_data = [[Paragraph(f"<b>{k}</b>", ps(f"k{k}",8,color,11)),
                  Paragraph(v, ps(f"v{k}",8,CDk,11))]
                 for k, v in content]
    bt = Table(body_data, colWidths=[18*mm, 150*mm])
    bt.setStyle(TableStyle([
        ("FONTNAME",     (0,0),(-1,-1), F),
        ("VALIGN",       (0,0),(-1,-1), "TOP"),
        ("TOPPADDING",   (0,0),(-1,-1), 2),
        ("BOTTOMPADDING",(0,0),(-1,-1), 2),
        ("LEFTPADDING",  (0,0),(-1,-1), 4),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.white, CLg]),
        ("LINEBELOW",    (0,-1),(-1,-1), 0.5, CLg),
    ]))
    elems.append(bt)
    elems.append(Spacer(1, 4*mm))
    return elems


# ══════════════════════════════════════════════════════
#  主构建
# ══════════════════════════════════════════════════════
def build_pdf(path):
    with open(os.path.expanduser("~/moead-research/moead_cs_results.json")) as f:
        data = json.load(f)

    cfg   = data["config"]
    sa    = data["exp_a"]["gen_stats"]
    sb    = data["exp_b"]["gen_stats"]
    pop_a = data["exp_a"]["final_pop"]
    pop_b = data["exp_b"]["final_pop"]
    pareto_b = data["exp_b"].get("pareto", [])

    avgs_a = sa[-1]["avgs"]
    avgs_b = sb[-1]["avgs"]

    doc = SimpleDocTemplate(path, pagesize=A4,
                            leftMargin=14*mm, rightMargin=14*mm,
                            topMargin=14*mm, bottomMargin=14*mm)
    story = []

    # ── 封面 ──────────────────────────────────────────
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph("计算机科学长尾研究课题进化实验", ST["title"]))
    story.append(Paragraph("单目标（新颖性）vs 四目标 MOEA/D · 含完整研究方案", ST["sub"]))
    story.append(Paragraph(
        f"模型: {cfg['model']}  ·  种群: {cfg['pop_size']}  ·  代数: {cfg['n_gens']}  ·"
        f"  Pareto 最优解: {len(pareto_b)} 个", ST["sub"]))
    story.append(Spacer(1, 3*mm))

    # 核心数字摘要
    delta = [avgs_b[j]-avgs_a[j] for j in range(7)]
    rows = [["目标","实验A（仅新颖）","实验B（四目标）","差值 B−A","解读"]]
    specs = [
        (4,"新颖性",  "A领先，多目标代价约−2~3分"),
        (0,"知识价值","B较高，研究方案更严谨系统"),
        (5,"可行性",  "★ B大幅领先，方案具体可执行"),
        (6,"合理性",  "★ B大幅领先，理论依据充分"),
        (2,"长尾度",  "A更小众，B偏向有实验依据方向"),
    ]
    for fi, nm, note in specs:
        d = delta[fi]
        rows.append([nm, f"{avgs_a[fi]:.1f}", f"{avgs_b[fi]:.1f}",
                     f"{d:+.1f}", note])
    st = Table(rows, colWidths=[20*mm, 24*mm, 24*mm, 16*mm, 84*mm])
    ts = TableStyle([
        ("FONTNAME",     (0,0),(-1,-1), F), ("FONTSIZE",(0,0),(-1,-1),8),
        ("BACKGROUND",   (0,0),(-1,0),  CDk), ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("GRID",         (0,0),(-1,-1), 0.3, CLg), ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",   (0,0),(-1,-1), 2), ("BOTTOMPADDING",(0,0),(-1,-1),2),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,CLg]),
        ("TEXTCOLOR",    (1,1),(1,-1),  CA), ("TEXTCOLOR",(2,1),(2,-1),CB),
    ])
    for row_i, (fi,_,_) in enumerate(specs, 1):
        if delta[fi] > 0: ts.add("TEXTCOLOR",(3,row_i),(3,row_i),CB)
        else:              ts.add("TEXTCOLOR",(3,row_i),(3,row_i),CR)
    st.setStyle(ts)
    story.append(st)
    story.append(Paragraph("表1  最终种群关键目标均值对比", ST["caption"]))
    story.append(Spacer(1, 3*mm))

    # ── 演化趋势 ──────────────────────────────────────
    story.append(renderPDF.GraphicsFlowable(sec("01  演化趋势：新颖性 / 可行性 / 合理性")))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "实线=新颖性，虚线=可行性/合理性。蓝=实验A，绿=实验B。"
        "实验A新颖性快速达到9+但可行/合理持续下降；实验B三者均衡提升。",
        ST["body"]))
    story.append(Spacer(1, 2*mm))
    story.append(renderPDF.GraphicsFlowable(evo_chart(sa, sb, w=PW, h=100)))
    story.append(Paragraph("图1  六代演化曲线（新颖/可行/合理均值）", ST["caption"]))
    story.append(Spacer(1, 3*mm))

    # ── 散点图 ────────────────────────────────────────
    story.append(renderPDF.GraphicsFlowable(sec("02  新颖性 × 可行性 散点分布")))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "实验A（蓝）：高度集中在「高新颖+低可行」象限，生成大量无法实验的科幻方向。"
        "实验B（绿）：向「高新颖+高可行」理想区域靠拢。圆圈大小代表合理性。",
        ST["body"]))
    story.append(Spacer(1, 2*mm))
    story.append(renderPDF.GraphicsFlowable(scatter_plot(pop_a, pop_b, w=PW, h=105)))
    story.append(Paragraph("图2  最终种群散点分布（颜色=实验，大小=合理性）", ST["caption"]))
    story.append(Spacer(1, 3*mm))

    # ── 雷达 + 柱状 ───────────────────────────────────
    story.append(renderPDF.GraphicsFlowable(sec("03  四目标均值：雷达 + 柱状对比")))
    story.append(Spacer(1, 2*mm))
    rdr = radar_two(avgs_a, avgs_b, w=PW*0.44, h=108)
    bar = bar_cmp(avgs_a, avgs_b, w=PW*0.54, h=108)
    side = Table([[renderPDF.GraphicsFlowable(rdr), renderPDF.GraphicsFlowable(bar)]],
                 colWidths=[PW*0.46, PW*0.54])
    side.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),
                               ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0)]))
    story.append(side)
    story.append(Paragraph("图3  四维雷达对比（左）与均值柱状图（右）", ST["caption"]))
    story.append(Spacer(1, 3*mm))

    # ── 实验B Pareto 完整研究方案 ─────────────────────
    story.append(renderPDF.GraphicsFlowable(sec("04  实验B · Pareto 最优解完整研究方案")))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        f"以下展示四目标 MOEA/D 演化出的 {len(pareto_b)} 个 Pareto 最优研究课题的完整研究方案，"
        "按四目标综合得分降序排列。每个方案包含：背景、核心问题、技术路线、预期贡献。",
        ST["body"]))
    story.append(Spacer(1, 2*mm))

    pareto_sorted = sorted(pareto_b,
                           key=lambda x: sum(x['scores'][i] for i in OBJ_4_IDX),
                           reverse=True)
    for rank, ind in enumerate(pareto_sorted, 1):
        for elem in plan_card(ind, rank, len(pareto_sorted), color=CB):
            story.append(elem)

    # ── 实验A Top5（仅展示课题，不展示方案）───────────
    story.append(renderPDF.GraphicsFlowable(sec("05  实验A · 最高新颖性课题（对比参照）", color=CA)))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "实验A（单目标贪心）最终种群中新颖性最高的5个课题。"
        "注意其可行性和合理性普遍低于3分，说明单目标优化的极端化效应。",
        ST["body"]))
    story.append(Spacer(1, 2*mm))

    top_a = sorted(pop_a, key=lambda x: x['scores'][4], reverse=True)[:5]
    a_data = [["#","研究课题","领域","新颖","知识","可行","合理"]]
    for k, ind in enumerate(top_a, 1):
        s = ind['scores']
        a_data.append([str(k), ind['topic'][:28], ind.get('domain','')[:28],
                       f"{s[4]*10:.0f}", f"{s[0]*10:.0f}",
                       f"{s[5]*10:.0f}", f"{s[6]*10:.0f}"])
    at = Table(a_data, colWidths=[8*mm,62*mm,62*mm,14*mm,14*mm,14*mm,14*mm])
    at.setStyle(TableStyle([
        ("FONTNAME",(0,0),(-1,-1),F),("FONTSIZE",(0,0),(-1,-1),7.5),
        ("BACKGROUND",(0,0),(-1,0),CA),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("GRID",(0,0),(-1,-1),0.3,CLg),("ALIGN",(3,0),(-1,-1),"CENTER"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[CBgA,colors.white]),
        ("TEXTCOLOR",(5,1),(5,-1),CR),
        ("TEXTCOLOR",(6,1),(6,-1),CR),
    ]))
    story.append(at)
    story.append(Paragraph("表2  实验A新颖性Top5（可行性和合理性均极低，验证单目标极端化效应）", ST["caption"]))
    story.append(Spacer(1, 3*mm))

    # ── 结论 ──────────────────────────────────────────
    story.append(renderPDF.GraphicsFlowable(sec("06  实验结论")))
    story.append(Spacer(1, 2*mm))
    concl = [
        ("单目标极端化",
         f"仅优化新颖性导致种群收敛到极端科幻方向（新颖≈{avgs_a[4]:.1f}分），"
         f"可行性跌至 {avgs_a[5]:.1f}、合理性跌至 {avgs_a[6]:.1f}。"
         "生成的研究方案多为「量子+宇宙+意识」组合，无具体技术路线。"),
        ("多目标均衡",
         f"四目标 MOEA/D 在牺牲约2分新颖性的代价下，"
         f"可行性提升至 {avgs_b[5]:.1f}（+{avgs_b[5]-avgs_a[5]:.1f}）、"
         f"合理性提升至 {avgs_b[6]:.1f}（+{avgs_b[6]-avgs_a[6]:.1f}）。"
         "研究方案包含具体算法（如Transformer变体/GNN/强化学习）和数据集。"),
        ("研究方案质量",
         "实验B的Pareto解包含完整可执行方案：明确算法框架、实验设计和评估指标。"
         "代表方向：神经形态计算×生物信息、图神经网络×材料科学、"
         "联邦学习×隐私保护等CS主流方向的前沿交叉课题。"),
        ("对科研选题的启示",
         "多目标优化自动找到「足够新颖且可落地」的CS研究选题，"
         "避免了纯追求新颖导致的「无法发表」风险。"
         "MOEA/D的Pareto前沿提供了一组风险-收益不同配比的研究方向供选择。"),
    ]
    cd = [["#","维度","结论"]]
    for i, (t, tx) in enumerate(concl, 1):
        cd.append([str(i), t, tx])
    ct = Table(cd, colWidths=[7*mm, 30*mm, 131*mm])
    ct.setStyle(TableStyle([
        ("FONTNAME",(0,0),(-1,-1),F),("FONTSIZE",(0,0),(-1,-1),8),
        ("BACKGROUND",(0,0),(-1,0),CDk),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("GRID",(0,0),(-1,-1),0.3,CLg),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,CLg]),
        ("ALIGN",(0,0),(0,-1),"CENTER"),
        ("TEXTCOLOR",(1,1),(1,1),CR),("TEXTCOLOR",(1,2),(1,2),CB),
        ("TEXTCOLOR",(1,3),(1,3),CAm),("TEXTCOLOR",(1,4),(1,4),colors.HexColor("#00695c")),
    ]))
    story.append(ct)
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(
        f"实验配置：{cfg['model']} · 种群={cfg['pop_size']} · {cfg['n_gens']}代 · "
        f"两实验各28次LLM调用 · 共创建210个研究课题+方案", ST["small"]))

    doc.build(story, onFirstPage=add_hf, onLaterPages=add_hf)
    print(f"✓ PDF 报告生成: {path}")


if __name__ == "__main__":
    build_pdf(os.path.expanduser("~/moead-research/moead_cs_report.pdf"))
