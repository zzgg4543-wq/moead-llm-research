#!/usr/bin/env python3
"""
MOEA/D 科研知识多目标权衡关系分析报告 PDF 生成器
"""

import json, numpy as np, os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, KeepTogether)
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import (Drawing, Rect, Line, Circle, String,
                                       Polygon, Group)
from reportlab.graphics import renderPDF

# ── 字体 ──────────────────────────────────────────────
FONT_PATH = "/Library/Fonts/Arial Unicode.ttf"
pdfmetrics.registerFont(TTFont("U", FONT_PATH))
F = "U"

# ── 颜色 ──────────────────────────────────────────────
C_BLUE   = colors.HexColor("#1565c0")
C_TEAL   = colors.HexColor("#00695c")
C_AMBER  = colors.HexColor("#e65100")
C_PURPLE = colors.HexColor("#6a1b9a")
C_RED    = colors.HexColor("#c62828")
C_GREEN  = colors.HexColor("#2e7d32")
C_BG     = colors.HexColor("#e8f5e9")
C_BG2    = colors.HexColor("#e3f2fd")
C_DARK   = colors.HexColor("#212121")
C_GRAY   = colors.HexColor("#546e7a")
C_LGRAY  = colors.HexColor("#eceff1")

OBJ_COLORS = [
    colors.HexColor("#1565c0"),  # 知识价值
    colors.HexColor("#00838f"),  # 社会影响
    colors.HexColor("#558b2f"),  # 长尾度
    colors.HexColor("#6a1b9a"),  # 跨学科性
    colors.HexColor("#e65100"),  # 前沿性
    colors.HexColor("#c62828"),  # 可行性
    colors.HexColor("#ad1457"),  # 合理性
]
OBJ_NAMES = ["知识价值", "社会影响", "长尾度", "跨学科性", "前沿性", "可行性", "合理性"]
OBJ_SHORT = ["知识", "社会", "长尾", "跨学", "前沿", "可行", "合理"]

W, H = A4
PW = W - 28*mm  # printable width

# ── 样式 ──────────────────────────────────────────────
def make_styles():
    def ps(name, fontSize, textColor, leading=14, **kw):
        return ParagraphStyle(name, fontName=F, fontSize=fontSize,
                              textColor=textColor, leading=leading, **kw)
    return {
        "title":   ps("title",  20, C_BLUE,  22, spaceAfter=4, spaceBefore=8, alignment=1),
        "sub":     ps("sub",    11, C_GRAY,  16, spaceAfter=2, alignment=1),
        "h1":      ps("h1",     13, C_BLUE,  18, spaceBefore=10, spaceAfter=4),
        "h2":      ps("h2",   10.5, C_TEAL,  15, spaceBefore=6, spaceAfter=3),
        "body":    ps("body",  8.5, C_DARK,  13, spaceAfter=3),
        "small":   ps("small", 7.5, C_GRAY,  12, spaceAfter=2),
        "label":   ps("label",   8, C_DARK,  12),
        "bold":    ps("bold",    9, C_DARK,  13, spaceAfter=2),
        "caption": ps("caption",7.5,C_GRAY,  11, alignment=1, spaceAfter=4),
    }

ST = make_styles()

def section_rule(title, color=C_BLUE):
    d = Drawing(PW, 18)
    d.add(Rect(0, 4, PW, 14, fillColor=color, strokeColor=None))
    d.add(String(6, 7, title, fontName=F, fontSize=10.5,
                 fillColor=colors.white))
    return d

# ═══════════════════════════════════════════════════════
#  图形组件
# ═══════════════════════════════════════════════════════

def heatmap_drawing(corr, labels, w=PW, h=None):
    """相关系数热图"""
    n = len(labels)
    h = h or (w * 0.6)
    cell = min(w, h) / (n + 1.5)
    ox = (w - cell * n) / 2
    oy = 8
    d = Drawing(w, cell * n + oy + 20)

    def r_color(r):
        if r > 0:
            t = r
            return colors.Color(1 - 0.6*t, 1 - 0.3*t, 1)
        else:
            t = -r
            return colors.Color(1, 1 - 0.6*t, 1 - 0.6*t)

    for i in range(n):
        for j in range(n):
            x = ox + j * cell
            y = oy + (n - 1 - i) * cell
            r = corr[i, j]
            fc = r_color(r)
            d.add(Rect(x, y, cell - 1, cell - 1,
                       fillColor=fc, strokeColor=colors.white, strokeWidth=0.5))
            txt = f"{r:+.2f}" if i != j else "1.00"
            fc_txt = colors.black if abs(r) < 0.6 else colors.white
            d.add(String(x + cell/2, y + cell/2 - 3.5, txt,
                         fontName=F, fontSize=max(5.5, cell * 0.22),
                         fillColor=fc_txt, textAnchor="middle"))

    # 轴标签
    for i, lb in enumerate(labels):
        x = ox + i * cell + cell / 2
        y = oy + n * cell + 2
        d.add(String(x, y, lb, fontName=F, fontSize=max(6, cell*0.24),
                     fillColor=C_DARK, textAnchor="middle"))
        x2 = ox - 4
        y2 = oy + (n - 1 - i) * cell + cell / 2 - 3.5
        d.add(String(x2, y2, lb, fontName=F, fontSize=max(6, cell*0.24),
                     fillColor=C_DARK, textAnchor="end"))
    return d


def structure_diagram(w=PW, h=90):
    """四目标结构示意图（探索轴 vs 验证轴）"""
    d = Drawing(w, h)
    cx, cy = w / 2, h / 2

    # 四个节点
    nodes = {
        "长尾度": (cx - 100, cy + 20),
        "前沿性": (cx + 100, cy + 20),
        "可行性": (cx - 100, cy - 20),
        "合理性": (cx + 100, cy - 20),
    }
    node_colors = {
        "长尾度": colors.HexColor("#558b2f"),
        "前沿性": colors.HexColor("#e65100"),
        "可行性": colors.HexColor("#c62828"),
        "合理性": colors.HexColor("#ad1457"),
    }
    # 关系线
    edges = [
        ("长尾度", "前沿性",  "+0.33", C_GREEN,  True,  "协同（探索轴）"),
        ("可行性", "合理性",  "+0.69", C_BLUE,   True,  "强协同（验证轴）"),
        ("长尾度", "可行性",  "−0.50", C_RED,    False, "强对抗"),
        ("前沿性", "可行性",  "−0.45", C_AMBER,  False, "对抗"),
        ("合理性", "长尾度",  "−0.49", C_PURPLE, False, "对抗"),
        ("前沿性", "合理性",  "−0.09", C_GRAY,   None,  "独立轴"),
    ]
    for (a, b, label, col, is_synergy, desc) in edges:
        x1, y1 = nodes[a]
        x2, y2 = nodes[b]
        dash = [3, 3] if is_synergy is None else ([] if is_synergy else [5, 2])
        lw = 2.5 if is_synergy is True else (1.5 if is_synergy is None else 2)
        d.add(Line(x1, y1, x2, y2, strokeColor=col, strokeWidth=lw,
                   strokeDashArray=dash))
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        offset_y = 5 if y1 == y2 else 0
        offset_x = 5 if x1 == x2 else 0
        d.add(String(mx + offset_x, my + offset_y + 2, label,
                     fontName=F, fontSize=7, fillColor=col, textAnchor="middle"))

    # 画节点
    r = 22
    for name, (nx, ny) in nodes.items():
        d.add(Circle(nx, ny, r, fillColor=node_colors[name],
                     strokeColor=colors.white, strokeWidth=1.5))
        d.add(String(nx, ny - 3.5, name, fontName=F, fontSize=7.5,
                     fillColor=colors.white, textAnchor="middle"))

    # 图例
    lx, ly = 14, 8
    for lbl, col, dash in [("协同（实线）", C_GREEN, []), ("独立（虚线）", C_GRAY, [3,3]),
                            ("对抗（粗线）", C_RED, [5,2])]:
        d.add(Line(lx, ly+4, lx+18, ly+4, strokeColor=col, strokeWidth=2,
                   strokeDashArray=dash))
        d.add(String(lx+22, ly+1, lbl, fontName=F, fontSize=6.5, fillColor=C_DARK))
        lx += 68
    return d


def bar_comparison(w=PW, h=80):
    """v2 vs v3 四目标均值对比柱状图"""
    d = Drawing(w, h)
    focus = ["长尾度", "前沿性", "可行性", "合理性"]
    v2_vals = [9.1, 8.2, 4.7, 5.9]
    v3_vals = [8.3, 7.5, 7.4, 7.0]
    fc = [colors.HexColor("#558b2f"), colors.HexColor("#e65100"),
          colors.HexColor("#c62828"), colors.HexColor("#ad1457")]

    n = len(focus)
    group_w = (w - 40) / n
    bar_w = group_w * 0.32
    ox, oy = 30, 10
    chart_h = h - oy - 14

    # y 轴 (0-10)
    for y_val in [0, 5, 10]:
        y = oy + (y_val / 10) * chart_h
        d.add(Line(ox - 4, y, w - 8, y, strokeColor=C_LGRAY, strokeWidth=0.5))
        d.add(String(ox - 6, y - 3, str(y_val), fontName=F, fontSize=6,
                     fillColor=C_GRAY, textAnchor="end"))

    for i, (name, v2, v3, color) in enumerate(zip(focus, v2_vals, v3_vals, fc)):
        gx = ox + i * group_w + group_w / 2
        # v2 柱（半透明蓝）
        bx2 = gx - bar_w - 1
        bh2 = (v2 / 10) * chart_h
        d.add(Rect(bx2, oy, bar_w, bh2,
                   fillColor=colors.Color(0.7, 0.7, 0.9, 0.7), strokeColor=None))
        d.add(String(bx2 + bar_w/2, oy + bh2 + 1.5, f"{v2:.1f}",
                     fontName=F, fontSize=6, fillColor=C_GRAY, textAnchor="middle"))
        # v3 柱
        bx3 = gx + 1
        bh3 = (v3 / 10) * chart_h
        d.add(Rect(bx3, oy, bar_w, bh3, fillColor=color, strokeColor=None))
        d.add(String(bx3 + bar_w/2, oy + bh3 + 1.5, f"{v3:.1f}",
                     fontName=F, fontSize=6.5, fillColor=color, textAnchor="middle"))
        # 标签
        d.add(String(gx, oy - 9, name, fontName=F, fontSize=7,
                     fillColor=C_DARK, textAnchor="middle"))
        # 变化箭头
        delta = v3 - v2
        arr_color = C_GREEN if delta > 0 else C_RED
        sign = "▲" if delta > 0 else "▼"
        d.add(String(gx + bar_w/2 + 4, oy + 2, f"{sign}{abs(delta):.1f}",
                     fontName=F, fontSize=6, fillColor=arr_color, textAnchor="start"))

    # 图例
    d.add(Rect(w-80, h-12, 10, 8, fillColor=colors.Color(0.7,0.7,0.9,0.7), strokeColor=None))
    d.add(String(w-67, h-13+2, "v2（无约束）", fontName=F, fontSize=6.5, fillColor=C_GRAY))
    d.add(Rect(w-80, h-22, 10, 8, fillColor=C_TEAL, strokeColor=None))
    d.add(String(w-67, h-23+2, "v3（强化约束）", fontName=F, fontSize=6.5, fillColor=C_DARK))
    return d


def quadrant_drawing(arr, w=PW, h=None):
    """四象限散点图：长尾度 x 可行性，颜色=合理性，大小=前沿性"""
    h = h or w * 0.55
    d = Drawing(w, h)
    ox, oy = 36, 14
    cw, ch = w - ox - 14, h - oy - 22

    # 背景象限
    mid_x = ox + cw / 2
    mid_y = oy + ch / 2
    quads = [
        (ox, mid_y, cw/2, ch/2, colors.HexColor("#fff3e0"), "高长尾+高可行\n「黄金三角型」"),
        (mid_x, mid_y, cw/2, ch/2, colors.HexColor("#fce4ec"), "低长尾+高可行\n「成熟研究型」"),
        (ox, oy, cw/2, ch/2, colors.HexColor("#e8f5e9"), "高长尾+低可行\n「天马行空型」"),
        (mid_x, oy, cw/2, ch/2, colors.HexColor("#f3e5f5"), "低长尾+低可行\n「理论假说型」"),
    ]
    for (qx, qy, qw, qh, qc, qlabel) in quads:
        d.add(Rect(qx, qy, qw, qh, fillColor=qc, strokeColor=C_LGRAY, strokeWidth=0.5))
        lines = qlabel.split("\n")
        for k, line in enumerate(lines):
            d.add(String(qx + qw/2, qy + qh/2 - 4 + k*9, line,
                         fontName=F, fontSize=6.5, fillColor=C_GRAY, textAnchor="middle"))

    # 中心线
    d.add(Line(mid_x, oy, mid_x, oy+ch, strokeColor=C_GRAY, strokeWidth=0.5, strokeDashArray=[3,3]))
    d.add(Line(ox, mid_y, ox+cw, mid_y, strokeColor=C_GRAY, strokeWidth=0.5, strokeDashArray=[3,3]))

    IDX = {n:i for i,n in enumerate(OBJ_NAMES)}
    for s in arr:
        x = ox + (s[IDX["长尾度"]] - 0.6) / 0.45 * cw  # 长尾度范围约 0.6-1.05
        y = oy + (s[IDX["可行性"]]       ) / 1.0  * ch  # 可行性 0-1
        x = max(ox+2, min(ox+cw-2, x))
        y = max(oy+2, min(oy+ch-2, y))
        ri = s[IDX["合理性"]]
        fr = s[IDX["前沿性"]]
        # 颜色按合理性（红低→绿高）
        g = ri * 0.8
        pt_color = colors.Color(1 - g, g * 0.7, g * 0.4)
        radius = 2.5 + fr * 3.5  # 大小按前沿性
        d.add(Circle(x, y, radius, fillColor=pt_color,
                     strokeColor=colors.white, strokeWidth=0.5))

    # 轴标签
    d.add(String(ox + cw/2, oy - 13, "← 长尾度（左=低，右=高）→",
                 fontName=F, fontSize=7, fillColor=C_DARK, textAnchor="middle"))
    for y_frac, label in [(0, "0"), (0.5, "5"), (1.0, "10")]:
        y = oy + y_frac * ch
        d.add(String(ox - 4, y - 3, label, fontName=F, fontSize=6,
                     fillColor=C_GRAY, textAnchor="end"))
    d.add(String(ox - 28, oy + ch/2, "可行性↑", fontName=F, fontSize=7,
                 fillColor=C_DARK, textAnchor="middle"))

    # 图例（颜色=合理性）
    lx = ox + cw + 2
    for i, (lbl, r_val) in enumerate([("合理低", 0.3), ("合理中", 0.5), ("合理高", 0.8)]):
        g = r_val * 0.8
        ly = oy + ch - 8 - i * 16
        d.add(Circle(lx + 5, ly + 3, 5, fillColor=colors.Color(1-g, g*0.7, g*0.4),
                     strokeColor=colors.white))
        d.add(String(lx + 13, ly, lbl, fontName=F, fontSize=6, fillColor=C_DARK))
    d.add(String(lx, oy + ch - 60, "圆大=前沿高", fontName=F, fontSize=5.5, fillColor=C_GRAY))
    return d


def radar_pareto(cases, w=None, h=80):
    """雷达图展示 C 类圣杯型个体的 4 目标轮廓"""
    w = w or PW * 0.5
    d = Drawing(w, h)
    focus_idx = [2, 4, 5, 6]  # 长尾, 前沿, 可行, 合理
    labels = ["长尾度", "前沿性", "可行性", "合理性"]
    n = len(focus_idx)
    import math
    cx, cy = w / 2, h / 2
    r_max = min(cx, cy) - 16

    # 背景蜘蛛网
    for lv in [0.25, 0.5, 0.75, 1.0]:
        pts = []
        for k in range(n):
            angle = math.pi/2 + 2*math.pi*k/n
            pts += [cx + r_max*lv*math.cos(angle), cy + r_max*lv*math.sin(angle)]
        pts += [pts[0], pts[1]]
        for i in range(0, len(pts)-2, 2):
            d.add(Line(pts[i], pts[i+1], pts[i+2], pts[i+3],
                       strokeColor=C_LGRAY, strokeWidth=0.5))
    for k in range(n):
        angle = math.pi/2 + 2*math.pi*k/n
        d.add(Line(cx, cy, cx + r_max*math.cos(angle), cy + r_max*math.sin(angle),
                   strokeColor=C_LGRAY, strokeWidth=0.5))
        lx = cx + (r_max + 10)*math.cos(angle)
        ly = cy + (r_max + 10)*math.sin(angle)
        d.add(String(lx, ly-3.5, labels[k], fontName=F, fontSize=6.5,
                     fillColor=C_DARK, textAnchor="middle"))

    # 绘制各类型
    type_styles = [
        ("B类-黄金三角", [8.0, 6.5, 8.5, 7.5], colors.HexColor("#ff8f00")),
        ("C类-理论圣杯", [9.0, 9.0, 6.0, 8.0], colors.HexColor("#1565c0")),
        ("A类-天马行空", [9.5, 8.0, 2.0, 4.5], colors.HexColor("#c62828")),
    ]
    for (lbl, vals, col) in type_styles:
        pts = []
        for k, v in enumerate(vals):
            angle = math.pi/2 + 2*math.pi*k/n
            rk = (v / 10) * r_max
            pts += [cx + rk*math.cos(angle), cy + rk*math.sin(angle)]
        pts += [pts[0], pts[1]]
        for i in range(0, len(pts)-2, 2):
            d.add(Line(pts[i], pts[i+1], pts[i+2], pts[i+3],
                       strokeColor=col, strokeWidth=2))

    # 图例
    lx, ly_base = 4, 12
    for (lbl, _, col) in type_styles:
        d.add(Line(lx, ly_base+3, lx+14, ly_base+3, strokeColor=col, strokeWidth=2))
        d.add(String(lx+17, ly_base, lbl, fontName=F, fontSize=6.5, fillColor=C_DARK))
        ly_base += 12
    return d


def score_trend_drawing(gen_stats, w=PW, h=70):
    """可行性和合理性的演化趋势折线图（v2 vs v3）"""
    d = Drawing(w, h)
    ox, oy = 36, 10
    cw, ch = w - ox - 16, h - oy - 18

    # 只展示4个关注目标
    focus = [("长尾度", 2, C_GREEN, [3,2]), ("前沿性", 4, C_AMBER, [3,2]),
             ("可行性", 5, C_RED, []), ("合理性", 6, colors.HexColor("#ad1457"), [])]
    max_gen = len(gen_stats) - 1

    for y_val in [4, 6, 8, 10]:
        y = oy + ((y_val - 3) / 7) * ch
        d.add(Line(ox, y, ox+cw, y, strokeColor=C_LGRAY, strokeWidth=0.5))
        d.add(String(ox-4, y-3, str(y_val), fontName=F, fontSize=6,
                     fillColor=C_GRAY, textAnchor="end"))

    for (name, idx, col, dash) in focus:
        pts = [(ox + g*(cw/max_gen),
                oy + ((gen_stats[g]['avgs'][idx] - 3)/7)*ch)
               for g in range(len(gen_stats))]
        for i in range(len(pts)-1):
            d.add(Line(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1],
                       strokeColor=col, strokeWidth=1.8, strokeDashArray=dash))
        d.add(String(pts[-1][0]+2, pts[-1][1]-3, name,
                     fontName=F, fontSize=6, fillColor=col))

    # x 轴刻度
    for g in range(0, max_gen+1, 2):
        x = ox + g*(cw/max_gen)
        d.add(Line(x, oy-2, x, oy, strokeColor=C_GRAY, strokeWidth=0.5))
        d.add(String(x, oy-10, f"{g}代", fontName=F, fontSize=6,
                     fillColor=C_GRAY, textAnchor="middle"))
    return d


def page_header(canvas, doc):
    canvas.saveState()
    canvas.setFont(F, 7.5)
    canvas.setFillColor(C_GRAY)
    canvas.drawString(14*mm, H - 9*mm, "MOEA/D 科研知识多目标权衡分析报告")
    canvas.drawRightString(W - 14*mm, H - 9*mm, f"第 {doc.page} 页")
    canvas.setStrokeColor(C_LGRAY)
    canvas.setLineWidth(0.5)
    canvas.line(14*mm, H - 10*mm, W - 14*mm, H - 10*mm)
    canvas.restoreState()

def page_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(F, 7)
    canvas.setFillColor(C_GRAY)
    canvas.drawCentredString(W/2, 7*mm, "MOEA/D × Claude  ·  跨学科长尾科研知识演化系统  ·  权衡分析报告")
    canvas.restoreState()

def add_hf(canvas, doc):
    page_header(canvas, doc)
    page_footer(canvas, doc)


# ═══════════════════════════════════════════════════════
#  数据准备
# ═══════════════════════════════════════════════════════
def load_data():
    all_inds = []
    gen_stats_v3 = None
    for f in ["/Users/zhe/moead_science_v2_results.json",
              "/Users/zhe/moead_science_v3_results.json"]:
        with open(f) as fp:
            d = json.load(fp)
        for src in [d['init_pop'], d['final_pop']]:
            all_inds += src
        if "v3" in f:
            gen_stats_v3 = d['gen_stats']

    arr = np.array([x['scores'] for x in all_inds])
    arr_u, ui = np.unique(arr, axis=0, return_index=True)
    inds_u = [all_inds[i] for i in ui]

    pareto_all = []
    for f in ["/Users/zhe/moead_science_v2_results.json",
              "/Users/zhe/moead_science_v3_results.json"]:
        with open(f) as fp:
            d = json.load(fp)
        pareto_all += d['pareto']

    return arr_u, inds_u, pareto_all, gen_stats_v3


# ═══════════════════════════════════════════════════════
#  主构建函数
# ═══════════════════════════════════════════════════════
def build_pdf(path):
    arr, inds, pareto_all, gen_stats = load_data()
    IDX = {n:i for i,n in enumerate(OBJ_NAMES)}
    corr = np.corrcoef(arr.T)

    doc = SimpleDocTemplate(path, pagesize=A4,
                            leftMargin=14*mm, rightMargin=14*mm,
                            topMargin=14*mm, bottomMargin=14*mm)
    story = []

    # ─── 封面 ────────────────────────────────────────────
    story.append(Spacer(1, 12*mm))
    story.append(Paragraph("科研长尾知识多目标权衡关系", ST["title"]))
    story.append(Paragraph("深度分析报告", ST["title"]))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "基于 MOEA/D × Claude 演化系统  ·  v2（7目标版）+ v3（可行性强化版）",
        ST["sub"]))
    story.append(Paragraph(
        f"样本量：{len(arr)} 个多样化长尾研究知识  ·  Pareto 解集：{len(pareto_all)} 个",
        ST["sub"]))
    story.append(Spacer(1, 4*mm))

    # 核心发现摘要卡片
    summary_data = [
        ["核心发现", "r 值", "类型", "含义"],
        ["可行性 ↔ 合理性", "+0.685", "强协同", "验证轴：理论严密 ⟺ 方法可行"],
        ["长尾度 ↔ 可行性", "−0.502", "强对抗", "最核心权衡：越小众越难落地"],
        ["合理性 ↔ 长尾度", "−0.489", "对抗",   "主流框架严密，但往往不够长尾"],
        ["前沿性 ↔ 可行性", "−0.447", "对抗",   "突破性想法领先于实验能力"],
        ["长尾度 ↔ 前沿性", "+0.325", "弱协同", "探索轴：小众方向天然趋向前沿"],
        ["前沿性 ↔ 合理性", "−0.088", "独立",   "★ 最反直觉：前沿不必牺牲严谨"],
    ]
    col_w = [56*mm, 22*mm, 22*mm, 68*mm]
    st = Table(summary_data, colWidths=col_w)
    row_colors = [
        colors.HexColor("#1565c0"),
        colors.HexColor("#1b5e20"),
        colors.HexColor("#b71c1c"),
        colors.HexColor("#880e4f"),
        colors.HexColor("#e65100"),
        colors.HexColor("#33691e"),
        colors.HexColor("#4a148c"),
    ]
    t_style = [
        ("FONTNAME",      (0,0), (-1,-1), F),
        ("FONTSIZE",      (0,0), (-1,-1), 8),
        ("BACKGROUND",    (0,0), (-1,0),  colors.HexColor("#1a237e")),
        ("TEXTCOLOR",     (0,0), (-1,0),  colors.white),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ("GRID",          (0,0), (-1,-1), 0.3, colors.HexColor("#cfd8dc")),
        ("ALIGN",         (1,0), (2,-1),  "CENTER"),
    ]
    light_bgs = [
        colors.HexColor("#e8f5e9"),  # 绿
        colors.HexColor("#fce4ec"),  # 红
        colors.HexColor("#fce4ec"),  # 红
        colors.HexColor("#fff3e0"),  # 橙
        colors.HexColor("#e8f5e9"),  # 绿
        colors.HexColor("#ede7f6"),  # 紫
    ]
    for row_i in range(1, len(summary_data)):
        t_style.append(("BACKGROUND", (0,row_i), (-1,row_i), light_bgs[row_i-1]))
        t_style.append(("TEXTCOLOR",  (1,row_i), (1,row_i), row_colors[row_i]))
        t_style.append(("FONTNAME",   (1,row_i), (1,row_i), F))
    st.setStyle(TableStyle(t_style))
    story.append(st)
    story.append(Spacer(1, 4*mm))

    # ─── 1. 目标结构示意图 ────────────────────────────────
    story.append(renderPDF.GraphicsFlowable(section_rule("01  四目标结构：探索轴 vs 验证轴")))
    story.append(Spacer(1, 2*mm))

    story.append(Paragraph(
        "实验数据揭示出四个核心目标形成清晰的<b>二维拉力场</b>："
        "<font color='#558b2f'>长尾度</font>与<font color='#e65100'>前沿性</font>构成「探索轴」（正相关，r=+0.33）；"
        "<font color='#c62828'>可行性</font>与<font color='#ad1457'>合理性</font>构成「验证轴」（强正相关，r=+0.69）。"
        "两轴之间存在系统性对抗关系。",
        ST["body"]))
    story.append(Spacer(1, 2*mm))
    story.append(renderPDF.GraphicsFlowable(structure_diagram(w=PW, h=100)))
    story.append(Paragraph("图1  四目标结构关系示意图（节点大小示意相对重要性，线型/颜色示关系类型）", ST["caption"]))
    story.append(Spacer(1, 3*mm))

    # ─── 2. 相关系数热图 ──────────────────────────────────
    story.append(renderPDF.GraphicsFlowable(section_rule("02  全目标相关系数热图（65个样本）")))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "热图展示7个目标两两之间的 Pearson 相关系数。"
        "深蓝=强正相关（协同），深红=强负相关（权衡），白色=独立。",
        ST["body"]))
    story.append(Spacer(1, 2*mm))
    story.append(renderPDF.GraphicsFlowable(heatmap_drawing(corr, OBJ_SHORT, w=PW, h=PW*0.55)))
    story.append(Paragraph("图2  7目标两两 Pearson 相关系数热图", ST["caption"]))
    story.append(Spacer(1, 3*mm))

    # ─── 3. 四象限散点图 ──────────────────────────────────
    story.append(renderPDF.GraphicsFlowable(section_rule("03  四象限分布：长尾度 × 可行性")))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "以<b>长尾度</b>（x轴）和<b>可行性</b>（y轴）定义四象限。"
        "每个点的<b>颜色</b>代表合理性（红低→绿高），<b>圆圈大小</b>代表前沿性。",
        ST["body"]))
    story.append(Spacer(1, 2*mm))
    story.append(renderPDF.GraphicsFlowable(quadrant_drawing(arr, w=PW, h=PW*0.52)))
    story.append(Paragraph("图3  四象限散点图（颜色=合理性，大小=前沿性）", ST["caption"]))
    story.append(Spacer(1, 2*mm))

    # 四象限统计表
    lt = arr[:, IDX["长尾度"]]; fe = arr[:, IDX["可行性"]]
    fr = arr[:, IDX["前沿性"]]; ri = arr[:, IDX["合理性"]]
    lt_med, fe_med = np.median(lt), np.median(fe)
    masks = [(lt>=lt_med)&(fe>=fe_med), (lt>=lt_med)&(fe<fe_med),
             (lt< lt_med)&(fe>=fe_med), (lt< lt_med)&(fe<fe_med)]
    quad_data = [
        ["象限", "特征", "样本量", "前沿性均值", "合理性均值", "代表类型"],
        ["高长尾+高可行", "长尾≥8.5, 可行≥5", f"{masks[0].sum()}",
         f"{fr[masks[0]].mean()*10:.1f}", f"{ri[masks[0]].mean()*10:.1f}", "★ 黄金三角型"],
        ["高长尾+低可行", "长尾≥8.5, 可行<5", f"{masks[1].sum()}",
         f"{fr[masks[1]].mean()*10:.1f}", f"{ri[masks[1]].mean()*10:.1f}", "天马行空型"],
        ["低长尾+高可行", "长尾<8.5, 可行≥5", f"{masks[2].sum()}",
         f"{fr[masks[2]].mean()*10:.1f}", f"{ri[masks[2]].mean()*10:.1f}", "成熟研究型"],
        ["低长尾+低可行", "长尾<8.5, 可行<5", f"{masks[3].sum()}",
         f"{fr[masks[3]].mean()*10:.1f}", f"{ri[masks[3]].mean()*10:.1f}", "理论假说型"],
    ]
    qt = Table(quad_data, colWidths=[38*mm, 32*mm, 16*mm, 22*mm, 22*mm, 38*mm])
    q_colors = [C_AMBER, colors.HexColor("#c62828"), C_TEAL, C_PURPLE]
    qt.setStyle(TableStyle([
        ("FONTNAME",  (0,0), (-1,-1), F),
        ("FONTSIZE",  (0,0), (-1,-1), 7.5),
        ("BACKGROUND",(0,0), (-1,0),  C_DARK),
        ("TEXTCOLOR", (0,0), (-1,0),  colors.white),
        ("GRID",      (0,0), (-1,-1), 0.3, C_LGRAY),
        ("VALIGN",    (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1), 2),
        ("BOTTOMPADDING",(0,0),(-1,-1), 2),
        ("BACKGROUND",(0,1), (-1,1), colors.HexColor("#fff8e1")),
        ("BACKGROUND",(0,2), (-1,2), colors.HexColor("#fce4ec")),
        ("BACKGROUND",(0,3), (-1,3), colors.HexColor("#e8f5e9")),
        ("BACKGROUND",(0,4), (-1,4), colors.HexColor("#f3e5f5")),
        ("TEXTCOLOR", (5,1), (5,1), C_AMBER),
    ]))
    story.append(qt)
    story.append(Paragraph("表1  四象限统计（以长尾度/可行性中位数划分）", ST["caption"]))
    story.append(Spacer(1, 3*mm))

    # ─── 4. 最反直觉发现 ──────────────────────────────────
    story.append(renderPDF.GraphicsFlowable(section_rule("04  最反直觉发现：前沿性 ↔ 合理性 独立（r=−0.09）")))
    story.append(Spacer(1, 2*mm))

    # 雷达图 + 文字并排
    radar = radar_pareto(None, w=PW*0.48, h=90)
    insight_text = Table([[
        renderPDF.GraphicsFlowable(radar),
        Paragraph(
            "<b>前沿性与合理性几乎不相关（r=−0.088）</b>，"
            "这与「大胆创新必然不严谨」的直觉相反。<br/><br/>"
            "真正制约两者共存的是<b>可行性瓶颈</b>：<br/>"
            "高前沿+高合理的研究虽然概念上自洽，<br/>"
            "但当前技术往往无法同时撑起「新颖」+「可验证」。<br/><br/>"
            "65个样本中只有<b>1个「理论圣杯型」</b>个体：<br/>"
            "<font color='#1565c0'>「古DNA表观遗传印记重建」</font><br/>"
            "长尾=9  前沿=9  可行=6  合理=8<br/><br/>"
            "这说明同时高前沿+高合理+高长尾是<b>Pareto 极端点</b>，"
            "极其罕见但确实存在。",
            ST["body"]),
    ]], colWidths=[PW*0.50, PW*0.48])
    insight_text.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(insight_text)
    story.append(Paragraph("图4  三类典型研究画像雷达图（长尾/前沿/可行/合理四维）", ST["caption"]))
    story.append(Spacer(1, 3*mm))

    # ─── 5. 演化趋势 ──────────────────────────────────────
    story.append(renderPDF.GraphicsFlowable(section_rule("05  v3 演化趋势（6代 · 四目标均值）")))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "强化约束算法（v3）运行 6 代的四目标均值演化曲线。"
        "实线=可行性/合理性（强化目标），虚线=长尾度/前沿性（被动调整目标）。",
        ST["body"]))
    story.append(Spacer(1, 2*mm))
    story.append(renderPDF.GraphicsFlowable(score_trend_drawing(gen_stats, w=PW, h=80)))
    story.append(Paragraph("图5  v3 四目标均值演化曲线（代数 0=初始种群）", ST["caption"]))
    story.append(Spacer(1, 3*mm))

    # ─── 6. v2 vs v3 对比 ────────────────────────────────
    story.append(renderPDF.GraphicsFlowable(section_rule("06  算法改进代价分析：v2 vs v3")))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "通过强制提升可行性+合理性（v3），演化方向从「宇宙量子假说」漂移到"
        "「冰川微生物基因组、古DNA表观遗传」等可落地方向。代价是长尾度和前沿性各降约 0.7 分。",
        ST["body"]))
    story.append(Spacer(1, 2*mm))
    story.append(renderPDF.GraphicsFlowable(bar_comparison(w=PW, h=90)))
    story.append(Paragraph("图6  v2 vs v3 四目标均值对比（深色=v3，浅色=v2，箭头=变化量）", ST["caption"]))
    story.append(Spacer(1, 3*mm))

    # 详细比较表
    cmp_data = [
        ["维度", "v2 无约束", "v3 强化约束", "变化", "解读"],
        ["长尾度均值", "9.1", "8.3", "−0.8", "略有下降，长尾性仍很高"],
        ["前沿性均值", "8.2", "7.5", "−0.7", "轻微代价，仍属高前沿"],
        ["可行性均值", "4.7", "7.4", "+2.7 ★", "大幅提升，核心改进目标"],
        ["合理性均值", "5.9", "7.0", "+1.1 ★", "显著提升，理论基础更扎实"],
        ["综合均值",   "7.1", "7.4", "+0.3", "整体质量提升"],
        ["Pareto解数", "17", "9",  "−8",   "约束收紧，解集规模缩小"],
        ["代表研究方向", "暗物质拓扑、量子生命起源", "古DNA表观、冰川微生物", "—", "从纯理论→可实验验证"],
    ]
    ct = Table(cmp_data, colWidths=[30*mm, 26*mm, 26*mm, 22*mm, 64*mm])
    ct.setStyle(TableStyle([
        ("FONTNAME",     (0,0), (-1,-1), F),
        ("FONTSIZE",     (0,0), (-1,-1), 7.5),
        ("BACKGROUND",   (0,0), (-1,0),  C_DARK),
        ("TEXTCOLOR",    (0,0), (-1,0),  colors.white),
        ("GRID",         (0,0), (-1,-1), 0.3, C_LGRAY),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",   (0,0), (-1,-1), 2),
        ("BOTTOMPADDING",(0,0), (-1,-1), 2),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, C_BG]),
        ("TEXTCOLOR",    (3,3), (3,3),   C_GREEN),
        ("TEXTCOLOR",    (3,4), (3,4),   C_GREEN),
        ("TEXTCOLOR",    (3,5), (3,6),   C_RED),
    ]))
    story.append(ct)
    story.append(Paragraph("表2  v2 与 v3 最终种群关键指标对比", ST["caption"]))
    story.append(Spacer(1, 3*mm))

    # ─── 7. Pareto 代表案例 ───────────────────────────────
    story.append(renderPDF.GraphicsFlowable(section_rule("07  Pareto 最优解：各目标极值代表")))
    story.append(Spacer(1, 2*mm))

    pa = np.array([p['scores'] for p in pareto_all])
    cases = [
        ("最高长尾度", IDX["长尾度"]),
        ("最高前沿性", IDX["前沿性"]),
        ("最高可行性", IDX["可行性"]),
        ("最高合理性", IDX["合理性"]),
    ]
    case_data = [["类型", "研究主题", "领域", "长尾", "前沿", "可行", "合理", "知识", "社会"]]
    for (label, obj_idx) in cases:
        best_i = int(np.argmax(pa[:, obj_idx]))
        ind = pareto_all[best_i]
        s = ind['scores']
        case_data.append([
            label, ind['topic'][:20], ind.get('domain','')[:22],
            f"{s[IDX['长尾度']]*10:.0f}", f"{s[IDX['前沿性']]*10:.0f}",
            f"{s[IDX['可行性']]*10:.0f}", f"{s[IDX['合理性']]*10:.0f}",
            f"{s[IDX['知识价值']]*10:.0f}", f"{s[IDX['社会影响']]*10:.0f}",
        ])
    ptable = Table(case_data, colWidths=[22*mm, 46*mm, 44*mm,
                                          11*mm, 11*mm, 11*mm, 11*mm, 11*mm, 11*mm])
    ptable.setStyle(TableStyle([
        ("FONTNAME",     (0,0), (-1,-1), F),
        ("FONTSIZE",     (0,0), (-1,-1), 7.5),
        ("BACKGROUND",   (0,0), (-1,0),  C_DARK),
        ("TEXTCOLOR",    (0,0), (-1,0),  colors.white),
        ("GRID",         (0,0), (-1,-1), 0.3, C_LGRAY),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",   (0,0), (-1,-1), 2),
        ("BOTTOMPADDING",(0,0), (-1,-1), 2),
        ("ALIGN",        (3,0), (-1,-1), "CENTER"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, C_BG2]),
        ("TEXTCOLOR",    (0,1), (0,1),   C_GREEN),
        ("TEXTCOLOR",    (0,2), (0,2),   C_AMBER),
        ("TEXTCOLOR",    (0,3), (0,3),   C_RED),
        ("TEXTCOLOR",    (0,4), (0,4),   colors.HexColor("#ad1457")),
    ]))
    story.append(ptable)
    story.append(Paragraph("表3  Pareto 前沿中各目标极值代表个体", ST["caption"]))
    story.append(Spacer(1, 3*mm))

    # ─── 8. 结论 ──────────────────────────────────────────
    story.append(renderPDF.GraphicsFlowable(section_rule("08  核心结论与科研选题启示")))
    story.append(Spacer(1, 2*mm))

    conclusions = [
        ("①", "最核心权衡",
         "长尾度 ↔ 可行性（r=−0.50）是整个系统中最难同时满足的对立关系。"
         "越小众的研究方向，越缺乏现成技术支撑——这是长尾知识的本质张力。"),
        ("②", "验证轴是统一维度",
         "可行性与合理性高度协同（r=+0.69），应作为整体「可研究性」评估，"
         "而非独立权衡。扎实的理论假设本身就需要具体实验设计来支撑。"),
        ("③", "最反直觉发现",
         "前沿性与合理性几乎独立（r=−0.09）。制约「高前沿+高合理」共存的"
         "真正瓶颈是可行性不足，而非两者之间的内在矛盾。"),
        ("④", "黄金选题策略",
         "寻找「当前技术刚好够得着的前沿」——即 C 类「理论圣杯型」研究。"
         "65个样本中仅1例（古DNA表观遗传印记重建），"
         "极其稀有但具有最高综合价值。"),
        ("⑤", "算法改进揭示真实代价",
         "强制提升可行性/合理性（v3）使两目标均值提升 +1.1~+2.7 分，"
         "但长尾度/前沿性各损失约 0.7 分。这个代价是真实且不可避免的——"
         "除非找到 C 类研究方向。"),
    ]
    con_data = [["", "维度", "结论"]]
    for num, title, text in conclusions:
        con_data.append([num, title, text])
    ct2 = Table(con_data, colWidths=[8*mm, 28*mm, 132*mm])
    ct2.setStyle(TableStyle([
        ("FONTNAME",     (0,0), (-1,-1), F),
        ("FONTSIZE",     (0,0), (-1,-1), 8),
        ("BACKGROUND",   (0,0), (-1,0),  C_DARK),
        ("TEXTCOLOR",    (0,0), (-1,0),  colors.white),
        ("GRID",         (0,0), (-1,-1), 0.3, C_LGRAY),
        ("VALIGN",       (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",   (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, C_BG]),
        ("ALIGN",        (0,0), (0,-1),  "CENTER"),
        ("FONTSIZE",     (0,1), (0,-1),  10),
        ("TEXTCOLOR",    (1,1), (1,1),   C_RED),
        ("TEXTCOLOR",    (1,2), (1,2),   C_BLUE),
        ("TEXTCOLOR",    (1,3), (1,3),   C_PURPLE),
        ("TEXTCOLOR",    (1,4), (1,4),   C_AMBER),
        ("TEXTCOLOR",    (1,5), (1,5),   C_GREEN),
    ]))
    story.append(ct2)
    story.append(Spacer(1, 4*mm))

    doc.build(story, onFirstPage=add_hf, onLaterPages=add_hf)
    print(f"✓ PDF 已生成: {path}")


if __name__ == "__main__":
    build_pdf(os.path.expanduser("~/moead_tradeoff_report.pdf"))
