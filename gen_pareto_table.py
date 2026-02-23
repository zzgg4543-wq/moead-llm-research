#!/usr/bin/env python3
"""从演化结果 JSON 生成 Pareto 排序表格 HTML"""
import json
from pathlib import Path

def gen_html(json_path: str, out_path: str, title: str = "Pareto 最优选项"):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    obj_names = [o["name"] for o in data["objectives"]]
    all_opts = data.get("all_options") or data["evaluated_options"]
    # 去重：按 (option_id, summary) 唯一条目
    seen_key = set()
    unique_opts = []
    for o in all_opts:
        c = o.get("content", {})
        s = c.get("summary", "")
        key = (o["option_id"], s)
        if key not in seen_key:
            seen_key.add(key)
            unique_opts.append(o)
    # 按 rank 排序，Pareto 在前
    unique_opts.sort(key=lambda x: (0 if x.get("pareto_front") else 1, x.get("rank", 999)))
    rows = []
    pareto_idx = 0
    other_idx = 0
    for o in unique_opts:
        c = o.get("content", {})
        sc = o.get("scores", [])
        sc_vals = [int(round(x * 10)) for x in sc[:len(obj_names)]]
        if o.get("pareto_front"):
            pareto_idx += 1
            rank_disp = f"★ {pareto_idx}"
        else:
            other_idx += 1
            rank_disp = str(pareto_idx + other_idx)
        rows.append({
            "rank": rank_disp,
            "opt_id": o["option_id"],
            "summary": c.get("summary", ""),
            "detail": c.get("detail", ""),
            "pros": " | ".join(c.get("pros", [])),
            "cons": " | ".join(c.get("cons", [])),
            "scores": sc_vals,
        })
    ths = "".join(f"<th>{n}</th>" for n in obj_names)
    body = ""
    for i, r in enumerate(rows):
        tds = "".join(f'<td class="score">{s}</td>' for s in r["scores"])
        body += f'''
      <tr>
        <td class="rank">{r["rank"]}</td>
        <td>{r["opt_id"]}</td>
        <td>{r["summary"]}</td>
        {tds}
        <td><button class="expand-btn" onclick="toggle({i})">展开 ▼</button></td>
      </tr>
      <tr class="detail-row" id="detail-{i}">
        <td colspan="{2+len(obj_names)}" class="detail-cell">
          <div><strong>详情：</strong>{r["detail"]}</div>
          <div><strong>优势：</strong>{r["pros"]}</div>
          <div><strong>劣势：</strong>{r["cons"]}</div>
        </td>
      </tr>'''
    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 1.5rem; background: #0f1419; color: #e6edf3; }}
    h1 {{ font-size: 1.25rem; color: #58a6ff; margin-bottom: 1rem; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    th, td {{ padding: 0.6rem 0.75rem; text-align: left; border-bottom: 1px solid #30363d; }}
    th {{ color: #8b949e; font-weight: 500; }}
    tr:hover {{ background: rgba(88,166,255,0.08); }}
    .rank {{ color: #58a6ff; font-weight: 600; }}
    .score {{ font-weight: 600; }}
    .detail-row {{ display: none; background: #161b22; }}
    .detail-row.show {{ display: table-row; }}
    .detail-cell {{ padding: 0.75rem 1rem; vertical-align: top; }}
    .detail-cell > div {{ margin-bottom: 0.5rem; font-size: 0.85rem; }}
    .detail-cell strong {{ color: #58a6ff; margin-right: 0.5rem; }}
    .expand-btn {{ background: none; border: none; color: #58a6ff; cursor: pointer; font-size: 0.85rem; }}
    .expand-btn:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>★ {title}</h1>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>选项</th>
        <th>概述</th>
        {ths}
        <th></th>
      </tr>
    </thead>
    <tbody>
{body}
    </tbody>
  </table>
  <script>
    function toggle(i) {{
      const row = document.getElementById('detail-' + i);
      const btn = row.previousElementSibling.querySelector('.expand-btn');
      row.classList.toggle('show');
      btn.textContent = row.classList.contains('show') ? '收起 ▲' : '展开 ▼';
    }}
  </script>
</body>
</html>'''
    Path(out_path).write_text(html, encoding="utf-8")
    print(f"已生成: {out_path}")

if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent
    gen_html(str(BASE / "fanfu_evolution_results.json"), str(BASE / "fanfu_pareto_table.html"), "王凡夫 Pareto 最优选项 — 研究生发展路径")
