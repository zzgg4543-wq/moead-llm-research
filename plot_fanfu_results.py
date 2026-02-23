#!/usr/bin/env python3
"""绘制王凡夫演化结果：排序雷达图 + 演化曲线"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE = Path(__file__).resolve().parent
with open(BASE / "fanfu_evolution_results.json", encoding="utf-8") as f:
    data = json.load(f)

objectives = [o["name"] for o in data["objectives"]]
# 去重得到唯一点
seen = {}
for opt in data["evaluated_options"]:
    oid = opt["option_id"]
    if oid not in seen:
        seen[oid] = opt
opts = list(seen.values())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左：雷达图（各 Pareto 选项的分数）
n_obj = len(objectives)
angles = np.linspace(0, 2 * np.pi, n_obj, endpoint=False).tolist()
angles += angles[:1]
ax1 = plt.subplot(121, projection='polar')
for i, opt in enumerate(opts[:5]):  # 最多5个
    vals = [x * 10 for x in opt["scores"]]
    vals += vals[:1]
    ax1.plot(angles, vals, 'o-', linewidth=2, label=opt["option_id"])
    ax1.fill(angles, vals, alpha=0.15)
ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(objectives)
ax1.set_ylim(0, 10)
ax1.set_title("Pareto 最优选项 四维评分", pad=20)
ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

# 右：演化曲线
ax2 = plt.subplot(122)
gens = [g["gen"] for g in data["gen_stats"]]
for j, obj in enumerate(objectives):
    avgs = [g["avgs"][j] for g in data["gen_stats"]]
    ax2.plot(gens, avgs, 'o-', label=obj, linewidth=2)
ax2.set_xlabel("代数")
ax2.set_ylabel("种群均值 (0-10)")
ax2.set_title("各目标演化曲线")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks(gens)

plt.suptitle("王凡夫 研究生发展路径 MOEA/D 演化结果", fontsize=14, y=1.02)
plt.tight_layout()
out = BASE / "fanfu_evolution_chart.png"
plt.savefig(out, dpi=120, bbox_inches='tight')
print(f"已保存: {out}")
plt.close()
