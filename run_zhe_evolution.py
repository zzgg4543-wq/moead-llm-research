#!/usr/bin/env python3
"""
赵哲（Zhe Zhao）博后/职业选择 MOEA/D 演化 - 命令行运行
"""
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from moead_engine import run

# 赵哲 profile（来自 moead_career_exp / moead_career_results）
PROFILE = """Zhe Zhao（赵哲），USTC+CityU HK 联合培养博士。
· 师从张青富系统学习多目标优化与进化算法。
· 开发 U2E 自动算法进化系统（早于 Alpha Evolve），在组学、PDE 等领域超越人类专家；探索矩阵乘法等底层优化。
· 斯坦福期间主导 Eureka 科研论文撰写系统，将假设与已有研究验证后自动生成可投稿论文。
· 一作：NIPS 2024/2025, ICML 2024/2025, AAAI 2024/2025, TKDE 等；长尾学习、图神经网络、因果图网络。
· 已有合作：NTU 安波(2025.3-7)、Stanford 仇夏婕 Qiu(2025.7-2026.6)。"""

OBJECTIVES = [
    {"name": "契合", "definition": "与 MOEA/LLM/AI4Science 背景匹配"},
    {"name": "影响力", "definition": "顶会顶刊、领域认可"},
    {"name": "可行", "definition": "落地概率"},
    {"name": "Agent", "definition": "与 AI 趋势契合"},
    {"name": "上限", "definition": "职业/认知天花板"},
]

CONFIG = {
    "pop_size": 10,
    "n_gens": 3,
    "n_crossover_pairs": 4,
    "p_mutation": 0.3,
}

def main():
    or_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    or_base = os.environ.get("OPENROUTER_API_BASE") or os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
    or_model = os.environ.get("OPENROUTER_MODEL") or "openai/gpt-4o-mini"
    api_key = or_key or os.environ.get("DEEPSEEK_API_KEY", "").strip()
    base_url = or_base if or_key else (os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com")
    model = or_model if or_key else "deepseek-chat"
    if not api_key:
        print("错误：未配置 OPENROUTER_API_KEY 或 DEEPSEEK_API_KEY")
        return

    config = {**CONFIG, "model": model}
    print("=== 赵哲 博后/职业选择 MOEA/D 演化 ===\n")
    res = run(
        profile=PROFILE,
        domain="博后职业选择",
        objectives=OBJECTIVES,
        config=config,
        api_key=api_key,
        base_url=base_url.rstrip("/") if or_key else base_url,
        progress_callback=lambda m: print(m),
    )
    print("\n--- Pareto 最优排序 ---")
    for i, opt_id in enumerate(res["pareto_ranking"], 1):
        for ind in res["evaluated_options"]:
            if ind["option_id"] == opt_id:
                s = ind["content"].get("summary", str(ind["content"])[:80])
                sc = ind.get("scores", [])
                sc_str = " ".join(f"{x*10:.1f}" for x in sc[:5])
                print(f"{i}. {opt_id} | {s[:60]}... | 分={sc_str}")
                break
    print("\n--- 完整结果已保存 ---")
    import json
    out = Path(__file__).resolve().parent / "zhe_evolution_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"→ {out}")

if __name__ == "__main__":
    main()
