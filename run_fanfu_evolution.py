#!/usr/bin/env python3
"""
王凡夫 研究生三年发展路径 MOEA/D 演化
（赵哲为参考信息，用于生成更贴合凡夫且有借鉴意义的选项）
"""
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from moead_engine import run

# 王凡夫 profile（含学长赵哲作为参考）
PROFILE = """王凡夫，兰州大学大四，保研至中国科学技术大学攻读研究生。
· 本科参与发表 2 篇 NeurIPS（CCF-A）合作论文。
· 优势：诚实、负责、合作意识强、待人接物有分寸，布置任务会尽力完成、无歪心思。
· 劣势：思维偏窄、论文读得少、了解不够细致、有时想法空洞、过于谨慎、易思虑过多错过机会。
· 读研目标：多发论文、真正懂科研、读研期间自给自足不需问家里要钱。
· 长远规划：读研→海外博士→山东省内（如青岛）高校任教；备选：大厂人才计划、做有影响力工作。
· 参考：学长赵哲，USTC+CityU 博士，U2E/Eureka、NIPS/ICML 一作，NTU/Stanford 合作，可作为发展路径参考。"""

OBJECTIVES = [
    {"name": "稳定性", "definition": "保下限：至少 2 篇 CCF-A 一作或进大厂人才计划"},
    {"name": "上限", "definition": "海外博士、跟学长创业/任教、最终回国"},
    {"name": "自主性", "definition": "有可支配收入覆盖日常开支"},
    {"name": "交涉范围", "definition": "不因新机会破坏与当前学长的合作（硬约束）"},
]

CONFIG = {
    "pop_size": 16,
    "n_gens": 6,
    "n_crossover_pairs": 8,
    "p_mutation": 0.45,
    "n_expand": 5,
    "n_neighbors": 6,
    "constraints": ["不因新机会破坏与当前学长的合作"],
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
    print("=== 王凡夫 研究生三年发展路径 MOEA/D 演化 ===\n")
    res = run(
        profile=PROFILE,
        domain="研究生三年发展路径",
        objectives=OBJECTIVES,
        config=config,
        api_key=api_key,
        base_url=base_url.rstrip("/") if or_key else base_url,
        progress_callback=lambda m: print(m),
    )
    print("\n--- Pareto 最优排序 ---")
    seen = set()
    for i, opt_id in enumerate(res["pareto_ranking"], 1):
        if opt_id in seen:
            continue
        seen.add(opt_id)
        for ind in res["evaluated_options"]:
            if ind["option_id"] == opt_id:
                c = ind["content"]
                s = c.get("summary", str(c)[:80])
                sc = ind.get("scores", [])
                sc_str = " ".join(f"{x*10:.1f}" for x in sc)
                print(f"{i}. {opt_id}")
                print(f"   {s}")
                print(f"   稳定性 上限 自主性 交涉={sc_str}")
                if c.get("detail"):
                    print(f"   详情: {c['detail'][:120]}...")
                break
    print("\n--- 完整结果已保存 ---")
    import json
    out = Path(__file__).resolve().parent / "fanfu_evolution_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"→ {out}")

if __name__ == "__main__":
    main()
